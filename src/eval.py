

from PIL import Image
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from pathlib import Path
import torch
import multiprocessing
from multiprocessing import Pool
from itertools import chain
from modules import shared
import numpy as np

import torch.multiprocessing as mp
import os
import threading

from modules.paths import script_path

import signal

from modules.shared import opts, cmd_opts, state
import modules.shared as shared
# import modules.ui
import modules.scripts
import modules.sd_hijack
import modules.codeformer_model
import modules.gfpgan_model
import modules.face_restoration
import modules.realesrgan_model as realesrgan
import modules.esrgan_model as esrgan
import modules.ldsr_model as ldsr
import modules.extras
import modules.lowvram
import modules.txt2img
import modules.img2img
import modules.swinir as swinir
import modules.sd_models


modules.codeformer_model.setup_codeformer()
modules.gfpgan_model.setup_gfpgan()
shared.face_restorers.append(modules.face_restoration.FaceRestoration())

# esrgan.load_models(cmd_opts.esrgan_models_path)
# swinir.load_models(cmd_opts.swinir_models_path)
# realesrgan.setup_realesrgan()
ldsr.add_lsdr()
queue_lock = threading.Lock()

TOKEN = "YOUR_AUTH_TOKEN"


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


def load_pipeline(step: int):
    if step == 0:
        # load the original model
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=TOKEN)
    else:
        # fine tuned models
        model_path = f"/user/disks/sdc/runs/finetune_cocomelon/step-{step}"
        text_encoder = CLIPTextModel.from_pretrained(
            model_path, subfolder="text_encoder"
        )
        vae = AutoencoderKL.from_pretrained(
            model_path, subfolder="vae"
        )
        unet = UNet2DConditionModel.from_pretrained(
            model_path, subfolder="unet"
        )

        tokenizer = CLIPTokenizer.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="tokenizer", use_auth_token=TOKEN
        )

        pipe = StableDiffusionPipeline(
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            tokenizer=tokenizer,
            scheduler=PNDMScheduler(
                beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", skip_prk_steps=True
                ),
            # safety_checker=StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker"),
            feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),
            )

    return pipe


def save_grid(grid, first_step, last_step, prompt, root_save_dir):
    counter = 0
    root_dir = root_save_dir
    
    while True:
        counter += 1
        fname = f"{prompt}_step_{first_step}_{last_step}_{counter}.png"
        path =  root_dir / fname
        if not path.exists():
            grid.save(str(root_dir / fname))
            break


def restore_faces(np_image):
    face_restorers = [x for x in shared.face_restorers if x.name() == shared.opts.face_restoration_model or shared.opts.face_restoration_model is None]
    if len(face_restorers) == 0:
        return np_image

    face_restorer = face_restorers[0]

    return face_restorer.restore(np_image)


# let's try to make the generate faster by running on multiple GPUs
def generate_sample_single_step_single_gpu(root_model_dir, gpu_idx, step, prompt):
    """
    takes a gpu index and a step number (number of fine tuning steps) and generates samples
    """
    device = f"cuda:{gpu_idx}"
    root_model_dir = Path(root_model_dir)

    if step == 0:
        # load the original model
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=TOKEN)
    else:
        # fine tuned models
        model_path = str(root_model_dir / f"step-{step}")

        text_encoder = CLIPTextModel.from_pretrained(
            model_path, subfolder="text_encoder"
        )
        vae = AutoencoderKL.from_pretrained(
            model_path, subfolder="vae"
        )
        unet = UNet2DConditionModel.from_pretrained(
            model_path, subfolder="unet"
        )

        tokenizer = CLIPTokenizer.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="tokenizer", use_auth_token=TOKEN
        )
        
        # placeholder_token_id = 49408
        # placeholder_token = "<style>"

        # tokenizer = CLIPTokenizer.from_pretrained(
        #     model_path, subfolder="tokenizer",
        # )

        # text_encoder.get_input_embeddings().weight.requires_grad = False
        # text_encoder.get_input_embeddings().weight[placeholder_token_id] = torch.load(root_model_dir / "learned_embeds.bin")[placeholder_token]

        pipe = StableDiffusionPipeline(
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            tokenizer=tokenizer,
            scheduler=PNDMScheduler(
                beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", skip_prk_steps=True
                ),
            # safety_checker=StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker"),
            feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),
            )
    
    pipe.to(device)
    # do a for loop and generate images five by five
    images = []
    for i in range(0, len(prompt), 5):
        print(f"Generating images {i} to {i+5}")
        images.extend(pipe(prompt[i:i+5]).images) 
    
    # images = pipe(prompt).images
    return (images, step)


def generate_samples_multi_gpu(root_model_dir, steps, prompt, root_save_dir):
    tot_images = []

    # a pool of 4 processes where each one call generate_sample_single_step_single_gpu and gets the samples in return
    with Pool(processes=len(steps)) as pool:
        results = pool.starmap(generate_sample_single_step_single_gpu, [(root_model_dir, i, step, prompt) for i, step in enumerate(steps)])

    results = sorted(results, key=lambda x: x[1])

    tot_images = [images for images, _ in results]
    tot_images = list(chain(*tot_images))
    print(len(tot_images))
    # np_imgs = [np.array(img) for img in tot_images]

    # np_imgs_restored = [restore_faces(np_img) for np_img in np_imgs]
    # pil_imgs_restored = [Image.fromarray(np_img) for np_img in np_imgs_restored]

    # save all the restored pil images one by one in root_save_dir
    # for i, pil_img in enumerate(pil_imgs_restored):
    #     pil_img.save(str(root_save_dir / f"image_{i}.png"))


    # grid_before_restore = image_grid(tot_images, rows=len(steps), cols=len(prompt))
    grid = image_grid(tot_images, rows=4, cols=5)
    # grid_after_restore = image_grid(pil_imgs_restored, rows=8, cols=10)

    # save_grid(grid_before_restore, steps[0], steps[-1], prompt[0] + "_before_restore", root_save_dir)
    save_grid(grid, steps[0], steps[-1], prompt[0], root_save_dir)
    # save_grid(grid_after_restore, steps[0], steps[-1], prompt[0] + "_after_restore", root_save_dir)


if __name__ == "__main__":
    # steps = [1000*i for i in range(1, 5)]
    torch.multiprocessing.set_start_method('spawn')
    # steps = [5000, 5000, 5000, 5000]
    steps = [2000, 3000, 4000, 5000]
    # steps = [0, 0, 0, 0]
    num_samples = 5
    num_trial_repeats = 1
    # prompt_set = ["playing with a kite", "running in a green garden", "playing with other kids", "swimming in the pool", "sitting on a chair", "playing soccer", "jumping in the air"] 
    # prompt_set = ["I survived a plane crash in the style of <style>"]
    # prompt_set = ["a realistic high quality accurate photo of nastya wearing spacesuit in the space with planet earth in background"]
    # prompt_set = ["a highly accurate realistic high quality photo of nastya in a green garden, sharp focus, golden ratio, extreme detail"]
    prompt_set = ["cocomelon is sick"]
    root_save_dir = "/home/user/thumbnail-stable-diffusion/data/cocomelon_video"
    Path(root_save_dir).mkdir(parents=True, exist_ok=True)

    for i in range(num_trial_repeats):
        for prompt in prompt_set:
            prompt = [prompt] * num_samples
            generate_samples_multi_gpu(
                root_save_dir=Path(root_save_dir),
                root_model_dir="/user/disks/sdc/runs/ftune_video_cocomelon",
                steps=steps,
                prompt=prompt)