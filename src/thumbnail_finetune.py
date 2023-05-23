import argparse
import itertools
import math
import os
import random
from pathlib import Path
from typing import Optional
from contextlib import nullcontext
import shutil

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
import wandb
# from data import download_blob

import PIL
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from huggingface_hub import HfFolder, Repository, whoami
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from utils import save_states
import logging
import sys


logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--train_data_dir_image", type=str, default=None, required=True, help="The parent folder containing the image training data."
    )
    parser.add_argument(
        "--train_data_dir_text", type=str, default=None, required=True, help="The parent folder containing the text training data."
    )
    
    ######################################################################
    # We are fine tuning so we should NOT freeze the params of the models
    ######################################################################
    # parser.add_argument(
    #     "--placeholder_token",
    #     type=str,
    #     default=None,
    #     required=True,
    #     help="A token to use as a placeholder for the concept.",
    # )

    # parser.add_argument(
    #     "--initializer_token", type=str, default=None, required=True, help="A token to use as initializer word."
    # )
    ######################################################################
    # The following option is for textual inversion 
    ######################################################################
    # parser.add_argument("--learnable_property", type=str, default="object", help="Choose between 'object' and 'style'")

    ######################################################################
    # What the hell is repeats?
    ######################################################################
    # parser.add_argument("--repeats", type=int, default=100, help="How many times to repeat the training data.")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/user/thumbnail-stable-diffusion/runs",
        help="Root folder to save the checkpoints and the logs.",
    )
    parser.add_argument(
        "--resume_ckpt_dir",
        type=str,
        help="ckpt dir to resume training from.",
    )

    # parser.add_argument(
    #     "--run_name",
    #     required=True,
    #     help="subdirectory of output_dir where checkpoints and logs will be saved",
    # )

    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution"
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )

    parser.add_argument(
        "--save_every_steps",
        type=int,
        default=10,
        help="Save checkpoint every X steps.",
    )

    # add num_workers argument
    parser.add_argument(
        "--num_workers",
        type=int,
        default=5,
        help="Number of workers for the dataloader.",
    )

    parser.add_argument(
        "--train_vae",
        action="store_true",
        help="Whether to train the VAE or not. If not, the VAE will be frozen."
    )

    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder or not. If not, the text encoder will be frozen."
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--use_auth_token",
        action="store_true",
        default=False,
        help=(
            "Will use the token generated when running `huggingface-cli login` (necessary to use this script with"
            " private models)."
        ),
    )
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    ######################################################################
    # train_data_dir has changed to train_data_dir_image and text
    ######################################################################
    # if args.train_data_dir is None:
    #     raise ValueError("You must specify a train data directory.")

    return args


imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]


class ThumbnailDataset(Dataset):
    def __init__(
        self,
        data_root_image,
        data_root_text,
        # title,  # this is the title of the video
        tokenizer,
        # learnable_property="object",  # [object, style]
        size=512,
        # repeats=100,
        interpolation="bicubic",
        flip_p=0.5,
        set="train",
        center_crop=False
    ):
        self.data_root_image = data_root_image
        self.data_root_text = data_root_text
        self.tokenizer = tokenizer
        # self.learnable_property = learnable_property
        self.size = size
        self.center_crop = center_crop
        self.flip_p = flip_p

        # self.image_paths = [os.path.join(self.data_root_image, file_path) for file_path in os.listdir(self.data_root_image)]
        self.image_paths = list(Path(self.data_root_image).rglob("*.jpg"))
        # self.text_paths = [os.path.join(self.data_root_text, file_path) for file_path in os.listdir(self.data_root_text)]
        self.text_paths = list(Path(self.data_root_text).rglob("*.txt"))

        # load all of the text into memory
        self.texts = {}
        for text_path in self.text_paths:
            with open(text_path, "r") as f:
                self.texts[os.path.basename(text_path).split(".")[0]] = f.read()
        # self.texts = dict()
        # video_ids = [Path(image_path).stem for image_path in self.image_paths]
        # for video_id in video_ids:
        #     self.texts[video_id] = title  

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        ######################################################################
        # repeats is just for textual inversion??
        ######################################################################
        # if set == "train":
        #     self._length = self.num_images * repeats

        self.interpolation = {
            "linear": PIL.Image.LINEAR,
            "bilinear": PIL.Image.Resampling.BILINEAR,
            "bicubic": PIL.Image.Resampling.BICUBIC,
            "lanczos": PIL.Image.Resampling.LANCZOS,
        }[interpolation]

        # self.templates = imagenet_style_templates_small if learnable_property == "style" else imagenet_templates_small
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)
        
    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        # image_path = self.image_paths[index % self.num_images]
        image_path = self.image_paths[index]
        thumbnail_id = image_path.stem
        image = Image.open(image_path)

        if not image.mode == "RGB":
            image = image.convert("RGB")

        # placeholder_string = self.placeholder_token
        # text = random.choice(self.templates).format(placeholder_string)
        text = self.texts[thumbnail_id]

        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w, = (
                img.shape[0],
                img.shape[1],
            )
            img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        return example


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def freeze_params(params):
    for param in params:
        param.requires_grad = False


def main():
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=["tensorboard", "wandb"],
        logging_dir=logging_dir,
    )
    if accelerator.is_main_process: 
        Path(logging_dir).mkdir(parents=True, exist_ok=True)

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer and add the placeholder token as a additional special token
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer", use_auth_token=args.use_auth_token
        )

    ######################################################################
    # placeholder token is just used in textual inversion
    ######################################################################
    # # Add the placeholder token in tokenizer
    # num_added_tokens = tokenizer.add_tokens(args.placeholder_token)
    # if num_added_tokens == 0:
    #     raise ValueError(
    #         f"The tokenizer already contains the token {args.placeholder_token}. Please pass a different"
    #         " `placeholder_token` that is not already in the tokenizer."
    #     )

    ######################################################################
    # no need for initializer token in fine tuning
    ######################################################################
    # Convert the initializer_token, placeholder_token to ids
    # token_ids = tokenizer.encode(args.initializer_token, add_special_tokens=False)
    # # Check if initializer_token is a single token or a sequence of tokens
    # if len(token_ids) > 1:
    #     raise ValueError("The initializer token must be a single token.")

    # initializer_token_id = token_ids[0]

    ######################################################################
    # placeholder token is just used in textual inversion 
    ######################################################################
    # placeholder_token_id = tokenizer.convert_tokens_to_ids(args.placeholder_token)

    # Load models and create wrapper for stable diffusion

    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", use_auth_token=args.use_auth_token
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", use_auth_token=args.use_auth_token
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", use_auth_token=args.use_auth_token
    )

    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    ######################################################################
    # No need to resize as we don't have new tokens 
    ######################################################################
    # text_encoder.resize_token_embeddings(len(tokenizer))

    # Initialise the newly added placeholder token with the embeddings of the initializer token
    # token_embeds = text_encoder.get_input_embeddings().weight.data

    ######################################################################
    # placeholder token is just used in textual inversion
    ######################################################################
    # token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]

    
    ######################################################################
    # We are fine tuning so we should NOT freeze the params of the models
    ######################################################################
    
    # # Freeze vae and unet
    # freeze_params(vae.parameters())
    # freeze_params(unet.parameters())
    # # Freeze all parameters except for the token embeddings in text encoder
    # params_to_freeze = itertools.chain(
    #     text_encoder.text_model.encoder.parameters(),
    #     text_encoder.text_model.final_layer_norm.parameters(),
    #     text_encoder.text_model.embeddings.position_embedding.parameters(),
    # )
    # freeze_params(params_to_freeze)

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    params_to_train = list(unet.parameters())
    if args.train_vae:
        params_to_train += list(vae.parameters())
    if args.train_text_encoder:
        params_to_train += list(text_encoder.text_model.encoder.parameters())

    optimizer = torch.optim.AdamW(
        # text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
        params_to_train,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # TODO (patil-suraj): laod scheduler using args
    noise_scheduler = DDPMScheduler(
        # beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, tensor_format="pt"
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
    )

    # FIXME: Change the train dataset
    # train_dataset = TextualInversionDataset(
    #     data_root=args.train_data_dir,
    #     tokenizer=tokenizer,
    #     size=args.resolution,
    #     placeholder_token=args.placeholder_token,
    #     repeats=args.repeats,
    #     learnable_property=args.learnable_property,
    #     center_crop=args.center_crop,
    #     set="train",
    # )
    train_dataset = ThumbnailDataset(
        data_root_image=args.train_data_dir_image,
        # title="NEW CoComelon Show! Emmy's Sick Song | CoComelon Animal Time | Animals for Kids",
        data_root_text=args.train_data_dir_text,
        tokenizer=tokenizer,
        # learnable_property="object",  # [object, style]  # just for textual inversion not used in fine tuning 
        size=512,
        # repeats=100,  # just for textual inversion with a few samples (I guess??) not used in fine tuning
        interpolation="bicubic",
        flip_p=0.5,
        set="train",
        center_crop=False,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.train_batch_size, 
        num_workers=args.num_workers,
        shuffle=True)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    ######################################################################
    # All the models should be prepared using accelerator.prepare
    ######################################################################
    # text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    #     text_encoder, optimizer, train_dataloader, lr_scheduler
    # )

    vae, text_encoder, unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        vae, text_encoder, unet, optimizer, train_dataloader, lr_scheduler
    )
    # vae, unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    #     vae, unet, text_encoder, optimizer, train_dataloader, lr_scheduler
    # )

    # # Move vae and unet to device
    # vae.to(accelerator.device)
    # text_encoder.to(accelerator.device)
    # unet.to(accelerator.device)

    ######################################################################
    # We are fine tuning everything so everything is in train mode
    ######################################################################
    # Keep vae and unet in eval model as we don't train these
    if args.train_vae: 
        vae.train()
    else:
        vae.eval()

    if args.train_text_encoder: 
        text_encoder.train()
    else:
        text_encoder.eval()


    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    wandbid_file = Path(logging_dir) / "wandb-id.txt"
    args.wandb_id = wandbid_file.read_text().strip() if wandbid_file.exists() else wandb.util.generate_id()
    if accelerator.is_main_process:
        accelerator.init_trackers(project_name="thumbnail_stable_diffusion", config=vars(args), init_kwargs={"wandb": {"id": args.wandb_id, "name": "finetune_kids_20_channels"}})

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    if os.path.exists(os.path.join(logging_dir, "checkpoint.txt")):
        step_dirname = (Path(logging_dir) / "checkpoint.txt").read_text().strip()
        args.resume_ckpt_dir = os.path.join(logging_dir, step_dirname)

    if args.resume_ckpt_dir is not None:
        if accelerator.is_local_main_process:
            shutil.copy(
                os.path.join(args.resume_ckpt_dir, 'vae', 'diffusion_pytorch_model.bin'), 
                os.path.join(args.resume_ckpt_dir, 'pytorch_model.bin')
                )
            
            shutil.copy(
                os.path.join(args.resume_ckpt_dir, 'unet', 'diffusion_pytorch_model.bin'), 
                os.path.join(args.resume_ckpt_dir, 'pytorch_model_2.bin')
                )
            shutil.copy(
                os.path.join(args.resume_ckpt_dir, 'text_encoder', 'pytorch_model.bin'), 
                os.path.join(args.resume_ckpt_dir, 'pytorch_model_1.bin')
                )
        accelerator.wait_for_everyone()
        accelerator.load_state(args.resume_ckpt_dir)
        resume_step = int(Path(args.resume_ckpt_dir).name.replace("step-", ""))
        start_global_step = resume_step
        global_step = start_global_step
        trained_epochs = (resume_step * args.gradient_accumulation_steps) // len(train_dataloader)
        resume_step = (resume_step * args.gradient_accumulation_steps) - (trained_epochs * len(train_dataloader))

        logger.info(f"====>Resuming training from step {resume_step} of epoch {trained_epochs}")
        logger.info(f"====>global_step: {global_step}")

        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            try:
                # not sure why in one of the urns I got error that the following files do not exist
                os.remove(os.path.join(args.resume_ckpt_dir, 'pytorch_model_1.bin'))
                os.remove(os.path.join(args.resume_ckpt_dir, 'pytorch_model_2.bin'))
                os.remove(os.path.join(args.resume_ckpt_dir, 'pytorch_model.bin'))
            except:
                pass
        accelerator.wait_for_everyone()
    else:
        start_global_step = 0
        global_step = start_global_step
        trained_epochs = 0
        resume_step = None

    if accelerator.is_main_process:
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    # progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_main_process)
    progress_bar.set_description("Steps")
    # global_step = 0

    for epoch in range(trained_epochs, args.num_train_epochs):
        accelerator.print(f"Epoch {epoch}")
        # text_encoder.train()
        # vae.train()
        # unet will always be trained
        unet.train()
        for step, batch in enumerate(train_dataloader):
            if resume_step is not None and epoch == trained_epochs:
                progress_bar.set_description(f"{epoch=}, {step=}, skip {resume_step=} steps")
                if step < resume_step:
                    continue
                elif step == resume_step:
                    progress_bar.update(global_step)
            with accelerator.accumulate(unet): 
                # conditional context manager that will accumulate gradients for vae if args.train_vae is true
                with accelerator.accumulate(vae) if args.train_vae else nullcontext():
                    with accelerator.accumulate(text_encoder) if args.train_text_encoder else nullcontext():
                        # Convert images to latent space
                        ######################################################################
                        # should not detach the latents
                        ######################################################################
                        if hasattr(vae, "module"):
                            vae = vae.module

                        if args.train_vae:
                            # latents = vae.module.encode(batch["pixel_values"]).latent_dist.sample()
                            latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
                        else:
                            latents = vae.encode(batch["pixel_values"]).latent_dist.sample().detach()

                        # latents = vae.encode(batch["pixel_values"]).sample()
                        latents = latents * 0.18215

                        # Sample noise that we'll add to the latents
                        noise = torch.randn(latents.shape).to(latents.device)
                        bsz = latents.shape[0]
                        # Sample a random timestep for each image
                        timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device).long()

                        # Add noise to the latents according to the noise magnitude at each timestep
                        # (this is the forward diffusion process)
                        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                        # Get the text embedding for conditioning
                        # encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                        if args.train_text_encoder:
                            encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                        else:
                            encoder_hidden_states = text_encoder(batch["input_ids"])[0].detach()

                        # Predict the noise residual
                        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                        # noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states)['sample']

                        if args.mixed_precision == "no":
                            loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
                        else:
                            loss = F.mse_loss(noise_pred['sample'], noise, reduction="none").mean([1, 2, 3]).mean()
                        accelerator.backward(loss)


                        # for name, param in unet.named_parameters():
                        #     if param.grad.shape == torch.Size([320, 320, 1, 1]):
                        #         print(name)


                        ######################################################################
                        # No need for embeddings gradients
                        ######################################################################
                        # Zero out the gradients for all token embeddings except the newly added
                        # embeddings for the concept, as we only want to optimize the concept embeddings
                        # if accelerator.num_processes > 1:
                        #     grads = text_encoder.module.get_input_embeddings().weight.grad
                        # else:
                        #     grads = text_encoder.get_input_embeddings().weight.grad
                        # # Get the index for tokens that we want to zero the grads for
                        # index_grads_to_zero = torch.arange(len(tokenizer)) != placeholder_token_id
                        # grads.data[index_grads_to_zero, :] = grads.data[index_grads_to_zero, :].fill_(0)

                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                total_number_of_samples = global_step * args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
            
            if global_step % args.save_every_steps == 0 and global_step > start_global_step:
                # if accelerator.is_main_process:
                #     accelerator.print(f"global step {global_step}: Trained on {total_number_of_samples} samples so far...")
                # logger.info(f"process rank {accelerator.process_index}: saving checkpoint at step {global_step}")
                save_states(
                    accelerator=accelerator, 
                    step=global_step, 
                    ckpt_save_dir=logging_dir, 
                    num_kept_ckpt=10, 
                    wandb_id=args.wandb_id, 
                    text_encoder=text_encoder, 
                    vae=vae, 
                    unet=unet,
                    tokenizer=tokenizer
                    )
                # except:
                #     if accelerator.is_main_process:
                #         accelerator.print("Exception! Perhaps starting from a checkpoint. You can ignore this for now.")

            logs = {
                "loss": loss.detach().item(), 
                "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            logs.update({
                "latent_norm": latents.norm().item(), 
                "text_hidden_states_norm": encoder_hidden_states.norm().item()})
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()

    # Create the pipeline using using the trained modules and save it.
    if accelerator.is_main_process:
        pass
        # pipeline = StableDiffusionPipeline(
        #     text_encoder=accelerator.unwrap_model(text_encoder),
        #     vae=vae,
        #     unet=unet,
        #     tokenizer=tokenizer,
        #     scheduler=PNDMScheduler(
        #         beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", skip_prk_steps=True
        #     ),
        #     safety_checker=StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker"),
        #     feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),
        # )
        # pipeline.save_pretrained(args.output_dir)

        ######################################################################
        # placeholder token is just used in textual inversion
        ######################################################################
        # Also save the newly trained embeddings
        # learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[placeholder_token_id]
        # learned_embeds_dict = {args.placeholder_token: learned_embeds.detach().cpu()}
        # torch.save(learned_embeds_dict, os.path.join(args.output_dir, "learned_embeds.bin"))

        # if args.push_to_hub:
        #     repo.push_to_hub(
        #         args, pipeline, repo, commit_message="End of training", blocking=False, auto_lfs_prune=True
        #     )

    accelerator.end_training()


if __name__ == "__main__":
    main()