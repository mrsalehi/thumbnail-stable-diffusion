import json
import os
import shutil
from pathlib import Path
import random
from PIL import Image
import logging
import sys

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

CHANNEL_NAME_MAPPING = {
    "5-Minute Crafts": "5-minute-crafts",
    "BLACKPINK": "blackpink",
    "Cocomelon - Nursery Rhymes": "cocomelon-nursery-rhymes",
    "HYBE LABELS": "hybe-labels",
    "✿ Kids Diana Show": "kids-diana-show",
    "MrBeast6000": "mrbeast6000",
    "setindia": "setindia",
    "Sony SAB": "sony-sab",
    "WWE": "wwe",
    "Zee TV": "zee-tv",
    "BANGTANTV": "bangtantv",
    "Canal KondZilla": "canal-kondzilla",
    "Goldmines": "goldmines",
    "Justin Bieber": "justin-bieber",
    "Like Nastya": "like-nastya",
    "PewDiePie": "pewdiepie",
    "Shemaroo Filmi Gaane": "shemaroo-filmi-gaane",
    "Vlad and Niki": "vlad-and-niki",
    "Zee Music Company": "zee-music-company"
}


# create a function that read the mappings of old folder name to new one from the json file and rename the folders accordingly
def rename_folders(root_path):
    """
    rename the channel directories names from channel names to mapped names
    """
    dirs = Path(root_path).iterdir()

    for dir_ in dirs:
        new_dir_ = Path(dir_.parent) / CHANNEL_NAME_MAPPING[dir_.name]
        shutil.move(dir_, new_dir_)



def copy_sample_files_for_sanity_check_finetune():
    root_dir_imgs = Path("/home/user/thumbnail-stable-diffusion/thumbnail-imgs")
    root_dir_titles = Path("/home/user/thumbnail-stable-diffusion/thumbnail-titles")

    dst_dir_imgs = Path("/mnt/disks/sdb/thumbnail-imgs")
    dst_dir_titles = Path("/mnt/disks/sdb/thumbnail-titles")

    for img_dir in root_dir_imgs.iterdir():
        img_dir_name = img_dir.name
        titles_dir = root_dir_titles / img_dir_name
        
        (dst_dir_imgs / img_dir_name).mkdir(parents=True)
        (dst_dir_titles / img_dir_name).mkdir(parents=True)

        # sample 10 images and copy them to the dst dir
        img_files = list(img_dir.glob("*.jpg"))
        sample_imgs = random.sample(img_files, 10)        

        for img_file in sample_imgs:
            shutil.copy(img_file, dst_dir_imgs / img_dir_name)
            shutil.copy(titles_dir / img_file.name.replace(".jpg", ".txt"), dst_dir_titles / img_dir_name)


def keep_valid_imgs():
    root_dir_imgs = Path("/home/user/stable-diffusion-webui/data/thumbnail-imgs")

    counter = 0
    for img_dir in root_dir_imgs.iterdir():
        img_files = list(img_dir.glob("*.jpg"))
        for img_file in img_files:
            try:
                img = Image.open(img_file)
            except:
                counter += 1
                print(img_file)
                os.remove(img_file)

    print("removed {} files".format(counter))


def remove_path(path, ignore_errors=True):
    path = Path(path)
    if path.is_file() or path.is_symlink():
        path.unlink(missing_ok=ignore_errors)
    elif path.is_dir():
        shutil.rmtree(path, ignore_errors=ignore_errors)
    else:
        import warnings

        warnings.warn(f"unknown path type: {path}, not removed!")


def handle_ckpt_files(ckpt_dir, pattern, num_keep):
    glob_checkpoints = [str(x) for x in Path(ckpt_dir).glob(pattern)]
    ordering_and_checkpoint_path = []
    for path in glob_checkpoints:
        ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    num_ckpts = len(checkpoints_sorted)
    if num_ckpts <= num_keep:
        return
    checkpoints_to_be_deleted = checkpoints_sorted[: num_ckpts - num_keep]
    for checkpoint in checkpoints_to_be_deleted:
        remove_path(checkpoint)


def save_states(accelerator, step, ckpt_save_dir, num_kept_ckpt, wandb_id=None, text_encoder=None, vae=None, unet=None, tokenizer=None):
    accelerator.wait_for_everyone()
    logger.info("process {} is saving states".format(accelerator.process_index))
    ckpt_save_dir = Path(ckpt_save_dir)
    ckpt_file = ckpt_save_dir / "checkpoint.txt"
    # ckpt_str = "best-ckpt" if is_best else f"step-{step}"
    ckpt_str = f"step-{step}"
    wandb_id_file = ckpt_save_dir / "wandb-id.txt"
    current_save_dir = ckpt_save_dir / ckpt_str
    
    if (current_save_dir).exists():
        return

    with accelerator.main_process_first():
        current_save_dir.mkdir(parents=True, exist_ok=True)
        (current_save_dir / "text_encoder").mkdir(parents=True, exist_ok=True)
        (current_save_dir / "vae").mkdir(parents=True, exist_ok=True)
        (current_save_dir / "unet").mkdir(parents=True, exist_ok=True)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        ckpt_file.write_text(ckpt_str)
        if wandb_id:
            wandb_id_file.write_text(wandb_id)
    accelerator.save_state(current_save_dir)

    # this is silly but save_state also saves the models weights and I am revmoing them as I will be saving them later myself 
    if accelerator.is_main_process:
        try:
            os.remove(current_save_dir / "pytorch_model.bin")
            os.remove(current_save_dir / "pytorch_model_1.bin")
            os.remove(current_save_dir / "pytorch_model_2.bin")
        except:
            pass

    if text_encoder is not None:
        accelerator.wait_for_everyone()
        unwrapped_text_encoder = accelerator.unwrap_model(text_encoder)
        unwrapped_vae = accelerator.unwrap_model(vae)
        unwrapped_unet = accelerator.unwrap_model(unet)

        unwrapped_text_encoder.config.save_pretrained(current_save_dir / "text_encoder")  # save config 
 
        unwrapped_text_encoder.save_pretrained(
            current_save_dir / "text_encoder",
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(text_encoder),
        )
        unwrapped_vae.save_pretrained(
            current_save_dir / "vae",
            save_function=accelerator.save,
        )
        unwrapped_unet.save_pretrained(
            current_save_dir / "unet",
            save_function=accelerator.save,
        )
        tokenizer.save_pretrained(
            current_save_dir / "tokenizer",
            save_function=accelerator.save)

    # delete old ckpt files
    if accelerator.is_main_process:
        handle_ckpt_files(ckpt_save_dir, "step-*", num_kept_ckpt)


if __name__ == "__main__":
    keep_valid_imgs()