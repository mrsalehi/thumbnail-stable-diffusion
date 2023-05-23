import torch
import timm
import cv2
from pathlib import Path
import os
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

def load_img(img_path):
    """
    Load image using cv2 and convert to torch tensor
    """
    # img = cv2.imread(str(img_path))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # # resize image to 360,480
    # img = cv2.resize(img, (224, 224))
    # # convert image into torch tensor
    # img = torch.from_numpy(img)
    img = Image.open(img_path)
    tr  = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = tr(img)
    return img


def find_most_similar_img(model, query_img_path, train_img_root_dir):
    # load query image
    query_img = load_img(query_img_path)

    with torch.no_grad(): 
        query_img = model(query_img.unsqueeze(0).cuda())

    # load train images
    train_imgs_paths = list(Path(train_img_root_dir).rglob("*.jpg"))
    train_imgs = [load_img(img_path) for img_path in train_imgs_paths]

    with torch.no_grad():
        for i in tqdm(range(len(train_imgs))):
            train_imgs[i] = model(train_imgs[i].unsqueeze(0).cuda())

    train_imgs = torch.stack(train_imgs)

    # compute dot product between query image and train images
    # dot_products = torch.dot(query_img, train_imgs) 
    dot_products = torch.einsum("bd,d->b", train_imgs, query_img)

    # find the indices of top 5 most similar images
    top_5_similar_img_idxs = torch.topk(dot_products, k=5, dim=0).indices
 
    # get the path of the most similar image
    most_similar_imgs = [train_imgs_paths[idx] for idx in top_5_similar_img_idxs]
    print(most_similar_imgs)
    # copy the most similar image to the current directory
    for img_path in most_similar_imgs: 
        os.system(f"cp {img_path} .")


if __name__ == "__main__":
    model = timm.create_model('vit_huge_patch14_224_in21k', pretrained=True)
    del(model.pre_logits)
    del(model.head)
    model.eval()
    model.cuda()
    model.load_state_dict(torch.load("/home/user/mae_pretrain_vit_huge.pth")["model"], strict=True)

    find_most_similar_img(
        model=model,
        query_img_path="/home/user/thumbnail-stable-diffusion/data/cocomelon_ftuned_5000_steps/playing with baloons_step_5000_5000_1.png",
        train_img_root_dir="/user/disks/sdc/data/thumbnail-imgs/cocomelon-nursery-rhymes"
    )	