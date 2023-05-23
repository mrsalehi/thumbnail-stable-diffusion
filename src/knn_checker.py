from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os
import torch
from pathlib import Path


def load_img(img_path):
	"""
	Load image using cv2 and convert to torch tensor
	"""
	img = cv2.imread(str(img_path))
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	# resize image to 360,480
	img = cv2.resize(img, (360, 480))
	# convert image into torch tensor
	img = torch.from_numpy(img)

	return img


def find_most_similar_img(query_img_path, train_img_root_dir):
	# load query image
	query_img = load_img(query_img_path)

	# load train images
	train_imgs_paths = list(Path(train_img_root_dir).rglob("*.jpg"))
	train_imgs = [load_img(img_path) for img_path in train_imgs_paths]
	train_imgs = torch.stack(train_imgs)

	# compute dot product between query image and train images
	# dot_products = torch.dot(query_img, train_imgs) 
	dot_products = torch.einsum("bwhc,whc->b", train_imgs, query_img)
	# find the index of the most similar image
	most_similar_img_idx = torch.argmax(dot_products)
	# get the path of the most similar image
	most_similar_img_path = train_imgs_paths[most_similar_img_idx]
	print(most_similar_img_path)
	# copy the most similar image to the current directory
	os.system(f"cp {most_similar_img_path} .")


if __name__ == "__main__":
	find_most_similar_img(
		query_img_path="/home/user/thumbnail-stable-diffusion/data/cocomelon_ftuned_5000_steps/playing with baloons_step_5000_5000_1.png",
		train_img_root_dir="/user/disks/sdc/data/thumbnail-imgs/cocomelon-nursery-rhymes"
	)	