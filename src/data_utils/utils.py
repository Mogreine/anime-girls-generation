import os
from typing import List

import numpy as np
import torch
from PIL import Image


# Loads images from a specified folder and returns a list of images
def load_images(folder: str) -> List[Image.Image]:
    images = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
            img.load()

    return images


# Resizes image to a specified size
def resize_image(image: Image, size: tuple) -> Image.Image:
    return image.resize(size, Image.BICUBIC)


def img_to_tensor(im):
    return torch.tensor(np.array(im.convert("RGB")) / 255).permute(2, 0, 1).unsqueeze(0) * 2 - 1


def tensor_to_image(t):
    return Image.fromarray(np.array(((t.squeeze().permute(1, 2, 0) + 1) * 127.5).clip(0, 255)).astype(np.uint8))
