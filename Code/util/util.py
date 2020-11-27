"""This module contains simple helper functions """
import torch
import numpy as np
from PIL import Image
import os
from math import log10, sqrt

# PSNR 측정
def PSNR(original, compressed):
    """ PSNR 측정 함수 """
    mse = np.mean((original - compressed) ** 2)
    if (mse == 0):
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

# 텐서를 numpy 이미지 배열로 변환
def tensor2im(input_image, imtype=np.uint8):
    """ 텐서를 Numpy 배열로 변화 """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.data
        else:
            return input_image

        # 넘파이로 변환
        image_numpy = image_tensor[0].cpu().float().numpy()

        # 흑백 이미지 인경우 RGB로 변경
        if image_numpy.shape[0] == 1:
            image_numpy = np.tile(image_numpy, (3, 1, 1))

        # 학습을 위해 배열 위치 조절
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = input_image
    return image_numpy.astype(imtype)

# 이미지 저장
def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """ 이미지를 저장하는 함수 """
    # numpy 이미지 배열로부터 PIL Image 형식으로 변환
    image_pil = Image.fromarray(image_numpy)

    # 이미지의 너비와 높이를 저장
    h, w, _ = image_numpy.shape

    # 이미지 저장
    # aspect_ratio - ??
    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)