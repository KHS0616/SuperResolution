"""
데이터의 유형이 폴더인 경우의 데이터 셋 파일
"""

# os - 파일 관리 및 경로 설정을 위한 모듈
# numpy - 다양한 연산을 위한 모듈
# cv2, PIL.Image - 이미지를 관리하기 위한 모듈
import os
from os import listdir
import numpy as np
import cv2
from PIL import Image
from data.BaseDataset import BaseDataset

# 파이토치
import torch 
from torch.utils.data import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize, Normalize

# crop size를 계산하는 메소드
def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

# hr 이미지를 랜덤하게 자르는 메소드
def train_hr_transform(grayscale=False):
    if not grayscale:
        return Compose([
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))        
        ])
    else:
        return Compose([
            ToTensor(),
            Normalize((0.5, 0.5), (0.5, 0.5))
        ])

# 학습을 위해 업 스케일 요소만큼 학습 데이터를 다운 스케일링 한다.
# 해당 과정에서 랜덤한 부분을 자른다.
def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# 채널을 적절하게 조절하는 함수
def set_channel(*opt, n_channels=3):
    def _set_channel(img):
        # 이미지의 차원이 2차원인지 확인
        # 2차원인 경우는 RGB 이미지가 아닌 단일 이미지 - 채널의 개수가 1
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        c = img.shape[2]

        # 설정된 채널의 개수와 실제 채널의 개수가 다른경우 원인 해결
        if n_channels == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channels == 3 and c == 1:
            img = np.concatenate([img] * n_channels, 2)
        return img        
    return [_set_channel(a) for a in opt]

# 이미지를 Tensor로 변환하는 메소드
# 연속적인 수를 변환
def np2Tensor(*opt, rgb_range=255):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)

        return tensor

    return [_np2Tensor(a) for a in opt]

# 학습용 이미지 데이터 셋을 불러오는 클래스
class TrainDataset(BaseDataset):
    # 데이터 가공 및 변수 할당
    def __init__(self, opt):
        super().__init__(opt)
        # 파서로 부터 정보를 변수에 저장
        self.train_dataset_dir = os.path.join(opt.dataroot, "train")
        self.image_filenames = sorted(self.make_dataset(self.train_dataset_dir, opt.max_dataset_size))
        self.crop_size = opt.crop_size
        self.scale = opt.scale

        # crop_size 설정 및 hr, lr 이미지 설정을 위한 메소드 할당
        # self.hr_transform = self.get_transform(opt, grayscale=(opt.input_nc==1))
        # self.lr_transform = self.get_transform(opt, grayscale=(opt.input_nc==1))
        self.crop_size = calculate_valid_crop_size(self.crop_size, self.scale)
        self.hr_transform = train_hr_transform(grayscale=(opt.input_nc == 1))
        self.hr_semi_transform = self.get_transform(opt, grayscale=(opt.input_nc == 1), convert=False)
        self.lr_transform = train_lr_transform(self.crop_size, self.scale)

    # 학습 또는 평가를 진핼할 때마다 데이터를 배치사이즈 간격으로 반환한다.
    # 이미지를 rescale, randomcrop을 하고 반환
    # 이미지는 가공되는 과정에서 Tensor 데이터로 변환된다.
    def __getitem__(self, index):
        hr_image = self.hr_semi_transform(Image.open(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)
        hr_image = self.hr_transform(hr_image)
        return {'LR': lr_image, 'HR': hr_image, 'LR_paths': self.image_filenames[index], 'HR_paths': self.image_filenames[index]}

    def __len__(self):
        return len(self.image_filenames)

# 평가할 때 사용되는 Dataset 모듈
# 평가용 이미지 데이터 셋을 불러오는 클래스
class EvalDataset(BaseDataset):
    def __init__(self, opt):
        super(EvalDataset, self).__init__(opt)

        # 이미들의 경로를 저장한다.
        # scale - 업 스케일링 수치
        self.scale = opt.scale
        self.eval_dataset_dir = os.path.join(opt.dataroot, "eval")
        self.image_filenames = sorted(self.make_dataset(self.eval_dataset_dir, opt.max_dataset_size))

        # 이미지의 경로를 통해 이미지 파일인지 확인하고 저장

    def __getitem__(self, index):
        # 이미지 저장 및 크기 측정
        hr_image = Image.open(self.image_filenames[index])
        w, h = hr_image.size

        # 스케일 수치에 따른 crop_size 저장
        # crop_size = calculate_valid_crop_size(min(w, h), self.scale)

        # 입력 이미지를 스케일 요소에 따른 크기조절 메소드 할당
        lr_scale = Compose([Resize((h//self.scale, w//self.scale), interpolation=Image.BICUBIC), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        hr_scale = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        # lr이미지는 다운 스케일링한다.
        # hr이미지는 crop을 진행하고 다운스케일 직전에 저장한다.
        # hr_restore_img이미지는 lr로 다운 스케일 후 bicubic을 이용하여 hr이미지의 크기에 맞춘다.        
        lr_image = lr_scale(hr_image)
        hr_image = hr_scale(hr_image)
        #hr_restore_img = hr_scale(lr_image)
        # 가공한 이미지를 Tensor로 변환하여 반환한다.
        return {'LR': lr_image, 'HR': hr_image, 'LR_paths': self.image_filenames[index], 'HR_paths': self.image_filenames[index]}

    def __len__(self):
        return len(self.image_filenames)


# 테스트할 때 사용되는 Dataset 모듈
class TestDataset(Dataset):
    def __init__(self, opt, train=False, benchmark=False):
        # 파서로 부터 정보를 변수에 저장
        # test_dataset_dir - 학습에 사용될 이미지가 저장된 폴더
        self.test_dataset_dir = os.path.join("InputImage", opt.test_dataset_dir)
        self.scale = opt.scale
        self.n_colors = opt.n_colors
        self.rgb_range = opt.rgb_range

        # 데이터를 변수에 저장
        # os.listdir - 매개변수로 입력받은 경로를 통해 파일 리스트를 반환한다.
        # 불러온 파일명 + 경로를 통해 이미지 경로들을 저장한다.
        self.image_filenames = [os.path.join(self.test_dataset_dir, x) for x in listdir(self.test_dataset_dir) if is_image_file(x)]

        # 테스트 이미지들을 오름차순으로 정렬한다.
        self.image_filenames.sort() 

    def __getitem__(self, idx):
        # 확장자를 분리하여 순수 파일 이름만 변수에 저장
        filename = os.path.splitext(os.path.basename(self.image_filenames[idx]))[0]

        # OpenCV를 통해 이미지를 lr 변수에 저장한다.
        lr = cv2.imread(self.image_filenames[idx])

        # 이미지의 채널을 조정하고 Tensor로 변환한다.
        lr, = set_channel(lr, n_channels=self.n_colors)
        lr_t, = np2Tensor(lr, rgb_range=self.rgb_range)
        return {'LR': lr_t, 'LR_paths': self.image_filenames[index]}

    def __len__(self):
        return len(self.image_filenames)

