import os.path
from PIL import Image
from data.BaseDataset import BaseDataset
import numpy as np
import random

# default 데이터 셋 클래스
class TrainDataset(BaseDataset):
    # 데이터 셋 초기화
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

        # 학습 대상 이미지 경로 설정
        self.opt = opt
        self.dir_A = os.path.join(opt.dataroot, "trainA")
        self.dir_B = os.path.join(opt.dataroot, "trainB")

        # 이미지 형식에 맞는 데이터들을 필터링하고 경로를 정렬해서 저장
        self.A_paths = sorted(self.make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(self.make_dataset(self.dir_B, opt.max_dataset_size))

        # 각 이미지의 개수를 저장
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        # B -> A 로 학습 설정
        btoA = self.opt.direction == 'BtoA'

        # 각 이미지들의 채널을 조정한다.
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc

        # 전처리 함수를 저장
        self.transform_A = self.get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = self.get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        # A 이미지의 경로를 인덱스 번호에 맞게 지정
        # epoch, batch_size 등이 이미지 보다 많을 경우 순차적으로 처리하기 위해 % 사용
        A_path = self.A_paths[index % self.A_size]
        self.origin_A_path = A_path

        # serial_batches 옵션 여부에 따라 데이터를 순차처리할지 랜덤처리할지 결정
        if self.opt.shuffle:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)

        # B 이미지의 경로를 저장
        B_path = self.B_paths[index_B]
        self.origin_B_path = B_path

        # 이미지를 Image 모듈을 이용하여 열고 RGB로 변환한다.
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        
        # 이미지 전처리를 진행한다.
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        # 이미지와 이미지 경로를 반환한다.
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

# default 데이터 셋 클래스
class TestDataset(BaseDataset):
    # 데이터 셋 초기화
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

        # 학습 대상 이미지 경로 설정
        self.opt = opt
        self.dir_A = os.path.join(opt.dataroot, "testA")
        self.dir_B = os.path.join(opt.dataroot, "testB")

        # 이미지 형식에 맞는 데이터들을 필터링하고 경로를 정렬해서 저장
        self.A_paths = sorted(self.make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(self.make_dataset(self.dir_B, opt.max_dataset_size))

        # 각 이미지의 개수를 저장
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        # B -> A 로 학습 설정
        btoA = self.opt.direction == 'BtoA'

        # 각 이미지들의 채널을 조정한다.
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc

        # 전처리 함수를 저장
        self.transform_A = self.get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = self.get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        # A 이미지의 경로를 인덱스 번호에 맞게 지정
        # epoch, batch_size 등이 이미지 보다 많을 경우 순차적으로 처리하기 위해 % 사용
        A_path = self.A_paths[index % self.A_size]
        self.origin_A_path = A_path

        # serial_batches 옵션 여부에 따라 데이터를 순차처리할지 랜덤처리할지 결정
        if self.opt.shuffle:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)

        # B 이미지의 경로를 저장
        B_path = self.B_paths[index_B]
        self.origin_B_path = B_path

        # 이미지를 Image 모듈을 이용하여 열고 RGB로 변환한다.
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        
        # 이미지 전처리를 진행한다.
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        # 이미지와 이미지 경로를 반환한다.
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)
