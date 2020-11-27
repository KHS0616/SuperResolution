"""
데이터 셋 최상위 모듈
"""
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from abc import abstractmethod
import os

class BaseDataset(Dataset):
    def __init__(self, opt):
        self.IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif', '.TIF', '.tiff', '.TIFF']

    @abstractmethod
    def __len__(self):
        """ 데이터 셋의 길이 반환하는 추상 메소드 """
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """ 데이터 셋의 데이터를 반환하는 추상 메소드 """
        pass

    @abstractmethod
    def get_transform(self, opt, params=None, grayscale=False, method=Image.BICUBIC, convert=True):
        pass

    # 이미지 형식이 맞는지 확인하는 함수
    def is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in self.IMG_EXTENSIONS)

    # 데이터 셋을 만드는 함수
    def make_dataset(self, dir, max_dataset_size=float("inf")):
        # 이미지 정보를 저장하기 위한 빈 리스트 선언
        images = []

        # os.walk는 모든 파일을 탐색한다.
        # root 경로, dirs 폴더, files 파일을 반환한다.    
        # 최종적으로 inf보다 작은 이미지들을 반환(크면 inf로 설정)
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if self.is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
        return images[:min(max_dataset_size, len(images))]

    # 이미지를 전처리하는 함수
    def get_transform(self, opt, params=None, grayscale=False, method=Image.BICUBIC, convert=True, image_type="HR"):
        transform_list = []

        # grayscale - 흑백이미지
        # 흑백이미지인경우 이미지를 흑백형태로 변환한다.
        # 파이토치의 transforms.Grayscale 사용
        if grayscale:
            transform_list.append(transforms.Grayscale(1))

        # 전처리 옵션중에 resize 또는 scale_width가 있으면 실행
        # 각각 이미지 크기를 재조정하거나 다운 스케일링
        if 'resize' in opt.preprocess:
            osize = [opt.load_size, opt.load_size]
            transform_list.append(transforms.Resize(osize, method))
        elif 'scale_width' in opt.preprocess:
            transform_list.append(transforms.Lambda(lambda img: self.__scale_width(img, opt.load_size, opt.crop_size, method)))

        # crop 옵션이 있을 경우 실행
        # 이미지를 랜덤하게 자르기
        if 'crop' in opt.preprocess:
            if params is None:
                transform_list.append(transforms.RandomCrop(opt.crop_size))
            else:
                transform_list.append(transforms.Lambda(lambda img: self.__crop(img, params['crop_pos'], opt.crop_size)))

        # none 옵션이 있을 경우 실행
        if opt.preprocess == 'none':
            transform_list.append(transforms.Lambda(lambda img: self.__make_power_2(img, base=4, method=method)))

        # 이미지를 뒤집지 않느 flip 옵션이 활성화 안될경우 실행
        # params가 None이면 이미지를 랜덤으로 좌우 반전
        # 아닐경우 Image 모듈을 이용한 좌우 반전
        if not opt.no_flip:
            if params is None:
                transform_list.append(transforms.RandomHorizontalFlip())
            elif params['flip']:
                transform_list.append(transforms.Lambda(lambda img: self.__flip(img, params['flip'])))

        # convert 옵션이 있을 경우 실행
        # 이미지 정규화
        if convert:
            transform_list += [transforms.ToTensor()]
            if grayscale:
                transform_list += [transforms.Normalize((0.5,), (0.5,))]
            else:
                transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        # 이미지 전처리 과정을 반환
        # 파이토치 Compose를 이용하여 리스트를 저장(nn.Sequential과 유사)
        return transforms.Compose(transform_list)

    # none 옵션일 때 실행되는 함수
    # 이미지를 정수로 바꾸기 위한 과정
    # 이때 사용되는 보간법은 바이큐빅
    def __make_power_2(self, img, base, method=Image.BICUBIC):
        ow, oh = img.size
        h = int(round(oh / base) * base)
        w = int(round(ow / base) * base)
        if h == oh and w == ow:
            return img

        __print_size_warning(ow, oh, w, h)
        return img.resize((w, h), method)

    # 이미지를 crop size로 줄이기
    def __scale_width(self, img, target_size, crop_size, method=Image.BICUBIC):
        ow, oh = img.size
        if ow == target_size and oh >= crop_size:
            return img
        w = target_size
        h = int(max(target_size * oh / ow, crop_size))
        return img.resize((w, h), method)

    # 이미지 crop
    def __crop(self, img, pos, size):
        ow, oh = img.size
        x1, y1 = pos
        tw = th = size
        if (ow > tw or oh > th):
            return img.crop((x1, y1, x1 + tw, y1 + th))
        return img

    # 이미지를 flip 하는 메소드
    # 좌우 반전
    def __flip(self, img, flip):
        if flip:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


    def __print_size_warning(self, ow, oh, w, h):
        """Print warning information about image size(only print once)"""
        if not hasattr(__print_size_warning, 'has_printed'):
            print("The image size needs to be a multiple of 4. "
                "The loaded image size was (%d, %d), so it was adjusted to "
                "(%d, %d). This adjustment will be done to all images "
                "whose sizes are not multiples of 4" % (ow, oh, w, h))
            __print_size_warning.has_printed = True