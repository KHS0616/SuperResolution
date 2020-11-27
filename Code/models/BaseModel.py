""" 모델 최상위 모듈 """
import os, torch
from torch.nn.modules import Module
from abc import abstractmethod
from collections import OrderedDict

class BaseModel(Module):
    def __init__(self, opt):
        super().__init__()
        # GPU 설정
        self.opt = opt
        self.gpu_ids = list(map(int, opt.gpu_ids.split(",")))
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')

        # 저장 경로 설정
        # checkpoints_dir - 체크포인트 저장 경로
        # name - 체크포인트 내부 대분류 폴더이름 
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if os.path.exists(self.save_dir):
            pass
        else:
            os.mkdir(self.save_dir)

        # 스케쥴러 작성을 위한 목적함수, 활성화 함수 리스트 선언
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []

    @abstractmethod
    def forward(self):
        pass

    def test(self):
        """ 모델 테스트 """
        with torch.no_grad():
            self.forward()

    # 모델의 기본적인 설정을 하는 메소드
    def setup(self, opt):
        """
        모델의 기본적인 설정
        """
        # 학습 중 일경우 학습률 scheduler 설정
        # 각각의 optG, optD에 대한 lr scheduler 불러오기
        if opt.mode == "Train":
            self.schedulers = [self.networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]

        # 테스트 중이거나 이어서 학습할 때 모델의 설정을 불러와서 사용
        # load_iter 값이 0보다 크면 iter 값에 해당하는 모델을 불러온다.
        # load_iter 값이 0보다 작거나 같으면 epoch에 해당하는 모델을 불러온다.(default latest)
        if not opt.mode == "Train" or opt.continue_train:
            load_suffix = opt.epoch

            # 설정된 이름에 해당하는 네트워크를 불러온다.
            self.load_networks(load_suffix)

    # 네트워크 저장
    def save_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    # 네트워크 불러오기
    def load_networks(self, epoch):
        """
        모델(네트워크)를 불러오는 메소드
        """
        # 지정된 모델 이름들로부터 네트워크를 불러온다.
        # 모델 이름은 각 모델별 클래스 내부에서 선언된다.
        for name in self.model_names:
            if isinstance(name, str):
                # 불러올 모델의 이름을 지정한다.
                # ex) 10_net_D_A.pth
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)

                # 정의된 네트워크를 불러온다
                # 형식은 net + 모델이름 (ex : netEDSR, netFSRCNN)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('%s 에서 모델을 불러왔습니다.' % load_path)
                
                # 네트워크 불러오기
                state_dict = torch.load(load_path, map_location=str(self.device))

                # 메타데이터가 있는경우 삭제..?
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata
                
                for key in list(state_dict.keys()):
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        # 키의 이름만을 key 변수에 저장
        key = keys[i]

        # key의 이름만 남은경우 실행
        # InstanceNorm의 에러 수정..?
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def get_current_visuals(self):
        """ 현재 결과를 출력 """
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_image_paths(self):
        """ Return image paths that are used to load current data"""
        return self.image_paths