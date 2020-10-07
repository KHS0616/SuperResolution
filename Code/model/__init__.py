"""
모델을 관리하는 최상위 파일
"""
# importlib.import_module - 코드 내부에서 모듈에 접근하기 위한 모듈
# os - 파일 경로등을 관리하기 위한 모듈
from importlib import import_module
import os

# 파이토치
import torch
import torch.nn as nn
import torch.nn.parallel as P
import torch.backends.cudnn as cudnn

# 메인 모델 클래스
class Model(nn.Module):
    # 초기화 메소드
    def __init__(self, args):
        super(Model, self).__init__()
        # 파서를 통해 변수에 정보를 저장한다.
        # model_name - SR 대상 네트워크의 이름
        # n_GPUs - 사용할 GPU의 개수
        # chop - 이미지를 분할처리 하는지 여부
        # scale - 업 스케일 수치
        # load_model_name - 불러올 모델의 이름 default -1
        self.model_name = args.model_name
        self.n_GPUs = args.n_GPUs
        self.chop = args.chop
        self.scale = args.scale
        self.load_model_name = args.load_model_name
        
        # 알림 문구 콘솔에 출력
        print("{} 모델 생성중...".format(self.model_name))

        # GPU-Cuda 사용가능 여부에 따른 학습/테스트 장치 설정
        if torch.cuda.is_available():
            cudnn.benchmark = True
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device("cpu")

        # 모델 생성
        # import_module을 통해 model 내부의 지정된 이름의 파이썬 파일을 가져온다.
        # 해당 과정에서 자동으로 객체가 만들어져서 module에 저장된다.
        module = import_module('model.' + self.model_name.lower())

        # 모델을 생성하고 GPU/CPU에 할당한다.
        self.model = module.make_model(args).to(self.device)

        # 불러올 모델이 있으면 모델을 불러온다.
        if self.load_model_name != None:
            self.load_state_dict()

    # 순전파 과정
    def forward(self, x):
        # 학습 대상 모델이 스케일 수치를 변경하는 메소드 존재시 변경
        # if hasattr(self.model, 'set_scale'):
        #     self.model.set_scale(self.scale)

        # chop 기능 활성화 여부, GPU 개수에 따른 분할처리
        if not self.chop:            
            # GPU 개수가 1개보다 많을 경우 데이터 분산처리
            if self.n_GPUs > 1:
                return P.data_parallel(self.model, x, range(self.n_GPUs))
            else:
                return self.model(x)
        else:
            return self.forward_chop(x)

    # 모델을 불러오는 메소드
    # pth, pt 파일에서 각 층별로 저장된 가중치, 편향 값을 불러와서 저장한다.
    def load_state_dict(self):
        # state_dict를 변수에 할당
        own_state = self.model.state_dict()

        # 모델 경로+이름 합치기
        MODEL_PATH = os.path.join("Model", self.load_model_name)

        # 모델을 불러와서 가중치, 편향 값 저장
        for n, p in torch.load(MODEL_PATH, map_location=lambda storage, loc: storage).items():
            if n in own_state.keys():
                own_state[n].copy_(p)
            else:
                raise KeyError(n)
            
    # 이미지를 분할하여 학습/테스트 하는 메소드
    def forward_chop(self, *args, shave=10, min_size=160000):
        # GPU의 개수는 최대 4개만 가능하도록 설정
        n_GPUs = min(self.n_GPUs, 4)        
        scale = self.scale

        # 이미지의 높이와 폭을 저장한다.
        h, w = args[0].size()[-2:]

        # 이미지를 자르는 slice 객체 생성
        # top - 0~높이의 절반+10
        # bottom - 높이 ~ 높이 - 높이의 절반 - 10
        # left, right도 top, bottom과 같이 10의 간격을 두고 양쪽에서 자른다.
        top = slice(0, h//2 + shave)
        bottom = slice(h - h//2 - shave, h)
        left = slice(0, w//2 + shave)
        right = slice(w - w//2 - shave, w)

        # 이미지를 10을 간격으로 4등분한다.
        # 최종 형태 torch.Size([4, 3, 550, 970]) - 원본 1920x1080 기준
        x_chops = [torch.cat([
            a[..., top, left],
            a[..., top, right],
            a[..., bottom, left],
            a[..., bottom, right]
        ]) for a in args]

        # y_chops를 저장하기위한 리스트 선언
        y_chops = []

        # 이미지 크기가 64만보다 작거나 큰 경우에 따라 분기
        if h * w < 4 * min_size:
            for i in range(0, 4, n_GPUs):
                # x변수는 GPU개수에 따라 잘라진 이미지를 저장한다.
                # y변수는 잘라진 이미지를 GPU에 맞게 분산처리한다.
                x = [x_chop[i:(i + n_GPUs)] for x_chop in x_chops]
                y = P.data_parallel(self.model, *x, range(n_GPUs))
                # 분산처리 후에 y가 리스트 안에 없으면 리스트 안에 추가한다.
                if not isinstance(y, list): y = [y]

                # y_chops가 빈 리스트일 경우 y를 각각 자른 영역만큼 추가
                # GPU에서 분할처리후 하나로 합쳐졌던 y를 다시 분할하는 과정                
                if not y_chops:
                    y_chops = [[c for c in _y.chunk(n_GPUs, dim=0)] for _y in y]
                else:
                    for y_chop, _y in zip(y_chops, y):
                        y_chop.extend(_y.chunk(n_GPUs, dim=0))

        # 크기가 클 경우 분할하여 이미지를 테스트한다.
        # 이미지를 4개로 분할하여 테스트 진행
        else:
            for p in zip(*x_chops):
                # 4차원 형태를 맞춰주기 위하여 빈 차원 생성
                # 가공된 이미지 조각을 GPU 분산 작업을 위해 자기자신 호출 - 재귀함수
                p1 = [p[0].unsqueeze(0)]
                y = self.forward_chop(*p1, shave=shave, min_size=min_size)
                #y = self.forward_chop(*p, shave=shave, min_size=min_size)

                # y가 리스트 내부에 없을 경우 리스트 내부로 이동
                if not isinstance(y, list): y = [y]

                # y_chops가 비어있을 경우 GPU연산에 의해 분할된 이미지들을 y_chops에 저장
                if not y_chops:
                    y_chops = [[_y] for _y in y]
                
                # y_chop에 이미지 조각들 추가
                else:
                    for y_chop, _y in zip(y_chops, y): y_chop.append(_y)

        # 높이와 너비를 스케일 수치만큼 곱한다.
        h *= scale
        w *= scale

        # 새로운 높이와 너비의 절반만큼 각 부분을 4등분 한다.
        # bottom_r, right_r
        top = slice(0, h//2)
        bottom = slice(h - h//2, h)
        bottom_r = slice(h//2 - h, None)
        left = slice(0, w//2)
        right = slice(w - w//2, w)
        right_r = slice(w//2 - w, None)

        # batch size, number of color channels
        # 조각난 이미지의 개수, 채널의 개수를 b, c에 저장한다.
        # 개수, 채널의 개수, 높이, 너비에 맞는 텐서를 난수 값을 이용하여 새로 만든다.
        b, c = y_chops[0][0].size()[:-2]
        y = [y_chop[0].new(b, c, h, w) for y_chop in y_chops]

        # 4등분 됬던 이미지들을 _y에 합친다.
        for y_chop, _y in zip(y_chops, y):
            _y[..., top, left] = y_chop[0][..., top, left]
            _y[..., top, right] = y_chop[1][..., top, right_r]
            _y[..., bottom, left] = y_chop[2][..., bottom_r, left]
            _y[..., bottom, right] = y_chop[3][..., bottom_r, right_r]

        # 이미지가 1개일 경우 맨 바깥 리스트를 없앤다.
        # 맨 바깥 리스트는 이미지의 개수를 분류하기 위한 리스트
        # 1개이므로 불필요해져서 없애고 반환
        # 또한 1개의 이미지가 4개로 분할되어 계산되었을 때를 위한 절차이기도 함
        if len(y) == 1: y = y[0]
        
        return y
