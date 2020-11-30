"""
SRCNN
--특징--
단순한 네트워크 구조, 우수한 복원 품질
고 해상도 이미지 처리속도는 만족스럽지 못함(실시간X)
LR이미지를 BICUBIC을 사용하여 업샘플링하여 입력한다
비선형 매핑단계에서 비용이 많이든다

--문제점--
입력, 출력의 크기는 동일하다
3단계로 구성, 중간 계층은 HR특징에 직접 매칭하는 계층
시간 복잡성이 HR크기에 비례하고 중간계층이 많이 기여한다.

FSRCNN
--특징--
마지막에 Deconvolution 레이어 추가
비선형매핑단계 대신, 개별적인 축소 확장레이어 추가
그 사이에 같은 필터크기의 여러 레이어를 추가 -> 모래시계 모양
기존성능을 우수하며 속도는 향상된다.
크기가 다른 업 스케일링 요소여도 Deconvolution만 교체하면된다.

원본 코드 출처 : https://github.com/yjn870/FSRCNN-pytorch
"""
# math - 수학 연산을 위한 모듈
from models.BaseModel import BaseModel
from torch import nn
import math, torch
from torch.nn import init
from torch.nn import parallel as P
from collections import OrderedDict
from util import util
from skimage.metrics import structural_similarity as compare_ssim

class FSRCNNModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)

        # Loss 함수들의 이름을 리스트로 선언한다.
        self.loss_names = ['FSRCNN']
        self.scale = opt.scale
        self.opt = opt

        # 사진 저장을 위한 리스트
        if opt.mode != "Test":
            self.visual_names = ['LR', 'HR', 'SR']
        else:
            self.visual_names = ['LR', 'SR']

        # 저장 또는 불러올 모델의 이름을 선언한다
        self.model_names = ['FSRCNN']
        self.n_GPUs = len(self.gpu_ids)
        self.networks = Network()

        # 네트워크를 정의한다.
        self.loss = nn.MSELoss()
        self.netFSRCNN = self.networks.define_FSRCNN(opt, self.gpu_ids)
        self.optimizer_FSRCNN = torch.optim.Adam(self.netFSRCNN.parameters(), lr=0.0001)
        self.optimizers.append(self.optimizer_FSRCNN)

    def cal_psnr(self):
        """ PSNR 측정 """
        original_img = util.tensor2im(self.HR)
        compressed_img = util.tensor2im(self.SR)
        return util.PSNR(original_img, compressed_img)
        
    def cal_ssim(self):
        """ SSIM 측정 """
        original_img = util.tensor2im(self.HR)
        compressed_img = util.tensor2im(self.SR)
        (score, diff) = compare_ssim(original_img, compressed_img, full=True, multichannel=True)
        return score

    def set_input(self, input):
        """
        이미지 데이터 GPU/CPU 할당 및 저장
        """
        # 데이터 방향에 따른 이미지 조정 후 device에 설정
        self.LR = input["LR"].to(self.device)
        if self.opt.mode != "Test":
            self.HR = input["HR"].to(self.device)
        self.image_paths = input["LR_paths"]

    def forward(self):
        """
        FSRCNN 순전파
        """
        if not self.opt.no_chop:
            self.SR = self.forward_chop(self.LR)
        else:
            self.SR = self.netFSRCNN(self.LR)

    def backward_FSRCNN(self):
        """
        FSRCNN 역전파
        """
        self.loss_FSRCNN = self.loss(self.SR, self.HR)
        self.loss_FSRCNN.backward()

    # 네트워크 실행 메소드
    def optimize_parameters(self):
        """ 학습을 진행하고 Loss, Gradient 측정 및 갱신 """
        # 학습을 진행하는 메소드
        # 순전파 진행
        self.forward()

        # G_A and G_B
        # Generator A, B의 가중치를 갱신한다.
        self.optimizer_FSRCNN.zero_grad()
        self.backward_FSRCNN()
        self.optimizer_FSRCNN.step()

    def get_current_losses(self):
        """ 학습 중인 Loss 결과를 출력한다. """
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

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
                y = P.data_parallel(self.netFSRCNN, *x, self.gpu_ids)
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

class Network():
    def __init__(self):
        pass

    def define_FSRCNN(self, opt, gpu_ids):
        net = FSRCNN(opt)
        return self.init_net(net, opt.init_type, opt.init_gain, gpu_ids)

    # 생성한 네트워크를 Device에 등록하는 메소드
    def init_net(self, net, init_type='normal', init_gain=0.02, gpu_ids=[]):
        """
        생성한 네트워크를 GPU 또는 CPU에 등록
        """
        if len(gpu_ids) > 0:
            assert(torch.cuda.is_available())
            net.to(gpu_ids[0])

            # gpu_ids가 여러개일 경우 병렬처리
            net = torch.nn.DataParallel(net, gpu_ids)
        self.init_weights(net, init_type, init_gain=init_gain)
        return net

        # 가중치 초기화 메소드
    def init_weights(self, net, init_type='normal', init_gain=0.02):
        """
        가중치 초기화
        """
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, init_gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=init_gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                init.normal_(m.weight.data, 1.0, init_gain)
                init.constant_(m.bias.data, 0.0)
        net.apply(init_func)

    # 스케쥴러 생성 메소드
    def get_scheduler(self, optimizer, opt):
        """
        learning_rate 스케쥴러 생성
        """
        # 학습률 scheduler 설정
        # 계산된 값을 초기 lr에 곱해서 사용한다
        # 최종적으로 0이 되도록 한다.
        scheduler = ""
        return scheduler

# 본 프로젝트에서 사용되는 FSRCNN 정의
class FSRCNN(nn.Module):
    def __init__(self, args, num_channels=3, d=56, s=12, m=4):
        super(FSRCNN, self).__init__()
        # 파서로부터 정보를 받아서 변수에 저장
        self.scale = args.scale       


        # 모델 1계층
        # 합성곱층, 필터크기 = 5*5, padding = 5//2 -> Parametric Relu
        # --특징 추출--
        # SRCNN에서는 필터의 크기는 9로 설정된다.
        # 필터의 크기가 9에 대하여 5의 패치에서도 거의 모든 정보를 다룰 수 있다.
        # 정보손실이 가장 적은 패치 5를 선정
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=5, padding=5//2),
            nn.PReLU(d)
        )

        # 모델 2계층
        # 합성곱층, 필터크기 = 1*1, 0 padding -> PRelu 
        # --축소--
        # SRCNN에서는 고차원LR의 특징이 HR특징과 직접 매핑되면서 계산복잡성이 높아진다
        # 이를 방지하기 위하여 축소 단계를 거친다.
        # 패치 크기를 1로 줄이면 필터는 LR 기능 내에서 선형조합처럼 작동한다.
        # --비 선형 매핑--
        # m번의 (합성곱층, 필터크기 = 3*3, padding = 3//2 -> PRelu) -> 합성곱층, 필터크기 = 1*1, 0 padding -> PRelu
        # SR성능에 영향을 미치는 가장 중요한 부분(가장 큰 요소는 너비-필터수*깊이, 레이어수)
        # SRCNN처럼 5*5는 1*1보다 좋은 결과를 얻는다 그러나 비효율
        # FSRCNN에서는 성능과 규모 사이의 중간인 3*3채택
        # --확장--
        # 축소의 역 프로세스
        # 축소는 계산효율을위해 LR기능 차원의 수를 줄인다.
        # 이러한 상태에서 이미지를 복원하면 품질이 많이 떨어진다.
        # 확장 레이어를 통해 매핑이 끝난 이미지의 차원을 축소 전 차원과 동일하게 확장한다.
        # 축소와의 일관성을 위해 필터는 1*1, 채널 수는 LR 특칭 추출 계층 수와 동일
        self.mid_part = [nn.Conv2d(d, s, kernel_size=1), nn.PReLU(s)]
        for _ in range(m):
            self.mid_part.extend([nn.Conv2d(s, s, kernel_size=3, padding=3//2), nn.PReLU(s)])
        self.mid_part.extend([nn.Conv2d(s, d, kernel_size=1), nn.PReLU(d)])
        self.mid_part = nn.Sequential(*self.mid_part)

        # 모델 3계층
        # Deconvolution (필터링한 특징 값을 다시 이미지로 변환)
        # Convolution의 역계층 입출력의 위치를 바꾼다.
        # 입력의 시간을 이용하여 보폭(stride)설정 - 원하는 업 스케일링 요소
        # 보폭이 있는 convolution layer, 역이므로 동일한 크기의 필터 9*9
        # H out = (H in −1)×stride − 2×padding + dilation×(kernel_size−1) + output_padding + 1
        # W out = (W in −1)×stride − 2×padding + dilation×(kernel_size−1) + output_padding + 1
        self.last_part = nn.ConvTranspose2d(d, num_channels, kernel_size=9, stride=self.scale, padding=9//2,
                                            output_padding=self.scale-1)

        # 모델 설계 완료 후, 가중치 초기화 실행
        # zero-gradient 현상을 방지하기 위해 모델의 활성화함수는 모두 PRelu로 통일
        self._initialize_weights()


    # 가중치 초기화 함수
    # 각 계층별 합성곱층이나 Deconvolution일 경우에 가중치 및 편차 수정
    # 가중치는 평균 0, 편차는 2를 출력 데이터의 채널* 필터의 첫번째 요소의 가중치의 갯수 나누고 제곱한 값
    # PRelu에 적합한 방법으로 가중치를 초기화한다.
    # 마지막 계층인 Deconvolutional Layer 에서는 활성화 함수가 없으므로 SRCNN과 동일한 방식으로 가중치를 초기화 한다.
    # SRCNN 에서는 가중치를 평균 0, 표준편차 0.001로 초기화를 진행했다.
    # 편차는 공통적으로 모든 계층에서 0으로 초기화
    def _initialize_weights(self):
        for m in self.first_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        for m in self.mid_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        nn.init.normal_(self.last_part.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.last_part.bias.data)

    # 설계한 모델로 순전파 진행
    # 역전파 과정은 자동으로 실행된다.
    def forward(self, x):
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)
        return x