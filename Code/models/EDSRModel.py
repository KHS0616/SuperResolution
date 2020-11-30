"""
EDSR - Enhanced DeepLearning Super Resolution
ResNet 기반의 딥러닝 네트워크
SRResNet, VDSR을 개선
ResBlock 내부는 conv, relu로 구성되있고 총 32개 존재
각각의 ResBlock간 스킵 커넥션이 이루어진다.
네트워크 마지막에는 PixelShuffle을 이용한 업 스케일링 수행

원본 코드 출처 : https://github.com/thstkdgus35/EDSR-PyTorch
"""
from models.BaseModel import BaseModel
import torch.nn as nn
import torch.nn.parallel as P
from torch.nn import init
import torch, math
from util import util
from skimage.metrics import structural_similarity as compare_ssim
from collections import OrderedDict

# EDSR 모델
class EDSRModel(BaseModel):
    def __init__(self, opt):
        """
        EDSR 생성자
        """
        super().__init__(opt)
        
        # Loss 함수들의 이름을 리스트로 선언한다.
        self.loss_names = ['EDSR']
        self.scale = opt.scale
        self.opt = opt
        # 사진 저장을 위한 리스트
        if opt.mode != "Test":
            self.visual_names = ['LR', 'HR', 'SR']
        else:
            self.visual_names = ['LR', 'SR']

        # 저장 또는 불러올 모델의 이름을 선언한다
        self.model_names = ['EDSR']
        self.n_GPUs = len(self.gpu_ids)
        self.networks = Network()

        # 네트워크를 정의한다.
        self.loss = nn.MSELoss()
        self.netEDSR = self.networks.define_EDSR(opt, self.gpu_ids)
        self.optimizer_EDSR = torch.optim.Adam(self.netEDSR.parameters(), lr=0.0001)
        self.optimizers.append(self.optimizer_EDSR)

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
        Cycle GAN 순전파
        """
        if not self.opt.no_chop:
            self.SR = self.forward_chop(self.LR)
        else:
            self.SR = self.netEDSR(self.LR)

    def backward_EDSR(self):
        """
        EDSR 역전파
        """
        self.loss_EDSR = self.loss(self.SR, self.HR)
        self.loss_EDSR.backward()

    # 네트워크 실행 메소드
    def optimize_parameters(self):
        """ 학습을 진행하고 Loss, Gradient 측정 및 갱신 """
        # 학습을 진행하는 메소드
        # 순전파 진행
        self.forward()

        # G_A and G_B
        # Generator A, B의 가중치를 갱신한다.
        self.optimizer_EDSR.zero_grad()
        self.backward_EDSR()
        self.optimizer_EDSR.step()

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
                y = P.data_parallel(self.netEDSR, *x, self.gpu_ids)
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

    def define_EDSR(self, opt, gpu_ids):
        net = EDSR(opt)
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

# 기본적인 Convolution 연산
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

# 학습 이전 가중치, 편향 값 초기화 메소드
# RGB 이미지의 각 채널당 평균 값을 빼거나 더한다
# 결과를 가중치와 편향 값을 초기화 하는데 사용한다.
class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False 

# 기존 SRResNet에서 사용한 ResBlock
# 비교용으로 생성
class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

# ResBlock 네트워크
class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=0.1):

        super(ResBlock, self).__init__()

        # bn 유무에 따른 층을 쌓는다.
        # 최종적으로 conv - relu - conv
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    # 순전파 과정
    def forward(self, x):
        res = self.body(x).mul(self.res_scale)

        # Multi 계층
        # Feature Map의 분산 값이 너무 커지는 것을 방지
        # 특정 상수 값을 곱함(0.1)
        res += x
        return res

# Upscaling 네트워크
class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        # 2,3,4 스케일 파라미터 수치에 따라 행동 분기
        # Conv 층을 통해 파라미터 수치의 제곱의 배 만큼 확장
        # 확장 이후 PixelShuffle을 통해 재배치
        # 스케일 4의 경우 스케일 2의 행동을 2번 반복
        if (scale & (scale - 1)) == 0:    
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))
        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        # 상위 클래스인 nn.Sequential을 호출하여 현재 누적된 레이어를 저장
        super(Upsampler, self).__init__(*m)

# EDSR 네트워크
class EDSR(nn.Module):
    def __init__(self, args, conv=default_conv):
        super(EDSR, self).__init__()

        # 변수 선언
        # n_resblocks = resblock의 개수 - 32
        # n_feats = feature map의 개수 - 256
        # kernel_size = 필터의 크기 - 3        
        n_resblocks = 32
        n_feats = 256
        kernel_size = 3 
        scale = args.scale
        act = nn.ReLU(True)

        # RGB 이미지 관련 초기 가중치, 편향 값 초기화
        # 각 채널별 평균을 더하거나 뺀다.
        # rgb_range - 이미지 컬러 범위 default 255
        self.sub_mean = MeanShift(args.rgb_range)
        self.add_mean = MeanShift(args.rgb_range, sign=1)

        # head(Pre-processing) 단계
        # Conv 1계층 통과
        # input - 3, output - 256
        # n_colors 입력 이미지의 채널 개수 default 3
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # body(ResBlocks) 단계
        # ResBlock 32개층으로 구성
        # ResBlock 마지막에 Conv 통과
        # res_scale - 분산 값이 커지는 것을 막기위해 특정 상수를 곱하는 것 default 0.1
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=0.1#args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # Tail(Upscailing) 단계
        # Upsampler 메소드를 통해 Conv-PixelShuffle 통과
        # 마지막에 Conv 통과
        m_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    # 순전파 과정
    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x 