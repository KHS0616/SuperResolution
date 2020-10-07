"""
EDSR - Enhanced DeepLearning Super Resolution
ResNet 기반의 딥러닝 네트워크
SRResNet, VDSR을 개선
ResBlock 내부는 conv, relu로 구성되있고 총 32개 존재
각각의 ResBlock간 스킵 커넥션이 이루어진다.
네트워크 마지막에는 PixelShuffle을 이용한 업 스케일링 수행
"""
# model.common - 내장 모듈인 common, 공통적으로 사용되는 작은 네트워크 사용
from model import common

# 파이토치
import torch.nn as nn

# 모델 생성 메소드
def make_model(args, parent=False):
    return EDSR(args)

# EDSR 네트워크
class EDSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSR, self).__init__()

        # 변수 선언
        # n_resblocks = resblock의 개수 - 32
        # n_feats = feature map의 개수 - 256
        # kernel_size = 필터의 크기 - 3        
        n_resblocks = 32#args.n_resblocks
        n_feats = 256#args.n_feats
        kernel_size = 3 
        scale = args.scale
        act = nn.ReLU(True)

        # RGB 이미지 관련 초기 가중치, 편향 값 초기화
        # 각 채널별 평균을 더하거나 뺀다.
        # rgb_range - 이미지 컬러 범위 default 255
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

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
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=0.1#args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # Tail(Upscailing) 단계
        # Upsampler 메소드를 통해 Conv-PixelShuffle 통과
        # 마지막에 Conv 통과
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
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