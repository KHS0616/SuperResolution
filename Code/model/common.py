"""
모델에서 공통적으로 사용하는 작은 네트워크를 모아둔 파일
"""
# math - 수학 연산을 위한 모듈
import math

# 파이토치
import torch
import torch.nn as nn
import torch.nn.functional as F

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