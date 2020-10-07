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
"""
# math - 수학 연산을 위한 모듈
import math

# 파이토치
from torch import nn

# 모델 생성 메소드
def make_model(args, parent=False):
    return FSRCNN(args)

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