"""
학습과 테스트를 진행하는 파일
"""

# os - 파일 매니저 등의 경로관리를 위한 모듈
# math - mul, div 등의 연산을 위한 모듈
# 현재 작업 진행상황을 확인하기 위한 모듈
import os
import math
from tqdm import tqdm

# utility.Utility - 이미지 저장, psnr 계산 등 보조작업을 위한 내장 모듈
from utility import Utility

# 파이토치
import torch
import torch.nn.utils as utils

# 학습 및 테스트를 위한 클래스
class Trainer():
    # 파서를 통해 입력받은 값을 변수에 저장 및 초기화
    def __init__(self, args, loader, my_model, my_loss):
        # 파서 및 스케일 수치를 변수로 저장
        # rgb_range - rgb 범위
        # scale - 업 스케일 수치
        # type - 학습인지 테스트인지 구분
        # mode - 테스트인경우 저장할지 전송할지 구분
        # model_name - 학습 및 테스트를 진행할 모델(네트워크)
        # num_epoch - 학습 횟수
        self.rgb_range = args.rgb_range
        self.scale = args.scale
        self.type = args.type
        self.mode = "save"
        self.model_name = args.model_name
        self.n_processes = 8
        self.num_epoch = args.num_epoch

        # 평가 진행중 PSNR, SSIM 결과 저장 리스트
        self.psnrs = []
        self.ssims = []

        # 데이터 로드
        self.loader = loader.loader

        # 다양한 기능을 위한 utility 객체를 생성
        self.utility = Utility(args)

        # 학습모드일 경우 평가 데이터 로드
        if self.type == "Train":
            self.eval_loader = loader.eval_loader

        # 모델 저장
        self.model = my_model
        
        # 손실함수, 활성화 함수 저장
        self.loss = torch.nn.MSELoss()#my_loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)#utility.make_optimizer(args, self.model)

        # 활성화 함수(optimizer)를 지정된 경로를 통해 불러온다.
        # if self.args.load != '':
        #     self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        # 오차 값 설정(초기 오차 값)
        # self.error_last = 1e8

    # 학습 메소드
    def train(self):
        for epoch in range(self.num_epoch):
            # 학습 모드 설정
            self.model.train()

            # 진행바 설정
            with tqdm(total=(len(self.loader)), ncols=80) as t:
                t.set_description('epoch: {}/{}'.format(epoch, self.num_epoch - 1))

                # 학습 시작
                for lr, hr in self.loader:
                    # 데이터 불러오기
                    lr, hr = self.prepare(lr, hr)

                    # 최적화 함수 초기화
                    # SR 예측 및 loss 측정
                    # loss 역전파 실행
                    self.optimizer.zero_grad()
                    sr = self.model(lr)
                    loss = self.loss(sr, hr)
                    loss.backward()

                    # 최적화 함수(optimizer) 기울기 갱신
                    self.optimizer.step()

                    # 상태바 업데이트
                    t.set_postfix(loss='{:.6f}'.format(loss))
                    t.update(1)

            # 학습종료 후, 학습 모델 저장
            torch.save(self.model.state_dict(), os.path.join("Model", self.model_name, 'epoch_{}.pt'.format(epoch)))

            # 모델 평가 시작
            self.model.eval()
            for lr, hr in self.eval_loader:
                # 데이터 메모리 할당
                lr, hr = self.prepare(lr, hr)

                # 기울기 갱신하지 않도록 설정
                with torch.no_grad():
                    sr = self.model(lr)

                # psnr 수치 계산
                self.psnrs.append(self.utility.calc_psnr(sr, hr).item())
            # 평균 psnr, ssim 출력
            print("epoch {}/{} result => PSNR : {}, SSIM : {}".format(epoch, self.num_epoch, sum(self.psnrs)/len(self.psnrs), 0))       

    # 테스트 메소드
    def test(self):
        # 기울기 갱신 X
        torch.set_grad_enabled(False)

        # 평가를 위한 설정
        self.model.eval()

        # 타이머 설정 및 테스트 시작
        #timer_test = utility.timer()

        # 백엔드 작업 시작
        self.utility.begin_background()

        # tqdm을 통해 진행 상황 시각화
        # lr, hr, 파일이름을 각각 데이터 셋으로 부터 받아온다.
        for lr, hr, filename in tqdm(self.loader, ncols=80):

            # 데이터 불러오기 및 변수 할당
            lr, hr = self.prepare(lr, hr)

            # 지정된 모델에 입력 이미지와 스케일 수치를 전달한다.
            # 예측된 sr이미지를 양자화 과정을 진행한다.
            sr = self.model(lr)   
            sr = self.utility.quantize(sr, self.rgb_range)

            # 양자화된 sr이미지를 리스트에 담는다.
            save_list = [sr]

            # log.txt에 현재 내용을 작성한다.
            # self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
            #     sr, hr, scale, self.args.rgb_range, dataset=d
            # )

            # LR, HR 이미지를 속성 값에 따라 저장할 리스트에 추가한다.
            # if self.args.save_gt:
            #     save_list.extend([lr, hr])

            # 저장 목록에 있는 파일들을 저장한다.
            self.utility.save_results(save_list, filename[0])

        # 백그라운드로 이미지를 저장중이던 프로세스를 중지
        self.utility.end_background()

        # 이미지를 전송한다면 이미지 넘파이 배열을 출력
        if self.mode == "send":
            return self.utility.send_image_list
        else:
            return 0

        # # 테스트를 진행하기 까지 걸린 시간 출력
        # # 저장 중이라는 알림 출력
        # # self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        # # self.ckp.write_log('Saving...')

        # # 백그라운드로 이미지를 저장중이던 프로세스를 중지
        # if self.args.save_results:
        #     self.ckp.end_background()

        # # 테스트 모드가 아닐 경우 모델을 저장한다.
        # if not self.args.test_only:
        #     self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

        # # 실행부터 저장까지 걸린 시간 출력
        # self.ckp.write_log(
        #     'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        # )

        # 테스트가 종료됬으므로 기울기 갱신을 다시 On으로 변경
        torch.set_grad_enabled(True)

    # 학습을 진행하기 이전 기본 설정
    # GPU/CPU 사용가능한 장치 확인후 메모리 할당
    def prepare(self, *args):        
        device = device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        def _prepare(tensor):
            return tensor.to(device)
        return [_prepare(a) for a in args]
