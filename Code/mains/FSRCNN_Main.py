"""
FSRCNN 메인 모듈
"""
from options.FSRCNNOptions import FSRCNNOptions
import models, time, tqdm, os
from data import createDataset
import torch, time
from util.visualizer import Visualizer
from util.visualizer import save_images
from util import util

# 옵션 저장 및 전역 변수 선언
opt = FSRCNNOptions().getOptions()
evalopt = FSRCNNOptions().getOptions()
evalopt.mode = "Eval"
evalopt.batch_size = 1
dataLoader = createDataset("FSRCNN", opt)
evalDataLoader = createDataset("FSRCNN", evalopt)

def Train():
    """ 학습 함수 """
    global opt

    # 데이터 셋 불러오기
    print("데이터 셋 불러오는 중")

    # 모델 불러오기
    print("모델 불러오는 중")
    models.createModel("FSRCNN", opt)
    model = models.getModel()
    model.setup(opt)
    total_iters = 0

    # 평가용 평균 psnr을 측정하기 위한 리스트 선언
    psnr_list = []
    ssim_list = []    
    
    for epoch in range(opt.epoch_count, opt.n_epochs + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()

        epoch_iter = 0

        visualizer = Visualizer(opt)
        visualizer.reset()

        with tqdm.tqdm(total=len(dataLoader), ncols=160) as t:
            t.set_description('epoch: {}/{}'.format(epoch, opt.n_epochs))

            # 학습 시작
            for i, v in enumerate(dataLoader):
                # 데이터 불러오기
                iter_start_time = time.time()

                # 학습 결과 빈도에 따른 결과 저장
                t_data = iter_start_time - iter_data_time

                # 배치사이즈에 따른 빈도 수 갱신
                total_iters += opt.batch_size
                epoch_iter += opt.batch_size

                # 데이터를 device에 등록
                model.set_input(v)

                # 학습 및 가중치 갱신
                model.optimize_parameters()
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                message = visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)

                # 상태바 업데이트
                t.set_postfix(loss='{:.6f}'.format(losses["FSRCNN"]))
                t.update(opt.batch_size)

        # 학습종료 후, 학습 모델 저장
        if epoch % opt.save_epoch_freq == 0:
            # 저장 경로 설정
            save_dir = os.path.join(evalopt.results_dir, evalopt.name, 'test_latest_iter{}'.format(epoch))

            # 모델 저장
            model.save_networks('latest')
            model.save_networks(epoch)

            # 이미지 비교 및 저장
            with tqdm.tqdm(total=len(evalDataLoader), ncols=160) as t:                
                for i, v in enumerate(evalDataLoader):
                    t.set_description('Eval Images: {}/{}'.format(i, len(evalDataLoader)))
                    # 데이터 장치에 등록 및 테스트
                    model.set_input(v)
                    model.test()

                    # 이미지 관련 정보 불러오기
                    visuals = model.get_current_visuals()
                    img_path = model.get_image_paths()
                    
                    # 이미지 저장
                    save_images(visuals,save_dir, img_path)

                    # 이미지의 psnr수치 비교                
                    psnr = model.cal_psnr()
                    ssim = model.cal_ssim()
                    psnr_list.append(psnr)        
                    ssim_list.append(ssim)

                    # 상태바 업데이트
                    t.update(1)
                avg = sum(psnr_list)/len(psnr_list)
                avg_s = sum(ssim_list)/len(ssim_list)    
                print(f"\n{epoch}번째 epoch 평과 결과 : 평균 PSNR = {round(avg, 3)} /  평균 SSIM = {round(avg_s, 3)}")

# # 학습 및 테스트를 위한 클래스
# class Trainer():
#     # 파서를 통해 입력받은 값을 변수에 저장 및 초기화
#     def __init__(self, opt, loader, my_model, my_loss):
#         # 파서 및 스케일 수치를 변수로 저장
#         # rgb_range - rgb 범위
#         # scale - 업 스케일 수치
#         # type - 학습인지 테스트인지 구분
#         # mode - 테스트인경우 저장할지 전송할지 구분
#         # model_name - 학습 및 테스트를 진행할 모델(네트워크)
#         # num_epoch - 학습 횟수
#         self.rgb_range = opt.rgb_range
#         self.scale = opt.scale
#         self.type = opt.type
#         self.mode = "save"
#         self.model_name = opt.model_name
#         self.n_processes = 8
#         self.num_epoch = opt.num_epoch

#         # 평가 진행중 PSNR, SSIM 결과 저장 리스트
#         self.psnrs = []
#         self.ssims = []

#         # 데이터 로드
#         self.loader = loader.loader

#         # 다양한 기능을 위한 utility 객체를 생성
#         self.utility = Utility(opt)

#         # 학습모드일 경우 평가 데이터 로드
#         if self.type == "Train":
#             self.eval_loader = loader.eval_loader

#         # 모델 저장
#         self.model = my_model
        
#         # 손실함수, 활성화 함수 저장  

#     # 테스트 메소드
#     def test(self):
#         # 기울기 갱신 X
#         torch.set_grad_enabled(False)

#         # 평가를 위한 설정
#         self.model.eval()

#         # 타이머 설정 및 테스트 시작
#         #timer_test = utility.timer()

#         # 백엔드 작업 시작
#         self.utility.begin_background()

#         # tqdm을 통해 진행 상황 시각화
#         # lr, hr, 파일이름을 각각 데이터 셋으로 부터 받아온다.
#         for lr, hr, filename in tqdm(self.loader, ncols=80):

#             # 데이터 불러오기 및 변수 할당
#             lr, hr = self.prepare(lr, hr)

#             # 지정된 모델에 입력 이미지와 스케일 수치를 전달한다.
#             # 예측된 sr이미지를 양자화 과정을 진행한다.
#             sr = self.model(lr)   
#             sr = self.utility.quantize(sr, self.rgb_range)

#             # 양자화된 sr이미지를 리스트에 담는다.
#             save_list = [sr]

#             # log.txt에 현재 내용을 작성한다.
#             # self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
#             #     sr, hr, scale, self.opt.rgb_range, dataset=d
#             # )

#             # LR, HR 이미지를 속성 값에 따라 저장할 리스트에 추가한다.
#             # if self.opt.save_gt:
#             #     save_list.extend([lr, hr])

#             # 저장 목록에 있는 파일들을 저장한다.
#             self.utility.save_results(save_list, filename[0])

#         # 백그라운드로 이미지를 저장중이던 프로세스를 중지
#         self.utility.end_background()

#         # 이미지를 전송한다면 이미지 넘파이 배열을 출력
#         if self.mode == "send":
#             return self.utility.send_image_list
#         else:
#             return 0

#         # # 테스트를 진행하기 까지 걸린 시간 출력
#         # # 저장 중이라는 알림 출력
#         # # self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
#         # # self.ckp.write_log('Saving...')

#         # # 백그라운드로 이미지를 저장중이던 프로세스를 중지
#         # if self.opt.save_results:
#         #     self.ckp.end_background()

#         # # 테스트 모드가 아닐 경우 모델을 저장한다.
#         # if not self.opt.test_only:
#         #     self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

#         # # 실행부터 저장까지 걸린 시간 출력
#         # self.ckp.write_log(
#         #     'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
#         # )

#         # 테스트가 종료됬으므로 기울기 갱신을 다시 On으로 변경
#         torch.set_grad_enabled(True)

#     # 학습을 진행하기 이전 기본 설정
#     # GPU/CPU 사용가능한 장치 확인후 메모리 할당
#     def prepare(self, *opt):        
#         device = device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#         def _prepare(tensor):
#             return tensor.to(device)
#         return [_prepare(a) for a in opt]



# Mode에 따른 진행 유형 결정
if opt.mode == "Train":
    Train()
elif opt.mode == "Test":
    Test()