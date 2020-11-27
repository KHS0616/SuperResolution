"""
CycleGAN 메인 모듈
"""
from options.CycleGANOptions import CycleGANOptions
import models, time, tqdm, random, os
from data import createDataset
from util.visualizer import Visualizer
from util.visualizer import save_images
from util import util

# 옵션 불러오기 및 전역함수 선언
opt = CycleGANOptions().getOptions()
evalopt = CycleGANOptions().getOptions()
dataLoader = createDataset("CycleGAN", opt)

# evalopt 테스트용으로 설정
evalopt.preprocess = "none"
evalopt.no_dropout = True
evalopt.mode = "Test"
evalopt.batch_size = 1
evalopt.shuffle = True
evalopt.no_flip = True
evalDataLoader = createDataset("CycleGAN", evalopt)

def Train():
    """ 학습 함수 """
    global opt

    # 데이터 셋 불러오기
    print("데이터 셋 불러오는 중")

    # 모델 불러오기
    print("모델 불러오는 중")
    models.createModel("CycleGAN", opt)
    model = models.getModel()
    model.setup(opt)
    total_iters = 0

    # 학습 시작
    # epoch_count 부터 n_ephcos + n_epochs_decay 만큼 학습
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        # 각 epoch별 시간 측정을 위해 현재 시간을 저장
        epoch_start_time = time.time()
        iter_data_time = time.time()

        # 각 epoch별 누적된 학습량을 표시하기 위한 변수 초기화
        epoch_iter = 0

        # 각 epoch 저장 기록 초기화
        visualizer = Visualizer(opt)
        visualizer.reset()

        with tqdm.tqdm(total=len(dataLoader), ncols=160) as t:
            t.set_description('epoch: {}/{}'.format(epoch, opt.n_epochs + opt.n_epochs_decay))
            # 데이터를 인덱스 번호와 데이터로 구분하여 반복문 진행
            for i, v in enumerate(dataLoader):
                # 각 이미지 데이터별 학습 시간 측정을 위해 현재 시간 저장
                iter_start_time = time.time()

                # 학습 결과 빈도에 따른 결과 저장
                if total_iters % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time

                # 배치사이즈에 따른 빈도 수 갱신
                total_iters += opt.batch_size
                epoch_iter += opt.batch_size

                # 데이터를 device에 등록
                model.set_input(v)

                # 학습 및 가중치 갱신
                model.optimize_parameters()

                # print 빈도에 따른 결과 출력
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                message = visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                # 시간 재 측정 시작
                iter_data_time = time.time()

                # 상태 바 업데이트
                #t.set_postfix(loss=f"{message}")
                t.update(1)

        # epoch 모델 저장에 따른 모델 저장
        if epoch % opt.save_epoch_freq == 0:
            print('\n학습된 모델(네트워크) 저장 중 (epoch %d, iters %d)' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)            
            Eval(epoch)#opt, epoch, dataLoader, opt.batch_size)

        # learning_rate 갱신
        model.update_learning_rate()

        # 학습 종료를 알리는 문구 출력
        print('학습 완료 Epoch %d / %d \t 소요시간: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        title_num = random.randrange(0, 6)
        pro_per = epoch / (opt.n_epochs + opt.n_epochs_decay) * 100

def Eval(epoch):#, epoch, dataset, pre_batch_size):
    """ 평가 함수 """
    global evalopt    
    # 데이터 셋 불러오기
    print("데이터 셋 불러오는 중") 

    # 모델 불러오기
    print("모델 불러오는 중")
    models.createModel("CycleGAN", evalopt)
    model = models.getModel()
    model.setup(evalopt)
    print(evalopt.preprocess)
    # 모델을 평가(테스트)용으로 설정
    model.eval()

    # 평가용 평균 psnr을 측정하기 위한 리스트 선언
    psnr_list = []
    ssim_list = []
    save_dir = os.path.join(evalopt.results_dir, evalopt.name, 'test_latest_iter{}'.format(epoch))

    for i, v in enumerate(evalDataLoader):
        # 테스트 횟수보다 많을 경우 테스트 중지
        if i >= evalopt.num_test:
            break

        # 데이터 장치에 등록 및 테스트
        model.set_input(v)
        model.test()

        # 이미지 관련 정보 불러오기
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        
        # 이미지 저장
        save_images(visuals,save_dir, img_path)

        # 이미지의 psnr수치 비교
        original_img = util.tensor2im(v["B"])
        psnr = model.cal_psnr(original_img)
        ssim = model.cal_ssim(original_img)
        psnr_list.append(psnr)        
        ssim_list.append(ssim)
    avg = sum(psnr_list)/len(psnr_list)
    avg_s = sum(ssim_list)/len(ssim_list)    
    print(f"{epoch}번째 epoch 평과 결과 : 평균 PSNR = {round(avg, 3)} /  평균 SSIM = {round(avg_s, 3)}")

    # 메모장에 저장
    with open("eval_psnr_log.txt", mode="a") as f:
        f.write(str(avg) + " " + str(avg_s) + "\n")

def Test():
    """ 테스트 함수 """
    print("테스트 시작")

# Mode에 따른 진행 유형 결정
if opt.mode == "Train":
    Train()
elif opt.mode == "Test":
    Test()