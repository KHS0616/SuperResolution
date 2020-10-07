"""
편의성을 위한 보조 연산을 하는 파일
"""
# numpy - 다양한 연산을 위한 모듈
# multiprocessing - 멀티프로세싱으로 이미지를 다중 관리하기 위한 모듈
# cv2 - 이미지를 관리하기 위한 모듈
import numpy as np
from multiprocessing import Queue, Process
import cv2

# 파이토치
import torch

# 메인 클래스
class Utility():
    # 초기화 메소드 - 변수 초기화 및 저장
    def __init__(self, args):
        super(Utility, self).__init__()
        # 파서로 부터 정보를 변수에 저장
        # send_image_list - send모드에서 전송을 위한 리스트
        # n_processes - 멀티프로세싱에서 사용할 프로세스의 개수
        # mode - 저장, 전달 선택
        # model_name - 사용하는 모델(네트워크) 이름
        self.send_image_list = []
        self.n_processes = 6
        self.mode = "save"
        self.model_name = args.model_name

    # 최대신호대잡음비(PSNR) 계산
    def calc_psnr(self, img1, img2):
        return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))

    # 이미지를 ycbcr로 변환하고 1개의 채널만 반환
    def preprocess(self, img):
        img = np.array(img).astype(np.float32)
        ycbcr = convert_rgb_to_ycbcr(img)
        x = ycbcr[..., 0]
        x /= 255.
        x = torch.from_numpy(x)
        x = x.unsqueeze(0).unsqueeze(0)
        return x

    # RGB를 ycbcr로 바꾸기위한 함수
    def convert_rgb_to_ycbcr(self, img, dim_order='hwc'):
        if dim_order == 'hwc':
            y = 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.
            cb = 128. + (-37.945 * img[..., 0] - 74.494 * img[..., 1] + 112.439 * img[..., 2]) / 256.
            cr = 128. + (112.439 * img[..., 0] - 94.154 * img[..., 1] - 18.285 * img[..., 2]) / 256.
        else:
            y = 16. + (64.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]) / 256.
            cb = 128. + (-37.945 * img[0] - 74.494 * img[1] + 112.439 * img[2]) / 256.
            cr = 128. + (112.439 * img[0] - 94.154 * img[1] - 18.285 * img[2]) / 256.
        return np.array([y, cb, cr]).transpose([1, 2, 0])

    # 양자화 함수
    # 양자화 - 신호나 정보를 디지털화 하는 작업
    def quantize(self, img, rgb_range):
        # rgb 범위를 이용하여 픽셀 범위를 지정
        pixel_range = 255 / rgb_range

        # 0~255로 값을 조정하고 반올림한다.
        return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

    # 이미지를 다중 프로세서를 이용해서 저장한다.
    # 큐 내부에 있는 이미지를 저장
    def begin_background(self):
        self.queue = Queue()
        def bg_target(queue):
            while True:                 
                if not queue.empty():
                    filename, tensor = queue.get()
                    if filename is None: break
                    cv2.imwrite(filename, tensor.numpy())

        # 프로세스를 생성
        self.process = [
            Process(target=bg_target, args=(self.queue,)) \
            for _ in range(self.n_processes)
        ]
        
        # 프로세스의 개수만큼 프로세스 실행
        for p in self.process: p.start()

    # 이미지 저장 종료
    def end_background(self):
        for _ in range(self.n_processes): self.queue.put((None, None))
        while not self.queue.empty(): time.sleep(1)
        for p in self.process: p.join()

    # 결과 저장
    def save_results(self, save_list, filename):
        for v in save_list:
            normalized = v[0].mul(255 / 255)            
            tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
            print(tensor_cpu)
            # 전송 모드일 경우 이미지를 저장하지 않고 전송 리스트에 담는다.
            if self.mode == "send":
                self.send_image_list.append(tensor_cpu.numpy())
            else:
                self.queue.put(('{}{}{}.png'.format("OutputImage/",self.model_name+"/", filename), tensor_cpu))