# -*- coding: utf-8 -*- 
import numpy as np
import os
import sys
import ntpath
import time
from . import util
from subprocess import Popen, PIPE


if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError

# 저장 설정
def save_images(visuals, image_raw_path, image_path, aspect_ratio=1.0, width=256):
    """ 이미지 저장을 위한 설정 """
    # 이미지 정보를 분해해서 저장
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]
    ims, txts, links = [], [], []

    # 이미지를 저장할 폴더가 존재하지 않으면 생성
    if not os.path.exists(image_raw_path):
        os.makedirs(image_raw_path)

    # model 에서 지정한 visual 변수 항목들을 이미지로 저장
    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_raw_path, image_name)
        util.save_image(im, save_path, aspect_ratio=aspect_ratio)

# 기록을 저장하는 클래스
class Visualizer():
    """ 학습 및 테스트 기록 저장 클래스 """

    def __init__(self, opt):
        """ 생성자 """
        # 옵션 저장
        self.opt = opt
        
        # 옵션 불러오기
        # opt.name - 결과가 저장될 폴더 이름
        self.name = opt.name
        self.saved = False

        # Loss 로그 파일이 저장될 폴더 설정
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')

        # Loss 로그 파일 제목 작성
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """ 저장 상태를 초기화 """
        self.saved = False

    # ???
    # def create_visdom_connections(self):
    #     """If the program could not connect to Visdom server, this function will start a new server at port < self.port > """
    #     cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
    #     print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
    #     print('Command: %s' % cmd)
    #     Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    # Loss 출력
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """ 현재 Loss를 출력 """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        # 로그 파일에 현재 기록 저장
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)
        return message
