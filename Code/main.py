import sys
from importlib import import_module

# 모델 리스트
model_list = ["FSRCNN", "EDSR", "CycleGAN"]

# 메인 구문
if __name__ == '__main__':
    try:
        # 모델 이름 선택
        index = int(input("---- 학습할 모델 이름을 선택하세요 ----\n 1. FSRCNN 2. EDSR  3. CycleGAN\n"))
        
        # 선택한 모듈 불러오기
        module_name = "mains." + model_list[index-1] + "_Main"
        module = import_module(module_name)
        
    except IndexError:
        print("제시된 범위 내에서 모델을 선택하세요")