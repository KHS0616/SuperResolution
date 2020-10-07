"""
실행을 담당하는 파일

----2020-09-18 사용 가능 기능----
FSRCNN 학습 및 평가
EDSR 테스트
"""
# 파이토치 모듈
import torch

# 내부 모듈
# data - 데이터를 관리하는 모듈
# model - 모델(네트워크)를 관리하는 모듈
# option.args - 파서를 관리하는 모듈
# trainer.Trainer - 학습, 테스트를 실행하는 모듈
import data
import model
from option import args
from trainer import Trainer

# 랜덤한 값을 특정 시드에 저장
torch.manual_seed(0)

def main():
    global model 
    # Data 객체 생성
    # Data 내부 __init__실행
    loader = data.Data(args)

    # Model 객체 생성
    # model 내부 __init__실행 
    _model = model.Model(args)

    # Loss 객체 생성
    # Loss 내부 __init__ 실행
    # args.test_only - 테스트를 위한 실행인지 아닌지를 확인하는 bool 타입 변수
    # 테스트 단계일 경우 해당 과정 생략
    #_loss = loss.Loss(args, checkpoint) if not args.test_only else None
   
    # 학습을 위한 객체를 생성하고 모델, 손실함수 등을 전달한다.
    t = Trainer(args, loader, _model, "dd")

    # 학습 및 테스트 시작
    if args.type == "Train":
        t.train()
    else:        
        t.test()

if __name__ == '__main__':
    main()
