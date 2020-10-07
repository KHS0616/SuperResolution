"""
데이터를 관리하는 최상위 파일
"""

# importlib.import_module - 코드 내부에서 모듈을 가져오는 모듈
# os.listdir 지정된 경로에 있는 파일들을 리스트로 관라하기 위한 모듈
from importlib import import_module
from os import listdir

# 파이토치
from torch.utils.data import dataloader
from torch.utils.data import ConcatDataset

# Data 클래스
class Data():
    def __init__(self, args):
        # 파서를 통해 정보를 변수에 저장
        # data_type - 데이터의 유형(폴더, 비디오 등등)
        # *_batch_size - 각각의 배치사이즈
        # pin_memory - ??
        # n_threads - 데이터를 로드할 때 사용하는 스레드 개수
        self.data_type = args.data_type
        self.type = args.type
        self.train_batch_size = args.train_batch_size
        self.test_batch_size = args.test_batch_size
        self.pin_memory = args.pin_memory
        self.n_threads = args.n_threads        

        # 데이터의 유형에 맞는 Dataset 가져오기
        # 학습하는 상황에서는 학습 및 평가 데이터 셋을 저장한다.
        # 테스트하는 상황에서는 테스트 데이터 셋을 저장한다.
        m = import_module('data.' + self.data_type.lower())
        if self.type == "Train":
            dataset = getattr(m, "TrainDataset")(args)
            self.loader = dataloader.DataLoader(
                dataset=dataset,
                batch_size=self.train_batch_size,
                shuffle=False,
                pin_memory=self.pin_memory,
                num_workers=self.n_threads
            )
            eval_dataset = getattr(m, "EvalDataset")(args)
            self.eval_loader = dataloader.DataLoader(
                dataset=eval_dataset,
                batch_size=self.test_batch_size,
                shuffle=False,
                pin_memory=self.pin_memory,
                num_workers=self.n_threads
            )
        elif self.type == "Test":
            dataset = getattr(m, "TestDataset")(args)
            self.loader = dataloader.DataLoader(
                dataset=dataset,
                batch_size=self.test_batch_size,
                shuffle=False,
                pin_memory=self.pin_memory,
                num_workers=self.n_threads
            )
