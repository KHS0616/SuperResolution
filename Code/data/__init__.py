""" 데이터 셋 모듈 """

from importlib import import_module
from torch.utils.data import DataLoader

def createDataset(module_name, opt):
    """ 데이터 셋 생성 함수 """
    # 데이터 셋으로 부터 데이터 로더를 생성
    data_loader = CustomDatasetDataLoader(module_name, opt)

    # 데이터 로더로 부터 가공된 데이터 셋을 불러온다.
    dataset = data_loader.load_data()

    # 데이터 셋 반환
    return dataset

# 사용자 정의 데이터 셋을 만드는 데이터 로더
class CustomDatasetDataLoader():
    def __init__(self, model_name, opt):
        # 파서 옵션 불러와서 저장
        self.opt = opt

        # 해당되는 모듈 불러오기
        module = import_module("data." + model_name + "Dataset")

        # 데이터 셋 객체 생성
        self.dataset = getattr(module, opt.mode + "Dataset")(opt)
        print("데이터 셋 - [%s] 이 생성되었습니다." % type(self.dataset).__name__)

        # 데이터 로더 생성
        # 파이토치 DataLoader 모듈 이용
        # opt.batch_size - 배치 사이즈
        # opt.serial_batches - 활성화 하면 데이터를 순차적으로 처리
        # num_workers - 학습에 사용될 스레드 할당
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=opt.shuffle,
            pin_memory=opt.pin_memory,
            num_workers=opt.n_threads)

    def load_data(self):
        return self

    def __len__(self):
        """데이터 셋의 길이를 반환한다."""
        # max_dataset_size - 실제 데이터 셋의 길이의 한계 값을 설정
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """ 데이터 셋에서 배치사이즈에 따라 하나씩 분배 """
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data