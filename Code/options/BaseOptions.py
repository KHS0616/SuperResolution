"""
기본적인 옵션을 설정하는 모듈
"""
# argparse - 파서를 등록하기 위한 모듈
import argparse

class BaseOptions():
    """
    기본 옵션 설정 메인 클래스
    """
    def __init__(self):
        # 파서 선언
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # 파서 내용 등록
        self.defineOptions(self.parser)

    def defineOptions(self, parser):
        """ 파서 내용을 등록하는 메소드 """
        # 경로 관련 파라미터
        parser.add_argument("--dataroot", required=True, help="데이터 경로, 해당 경로 내부에는 Train or Eval or Test 폴더가 있어야 한다.")
        parser.add_argument("--name", type=str, default="experiment_name", help="프로젝트 이름, 결과가 저장될 폴더의 이름")
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='모델이 저장될 체크포인트 경로 결정')
        parser.add_argument('--results_dir', type=str, default='./results/', help='결과를 저장할 경로')

        # 하드웨어 관련 파라미터
        parser.add_argument('--gpu_ids', type=str, default='5', help='사용할 GPU 번호 결정: e.g. 0  0,1,2, 0,2. use -1 for CPU')

        # 학습 및 테스트 관련 공통 파라미터
        parser.add_argument('--n_epochs', type=int, default=1000, help='초기 lr을 이용하여 학습할 Epoch')
        parser.add_argument("--mode", type=str, default="Train", help="학습인지, 테스트인지 설정 [Train, Test]")
        parser.add_argument("--batch_size", type=int, default="1", help="학습 및 테스트 배치사이즈 결정")
        parser.add_argument("--shuffle", action="store_true", help="활성화 하면 데이터를 섞는다")
        parser.add_argument("--pin_memory", type=bool, default=True, help="GPU 학습시 성능 향상")
        parser.add_argument("--n_threads", type=int, default=0, help="데이터 로딩시 사용할 스레드 수")
        parser.add_argument('--input_nc', type=int, default=3, help='입력 이미지의 채널 설정')
        parser.add_argument('--output_nc', type=int, default=3, help='출력 이미지의 채널 설정')

        parser.add_argument('--epoch', type=str, default='latest', help='불러올 모델의 epoch 수치를 결정')
        parser.add_argument('--continue_train', action='store_true', help='활성화 하면 기존 모델을 불러온다')
        parser.add_argument('--epoch_count', type=int, default=1, help='몇 번째 Epoch 부터 학습을 진행할 지 결정')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')

        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='데이터 셋의 최대 크기')
        parser.add_argument('--save_epoch_freq', type=int, default=10, help='모델을 저장할 Epoch 간격')
        parser.add_argument('--preprocess', type=str, default='crop', help='이미지 전처리 옵션 결정 [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
        parser.add_argument('--no_flip', action='store_true', help='이미지를 뒤집는 여부 결정')

    def getOptions(self):
        """ 파서에 등록된 옵션을 가져오는 메소드 """
        # 파서에 등록된 정보를 옵션 변수로 불러오기
        opt, _ = self.parser.parse_known_args()

        # 옵션 반환
        return opt
