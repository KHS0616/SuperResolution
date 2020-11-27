from options.BaseOptions import BaseOptions

class EDSROptions(BaseOptions):
    def __init__(self):
        super().__init__()
        self.addOptions()

    def addOptions(self):
        """ 파서 옵션 추가 정의 메소드 """
        # 전처리 관련 옵션
        self.parser.add_argument('--crop_size', type=int, default=160, help='전처리 과정에서 crop size 결정')
        self.parser.add_argument('--scale', type=int, default=2, help='스케일 요소 설정 [2, 3, 4]')
        self.parser.add_argument("--rgb_range", type=int, default=255, help="rgb range")
        self.parser.add_argument("--n_colors", type=int, default=3, help="channel input image")
        self.parser.add_argument("--model", type=str, default="EDSR", help="모델 이름")
        self.parser.add_argument("--no_chop", action="store_true", help="chop 여부 확인")