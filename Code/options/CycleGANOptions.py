from options.BaseOptions import BaseOptions

class CycleGANOptions(BaseOptions):
    def __init__(self):
        super().__init__()
        self.addOptions()

    def addOptions(self):
        """ 파서 옵션 추가 정의 메소드 """
        # 전처리 관련 옵션
        
        self.parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
        self.parser.add_argument('--load_size', type=int, default=880, help='학습을위해 조절할 이미지의 크기')        
        self.parser.add_argument('--crop_size', type=int, default=160, help='전처리 과정에서 crop size 결정')        

        # 네트워크 관련 옵션
        self.parser.add_argument('--ngf', type=int, default=64, help='Generation CNN 층의 필터 개수')
        self.parser.add_argument('--ndf', type=int, default=64, help='Discrimination CNN 층의 필터 개수')
        self.parser.add_argument('--netD', type=str, default='basic', help='Discriminator 네트워크 설정 [basic | n_layers | pixel].')
        self.parser.add_argument('--netG', type=str, default='resnet_9blocks', help='Generator 네트워크 설정 [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')


        # 학습 관련 파라미터
        # n_epochs - 학습 epochs 설정
        # n_epochs_decay - learning rate decay를 위한 epohchs 설정
        # lr - Learning Rate 초기 수치        
        self.parser.add_argument('--n_epochs_decay', type=int, default=1000, help='Learning Rate Decay Epoch')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='Adam에서 사용되는 Momentum의 관성 수치(대부분 0.9..)')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='ADAM의 첫 Learning_Rate 수치')
        self.parser.add_argument('--gan_mode', type=str, default='lsgan', help='GAN에서 사용되는 Objective Function. [vanilla| lsgan | wgangp]')
        self.parser.add_argument('--pool_size', type=int, default=50, help='이전에 생성된 GAN 이미지를 버퍼에 임시저장하는 크기(갯수)')
        self.parser.add_argument('--lr_policy', type=str, default='linear', help='Learning Rate 갱신 규칙..?. [linear | step | plateau | cosine]')
        self.parser.add_argument('--lr_decay_iters', type=int, default=50, help='Learning Rate Decay 수치')
        self.parser.add_argument('--no_dropout', action='store_true', help='Generator Dropout 여부 결정')

        self.parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
        self.parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
        self.parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
        
        self.parser.add_argument('--print_freq', type=int, default=100, help='학습결과를 출력할 빈도')
        self.parser.add_argument('--num_test', type=int, default=50, help='테스트 할 횟수')
        self.parser.add_argument("--model", type=str, default="CycleGAN", help="모델 이름")
        self.parser.set_defaults(no_dropout=True)