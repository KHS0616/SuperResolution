"""
파서의 내용을 정의하는 코드
"""

import argparse

# 파서 등록 시작
parser = argparse.ArgumentParser()

# 데이터 관련 파서
# data_type - 데이터 셋이 어떤 유형인지 표시(폴더, 영상 등)
# *_dataset_dir - 각각의 데이터 셋이 위치한 폴더의 이름(기본적으로 학습 및 평가는 TrainImage, 테스트는 InputImage 내부에 위치)
parser.add_argument("--data_type", type=str, default="Folder", help="Input yout dataset type ex: video, folder, ...")
parser.add_argument("--train_dataset_dir", type=str, default="DIV2K_train_HR", help="train dataset folder name")
parser.add_argument("--eval_dataset_dir", type=str, default="DIV2K_valid_HR", help="eval dataset folder name")
parser.add_argument("--test_dataset_dir", type=str, default="dd", help="test dataset folder name")

# 모델 관련 파서
# model_name - 모델(네트워크)의 이름
# load_model_name - 불러올 모델의 이름(default값이 None이고 문자열이 입력되면 해당 파일을 불러온다.)
parser.add_argument("--model_name", type=str, default="FSRCNN", help="super resolution network-model")
parser.add_argument("--load_model_name", type=str, default=None, help="model file name to load")
#standard-x2r32c256.pt

# 하드웨어 관련 설정
parser.add_argument("--pin_memory", type=bool, default=True, help="GPU train or test speed up")
parser.add_argument("--n_threads", type=int, default=6, help="number of threads for data loading")
parser.add_argument('--n_GPUs', type=int, default=1, help='number of GPUs')

# 학습 관련 설정
# type - 학습인지 테스트인지 구분한다.
# *.batch_size - 학습 및 테스트 진행시 배치사이즈
# crop_size - 학습과정에서 crop을 이용한 전처리 과정에서 이미지를 자르는 크기
# scale - 업 스케일링 수치
# chop - 학습 및 테스트 진행단계에서 이미지를 분할하여 학습할지 여부
# rgb_range - rgb 범위 측정 default 255
# n_colors - 입력하는 이미지의 채널의 개수 설정 default 3
# num_epoch - 학습 반복 횟수
parser.add_argument("--type", type=str, default="Train", help="Select Train or Test")
parser.add_argument("--train_batch_size", type=int, default=64, help="number of train batch size")
parser.add_argument("--test_batch_size", type=int, default=1, help="number of test batch size")
parser.add_argument("--crop_size", type=int, default=88, help="size of image crop for training")
parser.add_argument("--scale", type=int, default=2, help="super resolution scale")
parser.add_argument("--chop", action="store_true", help="divide image")
parser.add_argument("--rgb_range", type=int, default=255, help="rgb range")
parser.add_argument("--n_colors", type=int, default=3, help="channel input image")
parser.add_argument("--num_epoch", type=int, default=100, help="training epoch")
# parser.add_argument('--debug', action='store_true',
#                     help='Enables debug mode')
# parser.add_argument('--template', default='.',
#                     help='You can set various templates in option.py')

# Hardware specifications
# parser.add_argument('--cpu', action='store_true',
#                     help='use cpu only') 

# parser.add_argument('--seed', type=int, default=1,
#                     help='random seed')
 
# # 데이터 정보에 관련된 속성
# # data_train - 학습 대상이 되는 데이터 셋의 이름
# # data_test - 테스트에 사용되는 데이터 셋의 이름
# # data_range - 학습/테스트에 사용되는 데이터 셋의 범위(개수)
# # ext - 데이터 확장자
# parser.add_argument('--dir_data', type=str, default='../../../dataset',
#                     help='dataset directory')
# parser.add_argument('--dir_demo', type=str, default='../test',
#                     help='demo image directory')
# parser.add_argument('--data_train', type=str, default='DIV2K',
#                     help='train dataset name')
# parser.add_argument('--data_test', type=str, default='DIV2K',
#                     help='test dataset name')
# parser.add_argument('--data_range', type=str, default='1-800/801-810',
#                     help='train/test data range')
# parser.add_argument('--ext', type=str, default='sep',
#                     help='dataset file extension')
# parser.add_argument('--scale', type=str, default='4',
#                     help='super resolution scale')
# parser.add_argument('--patch_size', type=int, default=192,
#                     help='output patch size')
# parser.add_argument('--rgb_range', type=int, default=255,
#                     help='maximum value of RGB')
# parser.add_argument('--n_colors', type=int, default=3,
#                     help='number of color channels to use')
# parser.add_argument('--chop', action='store_true',
#                     help='enable memory-efficient forward')
# parser.add_argument('--no_augment', action='store_true',
#                     help='do not use data augmentation')

# # Model specifications
# parser.add_argument('--model', default='EDSR',
#                     help='model name')

# parser.add_argument('--act', type=str, default='relu',
#                     help='activation function')
# parser.add_argument('--pre_train', type=str, default='',
#                     help='pre-trained model directory')
# parser.add_argument('--extend', type=str, default='.',
#                     help='pre-trained model directory')
# parser.add_argument('--n_resblocks', type=int, default=32,
#                     help='number of residual blocks')
# parser.add_argument('--n_feats', type=int, default=256,
#                     help='number of feature maps')
# # 픽셀 정보
# parser.add_argument('--res_scale', type=float, default=0.1,
#                     help='residual scaling')
# parser.add_argument('--shift_mean', default=True,
#                     help='subtract pixel mean from the input')
# parser.add_argument('--dilation', action='store_true',
#                     help='use dilated convolution')
# parser.add_argument('--precision', type=str, default='single',
#                     choices=('single', 'half'),
#                     help='FP precision for test (single | half)')

# # Option for Residual dense network (RDN)
# parser.add_argument('--G0', type=int, default=64,
#                     help='default number of filters. (Use in RDN)')
# parser.add_argument('--RDNkSize', type=int, default=3,
#                     help='default kernel size. (Use in RDN)')
# parser.add_argument('--RDNconfig', type=str, default='B',
#                     help='parameters config of RDN. (Use in RDN)')

# # Option for Residual channel attention network (RCAN)
# parser.add_argument('--n_resgroups', type=int, default=10,
#                     help='number of residual groups')
# parser.add_argument('--reduction', type=int, default=16,
#                     help='number of feature maps reduction')

# # 학습, 테스트에 관련된 속성
# # test_only - 테스트만 하기 위함인지 확인하기 위한 속성
# parser.add_argument('--reset', action='store_true',
#                     help='reset the training')
# parser.add_argument('--test_every', type=int, default=10,
#                     help='do test per every N batches')
# parser.add_argument('--epochs', type=int, default=300,
#                     help='number of epochs to train')
# parser.add_argument('--batch_size', type=int, default=16,
#                     help='input batch size for training')
# parser.add_argument('--split_batch', type=int, default=1,
#                     help='split the batch into smaller chunks')
# parser.add_argument('--self_ensemble', action='store_true',
#                     help='use self-ensemble method for test')
# parser.add_argument('--test_only', action='store_true',
#                     help='set this option to test the model')
# parser.add_argument('--gan_k', type=int, default=1,
#                     help='k value for adversarial loss')

# # Optimization specifications
# parser.add_argument('--lr', type=float, default=1e-4,
#                     help='learning rate')
# parser.add_argument('--decay', type=str, default='200',
#                     help='learning rate decay type')
# parser.add_argument('--gamma', type=float, default=0.5,
#                     help='learning rate decay factor for step decay')
# parser.add_argument('--optimizer', default='ADAM',
#                     choices=('SGD', 'ADAM', 'RMSprop'),
#                     help='optimizer to use (SGD | ADAM | RMSprop)')
# parser.add_argument('--momentum', type=float, default=0.9,
#                     help='SGD momentum')
# parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
#                     help='ADAM beta')
# parser.add_argument('--epsilon', type=float, default=1e-8,
#                     help='ADAM epsilon for numerical stability')
# parser.add_argument('--weight_decay', type=float, default=0,
#                     help='weight decay')
# parser.add_argument('--gclip', type=float, default=0,
#                     help='gradient clipping threshold (0 = no clipping)')

# # Loss 관련 속성
# # loss - 어떤 Loss 함수를 사용할지 저장되는 속성
# parser.add_argument('--loss', type=str, default='1*L1',
#                     help='loss function configuration')
# parser.add_argument('--skip_threshold', type=float, default='1e8',
#                     help='skipping batch that has large error')

# # 저장, 불러오기에 관련된 속성
# # load - 저장된 모델 파일의 이름(경로)
# # save_gt - LR 이미지와 HR 이미지를 SR 이미지와 같이 저장할지 여부
# # resume - 모델을 다운로드 받을지, 실제 모델을 사용할 지 결정
# parser.add_argument('--save', type=str, default='test',
#                     help='file name to save')
# parser.add_argument('--load', type=str, default='',
#                     help='file name to load')
# parser.add_argument('--resume', type=int, default=0,
#                     help='resume from specific checkpoint')
# parser.add_argument('--save_models', action='store_true',
#                     help='save all intermediate models')
# parser.add_argument('--print_every', type=int, default=100,
#                     help='how many batches to wait before logging training status')
# parser.add_argument('--save_results', action='store_true',
#                     help='save output results')
# parser.add_argument('--save_gt', action='store_true',
#                     help='save low-resolution and high-resolution images together')

args = parser.parse_args()
#template.set_template(args)

# scale, data_train, data_test 항목들을 +를 기준으로 스플릿하여 리스트로 저장
# args.scale = list(map(lambda x: int(x), args.scale.split('+')))
# args.data_train = args.data_train.split('+')
# args.data_test = args.data_test.split('+')

# if args.epochs == 0:
#     args.epochs = 1e8

# for arg in vars(args):
#     if vars(args)[arg] == 'True':
#         vars(args)[arg] = True
#     elif vars(args)[arg] == 'False':
#         vars(args)[arg] = False

