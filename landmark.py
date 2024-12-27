import warnings

# from sklearn.metrics import confusion_matrix, plot_confusion_matrix
warnings.filterwarnings("ignore")
import argparse
import datetime
from models.PosterV2_7cls import *
from models.seesaw_shuffleFaceNet import seesaw_shuffleFaceNet
from collections import OrderedDict

now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H-%M]-")

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default=r'data/raf-db/')
parser.add_argument('--data_type', default='RAF-DB', choices=['RAF-DB', 'AffectNet-7', 'CAER-S'],
                        type=str, help='dataset option')
parser.add_argument('--checkpoint_path', type=str, default='./checkpoint/' + time_str + 'model.pth')
parser.add_argument('--best_checkpoint_path', type=str, default='./checkpoint/' + time_str + 'model_best.pth')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=144, type=int, metavar='N')
parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')

parser.add_argument('--lr', '--learning-rate', default=0.000035, type=float, metavar='LR', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=30, type=int, metavar='N', help='print frequency')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to checkpoint')
parser.add_argument('-e', '--evaluate', default='checkpoint/raf-db-model_best.pth', type=str, help='None for train, evaluate model on test set')
parser.add_argument('--beta', type=float, default=0.6)
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()

# traindir = os.path.join(args.data, 'train')
#
# train_dataset = datasets.ImageFolder(traindir,
#                                                  transforms.Compose([transforms.Resize((224, 224)),
#                                                                      transforms.RandomHorizontalFlip(),
#                                                                      transforms.ToTensor(),
#                                                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                                                                           std=[0.229, 0.224, 0.225]),
#                                                                      transforms.RandomErasing(scale=(0.02, 0.1))]))
#
#
# face_landback = MobileFaceNet([112, 112], 136)
# face_landback_checkpoint = torch.load(r'models/pretrain/mobilefacenet_model_best.pth',
#                                       map_location=lambda storage, loc: storage)
# face_landback.load_state_dict(face_landback_checkpoint['state_dict'])
#
# for param in face_landback.parameters():
#     param.requires_grad = False
#
#
# dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
# for images, target_landmarks in dataloader:
#     with torch.no_grad():
#         x_face = F.interpolate(images, size=112)
#         x_face1 , x_face2, x_face3 = face_landback(x_face)
#         # x_face1 , x_face2, x_face3 [144, 64, 28, 28] [144, 128, 14, 14] [144, 512, 7, 7]
#         break

x = torch.rand(3,3,112,112)
# backbone = Backbone(50, 0.0, 'ir')
backbone = seesaw_shuffleFaceNet(512)
modelDict = torch.load('models/pretrain/seesawfacenet.pth')
prefix = "module."
new_odict = OrderedDict((k[len(prefix):], v) for k,v in modelDict.items() if k.startswith(prefix))

backbone.load_state_dict(new_odict, strict=False)
out = backbone(x)
print(out.shape)