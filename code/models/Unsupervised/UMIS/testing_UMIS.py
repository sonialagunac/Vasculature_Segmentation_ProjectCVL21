import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import numpy as np
from morphpoollayer import MorphPool3D
from data.datasets import Directory_Image_Train, Single_Image_Eval
from networks.segmentation import SegmentNet3D_Resnet
from networks.utils import GradXYZ, norm_range
import time

tic = time.time()

# args
parser = argparse.ArgumentParser(description='PyTorch Unsupervised Vessels Segmentation')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='Number of epochs')
parser.add_argument('--batch-size', type=int, default=6, metavar='N',
                    help='#CUDA * batch_size')
parser.add_argument('--lr', type=float, default=1e-4, metavar='N',
                    help='Learning rate')
parser.add_argument('--lmd1', type=int, default=1, metavar='N',
                    help='lambda 1')
parser.add_argument('--lmd2', type=int, default=2, metavar='N',
                    help='lambda 2')
parser.add_argument('--lmd_area', type=float, default=5e-8, metavar='N',
                    help='lambda mean')
parser.add_argument('--range-norm', action='store_true',
                    help='range-norm')
parser.add_argument('--loss', type=str, default='LS', metavar='N',
                    help='Loss function')
parser.add_argument('--train-dataset', type=str, default='InH_split', metavar='N',
                    help='Training dataset name')
parser.add_argument('--val-image-path', type=str, default='/home/slaguna/Documents/semproject/input_data/deepvess_slice/image_test/HaftJavaherian_DeepVess2018_GroundTruthImage_25.tif', metavar='N',
                   help='Validation image path')
parser.add_argument('--val-label-path', type=str, default='/home/slaguna/Documents/semproject/input_data/deepvess_slice/label_test/HaftJavaherian_DeepVess2018_GroundTruthLabel_25.tif', metavar='N',
                  help='Validation label path')
parser.add_argument('--save-valpath', type=str, default='/home/slaguna/Documents/semproject/outputs/UMIS_DV_Epoch1k', metavar='N',
                   help='Saving output prediction path')
parser.add_argument('--validate', default=True, action='store_true',
                    help='validate')

# check point
parser.add_argument('--resume', type=str, default='/itet-stor/slaguna/net_scratch/semproject/UMIS/run/DVSplits/UMIS_DV_Epoch1k/experiment_1/checkpoint.pth.tar',
                        help='add path to restore')
parser.add_argument('--checkname', type=str, default='UMIS_DV_Epoch1k',
                    help='set the checkpoint name')
args = parser.parse_args()

# Data - validation
dataset_val = Single_Image_Eval(image_path=args.val_image_path,
                                label_path=args.val_label_path,
                                data_shape=(32, 128, 128),
                                lables_shape=(32, 128, 128),
                                stride=(8, 16, 16),
                                range_norm=args.range_norm)
dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=2)

# Train
model = SegmentNet3D_Resnet().cuda()
grad_fn = GradXYZ().cuda()
mp3d = MorphPool3D().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# DataParallel
model = torch.nn.DataParallel(model)
grad_fn = torch.nn.DataParallel(grad_fn)
mp3d = torch.nn.DataParallel(mp3d)


checkpoint = torch.load(args.resume)
args.start_epoch = checkpoint['epoch']
model.module.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])
best_pred = checkpoint['best_pred']

print('Model loaded')
is_best = False
best_pred = 0

with torch.no_grad():
   model.eval()
   iterator = tqdm(dataloader_val,
                   leave=True,
                   dynamic_ncols=True,
                   desc='Validation ::')
   input = dataset_val.img[
           dataset_val.effective_lable_idx[0][0]:dataset_val.effective_lable_idx[0][1],
           dataset_val.effective_lable_idx[1][0]:dataset_val.effective_lable_idx[1][1],
           dataset_val.effective_lable_idx[2][0]:dataset_val.effective_lable_idx[2][1]
           ]
   input_gt = dataset_val.lbl[
              dataset_val.effective_lable_idx[0][0]:dataset_val.effective_lable_idx[0][1],
              dataset_val.effective_lable_idx[1][0]:dataset_val.effective_lable_idx[1][1],
              dataset_val.effective_lable_idx[2][0]:dataset_val.effective_lable_idx[2][1]
              ]
   print(input_gt.shape,'input_gt shape')
   input_gt = input_gt // input_gt.max()
   output = np.zeros((1,
                       dataset_val.effective_lable_shape[0],
                       dataset_val.effective_lable_shape[1],
                       dataset_val.effective_lable_shape[2]))
   print(output.shape,'output shape')
   idx_sum = np.zeros((1,
                        dataset_val.effective_lable_shape[0],
                        dataset_val.effective_lable_shape[1],
                        dataset_val.effective_lable_shape[2]))
   for index, (data, lables) in enumerate(iterator):
          # To CUDA
          data = data.cuda()
          lables = lables.cuda()

          # Network
          seg, _ = model(data)

          # Smooth
          seg = mp3d(seg)
          seg = mp3d(seg, True)
          seg = mp3d(seg)
          seg = mp3d(seg, True)
          seg = mp3d(seg)
          seg = mp3d(seg, True)

          for batch_idx, val in enumerate(seg[:, 0]):
              out_i = index * dataloader_val.batch_size + batch_idx
              z, y, x = np.unravel_index(out_i, (dataset_val.dz, dataset_val.dy, dataset_val.dx))
              z = z * dataset_val.stride[0]
              y = y * dataset_val.stride[1]
              x = x * dataset_val.stride[2]

              idx_sum[0,
              z: z + dataset_val.lables_shape[0],
              y: y + dataset_val.lables_shape[1],
              x: x + dataset_val.lables_shape[2]] += 1

              output[0,
              z: z + dataset_val.lables_shape[0],
              y: y + dataset_val.lables_shape[1],
              x: x + dataset_val.lables_shape[2]] += val.cpu().data.numpy()
   output = output / idx_sum
   output = torch.Tensor(output).unsqueeze(0).cuda()
   input_gt = torch.Tensor(input_gt).unsqueeze(0)

   output_np = output.detach().cpu().numpy()
   input_np = input_gt.detach().cpu().numpy()
   toc = time.time()
   print("The time needed to run inference is: ", toc-tic, "seconds")

   np.save(args.save_valpath + 'output.npy', output_np)
   np.save(args.save_valpath + 'input.npy', input_np)
