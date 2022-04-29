import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os
import numpy as np
from data.datasets import Single_Image_Eval
from networks.segmentation import DeepVess, VessNN
import time

tic = time.time()
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# args
parser = argparse.ArgumentParser(description='PyTorch Supervised Vessels Segmentation')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='Number of epochs')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='#CUDA * batch_size')
parser.add_argument('--lr', type=float, default=1e-4, metavar='N',
                    help='Learning rate')
parser.add_argument('--range-norm', action='store_true',
                    help='range-norm')
parser.add_argument('--train-dataset', type=str, default='DeepVess', metavar='N',
                    help='Training dataset name')
parser.add_argument('--val-image-path', type=str, default=r'C:\Users\sonil\PycharmProjects\Semester_project\DeepVess\input_data\deepvess_slice\image_test\HaftJavaherian_DeepVess2018_GroundTruthImage_25.tif', metavar='N',
                    help='Validation image path')
parser.add_argument('--val-label-path', type=str, default=r'C:\Users\sonil\PycharmProjects\Semester_project\DeepVess\input_data\deepvess_slice\label_test\HaftJavaherian_DeepVess2018_GroundTruthLabel_25.tif', metavar='N',
                   help='Validation label path')
parser.add_argument('--save-valpath', type=str, default=r'C:\Users\sonil\PycharmProjects\Semester_project\outputs\DeepVess_DV_Epoch1k', metavar='N',
                   help='Saving output prediction path')
parser.add_argument('--validate', action='store_false',
                    help='validate')

# check point
parser.add_argument('--resume', type=str, default=r'C:\Users\sonil\PycharmProjects\Semester_project\DeepVess\run\DVSplits\DeepVess_DV_Epoch1k\experiment_1\checkpoint.pth.tar',
                     help='put the path to resuming file if needed')
parser.add_argument('--checkname', type=str, default='DeepVess_DV_Epoch1k',
                    help='set the checkpoint name')
args = parser.parse_args()

# Data - validation
dataset_val = Single_Image_Eval(image_path=args.val_image_path,
                                label_path=args.val_label_path,
                                data_shape=(7, 33, 33),
                                lables_shape=(1, 4, 4),
                                stride=(1, 1, 1),
                                range_norm=args.range_norm)
dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=2)
print("This says if I'm going to validate: ", args.validate)

# Train in CPU
model = DeepVess().to(device)
CE = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# DataParallel
model = torch.nn.DataParallel(model)

if args.resume:
    if not os.path.isfile(args.resume):
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    args.start_epoch = checkpoint['epoch']
    model.module.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    best_pred = checkpoint['best_pred']


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
    input_gt = input_gt // input_gt.max()

    output = np.zeros((1,
                       dataset_val.effective_lable_shape[0],
                       dataset_val.effective_lable_shape[1],
                       dataset_val.effective_lable_shape[2]))
    idx_sum = np.zeros((1,
                        dataset_val.effective_lable_shape[0],
                        dataset_val.effective_lable_shape[1],
                        dataset_val.effective_lable_shape[2]))

    for index, (data, lables) in enumerate(iterator):
        # To CPU
        data = data.to(device)
        lables = lables.to(device)

        # Network
        seg = model(data)
        seg = F.softmax(seg, 1, _stacklevel=5)

        _, pred_idx = seg.max(1)

        for batch_idx, val in enumerate(seg[:, 1]):
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
    output = torch.Tensor(output).unsqueeze(0)
    input_gt = torch.Tensor(input_gt).unsqueeze(0)

    print("This is the output shape: ", output)
    print("This is the input shape: ", input_gt)

    output_np = output.detach().cpu().numpy()
    input_np = input_gt.detach().cpu().numpy()
    toc = time.time()
    print("The time needed to run inference is: ", toc - tic, "seconds")

    np.save(args.save_valpath + 'output.npy', output_np)
    np.save(args.save_valpath + 'input.npy', input_np)

