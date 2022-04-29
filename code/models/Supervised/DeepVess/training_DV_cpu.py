import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os
import numpy as np
from data.datasets import Directory_Image_Train, Single_Image_Eval
from networks.segmentation import DeepVess, VessNN
from utils import Saver

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# args
parser = argparse.ArgumentParser(description='PyTorch Supervised Vessels Segmentation')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='Number of epochs')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='batch_size')
parser.add_argument('--lr', type=float, default=1e-4, metavar='N',
                    help='Learning rate')
parser.add_argument('--range-norm', action='store_true',
                    help='range-norm')
parser.add_argument('--train-dataset', type=str, default='DeepVess', metavar='N',
                    help='Training dataset name')
parser.add_argument('--train-images-path', type=str, default=r'C:\Users\sonil\PycharmProjects\Semester project\DeepVess\input_data\deepvess_slice\image_train', metavar='N',
                    help='Training dataset images path')
parser.add_argument('--train-labels-path', type=str, default=r'C:\Users\sonil\PycharmProjects\Semester project\DeepVess\input_data\deepvess_slice\label_train', metavar='N',
                    help='Training dataset labels path')
parser.add_argument('--save-valpath', type=str, default=r'C:\Users\sonil\PycharmProjects\Semester_project\outputs\DeepVess_DV_Epoch1k', metavar='N',
                   help='Saving loss')
parser.add_argument('--validate', action='store_true',
                    help='validate')

# checking point
parser.add_argument('--resume', type=str, default=None,
                    help='put the path to resuming file if needed')
parser.add_argument('--checkname', type=str, default='DeepVess_DV_Epoch1k',
                    help='set the checkpoint name')
args = parser.parse_args()

# Define Saver
saver = Saver(args)
saver.save_experiment_config()


# Data
dataset = Directory_Image_Train(images_path=args.train_images_path,
                                labels_path=args.train_labels_path,
                                max_iter=20000,
                                range_norm=args.range_norm,
                                data_shape=(7, 33, 33),
                                lables_shape=(1, 4, 4)
                                )
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
loss_list=[] #Saving loss

# Train
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
if __name__ == '__main__':
    for epoch in range(args.epochs):
        model.train()
        iterator = tqdm(dataloader,
                        leave=True,
                        dynamic_ncols=True)
        for i, (data, lables) in enumerate(iterator):
            # To CPU
            data = data.to(device)
            lables = lables.to(device)

            # Network
            seg = model(data)

            # Loss
            loss = CE(seg, lables)
            loss_list.append(loss.item())

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print loss
            iterator.set_description(
                'Epoch [{epoch}/{epochs}] :: Train Loss {loss:.4f}'.format(epoch=epoch, epochs=args.epochs,
                                                                           loss=loss.item()))
        is_best = False
        best_pred = 123
        if (epoch+1) % 100 == 0:
            saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_pred': best_pred,
            }, is_best)

    lst = np.array(loss_list)
    np.save(args.save_valpath + 'loss.npy', lst)