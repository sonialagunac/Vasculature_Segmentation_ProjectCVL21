# Copyright 2019-2020, Mohammad Haft-Javaherian. (mh973@cornell.edu).
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#   References:
#   -----------
#   [1] Haft-Javaherian, M., Villiger, M., Schaffer, C. B., Nishimura, N., Golland, P., & Bouma, B. E. (2020).
#       A Topological Encoding Convolutional Neural Network for Segmentation of 3D Multiphoton Images of Brain
#       Vasculature Using Persistent Homology. In Proceedings of the IEEE/CVF Conference on Computer Vision and
#       Pattern Recognition Workshops (pp. 990-991).
#       http://openaccess.thecvf.com/content_CVPRW_2020/html/w57/Haft-Javaherian_A_Topological_Encoding_Convolutional_Neural_Network_for_Segmentation_of_3D_CVPRW_2020_paper.html
# =============================================================================

from __future__ import print_function


import sys
from random import shuffle
import itertools as it
import tifffile as T
import numpy as np
from six.moves import range
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils import data
from topologylayer.nn import LevelSetLayer2D, SumBarcodeLengths, PartialSumBarcodeLengths
from tqdm import tqdm
import time

tic = time.time()

lr = 1e-4
# Change isTrain to True if you want to train the network
isTrain = False
# Change isForward to True if you want to test the network
isForward = True
# padSize is the padding around the central voxel to generate the field of view
padSize = ((3, 3), (48, 49), (48, 49), (0, 0))
WindowSize = np.sum(padSize, axis=1) + 1
# pad Size around the central voxel to generate 2D region of interest
corePadSize = 10
# number of epoch to train
nEpoch = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
# The input h5 file location and batch size
# inputData = sys.argv[1] if len(sys.argv) > 1 else input("Enter h5 input file path (e.g. ../a.h5)> ")
# batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 500
batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 50 #changed it for memory purposes

# Import Data
path_save_output='/home/slaguna/Documents/semproject/outputs/TopoDeepVess_DV'
path_img_test='/home/slaguna/Documents/semproject/input_data/deepvess_slice/image_test/HaftJavaherian_DeepVess2018_GroundTruthImage_25.tif'
path_img='/home/slaguna/Documents/semproject/input_data/HaftJavaherian_DeepVess2018_GroundTruthImage.tif'
path_lbl='/home/slaguna/Documents/semproject/input_data/HaftJavaherian_DeepVess2018_GroundTruthLabel.tif'


if isTrain:
    im=T.imread(path_img)
    im_95 = np.percentile(im, 95)
    im[im >= im_95] = im_95
    im = np.divide(im, im_95)-0.5
    im = im.reshape(im.shape + (1,))
    imSize = im.size
    imShape = im.shape

    l=T.imread(path_lbl)
    l[l == 2] = 0
    l=np.divide(l,np.amax(l))
    l = l.reshape(l.shape + (1,))

    nc = im.shape[1]
    tst = im[:, :(nc // 4), :]
    tstL = l[:, :(nc // 4), :]
    trn = im[:, (nc // 2):, :]
    trnL = l[:, (nc // 2):, :]
    tst = np.pad(tst, padSize, 'symmetric')
    trn = np.pad(trn, padSize, 'symmetric')

if isForward:
    im=T.imread(path_img_test)
    im_95 = np.percentile(im, 95)
    im[im >= im_95] = im_95
    im = np.divide(im, im_95)-0.5
    im = im.reshape(im.shape + (1,))
    imSize = im.size
    imShape = im.shape
    im = np.pad(im, padSize, 'symmetric')
    V = np.zeros(imShape, dtype=np.float32)
print("Data loaded.")


class Dataset(data.Dataset):
    def __init__(self, im, l, imShape, WindowSize, corePadSize, isTrain=False, offset=None):
        self.im, self.l = im, l
        self.imShape, self.WindowSize, self.corePadSize = imShape, WindowSize, corePadSize
        self.sampleID, self.isTrain, self.offset = [], isTrain, offset if offset is not None else (0, 0)
        self.__shuffle__()

    def __shuffle__(self):
        self.sampleID = []
        if self.isTrain:
            self.offset = np.random.randint(0, 2 * self.corePadSize, 2)
        for i in range(0, imShape[0]):
            for j in it.chain(range(self.corePadSize+ self.offset[0], self.imShape[1] - self.corePadSize,
                                    2 * self.corePadSize + 1), [self.imShape[1] - self.corePadSize - 1]):
                for k in it.chain(range(self.corePadSize+ self.offset[1], self.imShape[2] - self.corePadSize,
                                        2 * self.corePadSize + 1), [self.imShape[2] - self.corePadSize - 1]):
                    self.sampleID.append(np.ravel_multi_index((i, j, k, 0), self.imShape))
        if self.isTrain:
            shuffle(self.sampleID)

    def __len__(self):
        return len(self.sampleID)

    def __getitem__(self, index):
        """Generates one sample of data"""
        ID = self.sampleID[index]
        r = np.unravel_index(ID, self.imShape)
        im_ = self.im[r[0]:(r[0] + self.WindowSize[0]), r[1]:(r[1] + self.WindowSize[1]),
                      r[2]:(r[2] + self.WindowSize[2]), :].transpose((-1, 0, 1, 2)).copy()
        if self.isTrain:
            # print(im_, 'before clip')
            im_ = np.clip(0.1 * (np.random.rand() - 0.5) +
                          im_ * (1 + 0.20 * (np.random.rand() - 0.5)), -.5, .5)
            # print(im_, 'after clip')
        if self.l is not None:
            l_ = self.l[r[0], (r[1] - self.corePadSize):(r[1] + self.corePadSize + 1),
                        (r[2] - self.corePadSize):(r[2] + self.corePadSize + 1), 0].flatten().astype('int64')
            print(l_, 'l_')
        im_=im_.astype(float)
        return im_, l_ if self.l is not None else ID


class DeepVess(nn.Module):
    def __init__(self):
        super(DeepVess, self).__init__()

        self.activ = nn.LeakyReLU()

        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 32, 3),
            self.activ ,
            nn.Conv3d(32, 32, 3),
            self.activ ,
            nn.Conv3d(32, 32, 3),
            self.activ ,
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, (3, 3)),
            self.activ,
            nn.Conv2d(64, 64, (3, 3)),
            self.activ,
            nn.MaxPool2d((2, 2), stride=(2, 2))
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 1 * 21 * 21, 1024),
            self.activ ,
            nn.Dropout(),
            nn.Linear(1024, 2 * 1 * 21 * 21)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = x.reshape(x.shape[:2] + x.shape[3:])
        x = self.conv2(x)
        x = x.reshape(x.shape[0], np.prod(x.shape[1:]))
        x = self.fc(x)
        x = x.reshape(x.shape[0], 2, 21* 21)
        return x


class TopLoss(nn.Module):
    def __init__(self, size):
        super(TopLoss, self).__init__()
        self.pdfn = LevelSetLayer2D(size=size,  sublevel=False)
        self.topfn = PartialSumBarcodeLengths(dim=1, skip=0)  # penalize more than 0 hole
        self.topfn2 = SumBarcodeLengths(dim=0)  # penalize more than 1 max

    def forward(self, beta):
        eps = 1e-7
        beta = torch.clamp(F.softmax(beta, dim=1), eps, 1 - eps)
        beta = beta[0:beta.shape[0] // 10, ...]
        loss = 0
        for i in range(beta.shape[0]):
            dgminfo = self.pdfn(beta[i, 1, :])
            loss += self.topfn(dgminfo) + self.topfn2(dgminfo)
        # print(np.unique(beta.detach().cpu().numpy()), beta.shape)
        if beta.shape[0]==0:
            print('beta.shape[0] is zero:',beta.shape[0], 'beta :', beta,'F.softmax: ' ,F.softmax(beta, dim=1))
            return loss
        else:
            return loss / beta.shape[0]


def dice_loss(y, l):
    """loss function based on mulitclass Dice index"""
    eps = 1e-7
    l = l.type(torch.cuda.FloatTensor)
    y = torch.clamp(F.softmax(y, dim=1), eps, 1 - eps)[:, 1, :]
    yl, yy, ll = y * l, y * y, l * l
    return 1 - (2 *yl.sum() + eps) / (ll.sum() + yy.sum() + eps)


model = DeepVess()
model = nn.DataParallel(model)
model = model.cuda()
CE = torch.nn.CrossEntropyLoss().cuda()
tloss = TopLoss((2 * corePadSize + 1,) * 2).cuda()
Loss = lambda y_, l_: dice_loss(y_, l_).cuda() + CE(y_, l_) + tloss(y_) / 100
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

if isForward:
    checkpoint = torch.load("model_DV-epoch920.pt")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    model.eval()
    print("model_DV-epoch920.pt restored.")
    I = [0, int( 2 * corePadSize / 3 + 1), int(4 * corePadSize / 3 + 1)]
    for offset in [(ii, ij) for ii in I for ij in I]:
        Forward_data = data.DataLoader(Dataset(im, None, imShape, WindowSize, corePadSize, isTrain=False,
                                               offset=offset), batch_size=batch_size)
        numBatch = Forward_data.dataset.__len__() // batch_size + 1
        for i, d in tqdm(enumerate(Forward_data), 'Forward', numBatch):
            x1, vID = d[0].cuda(), d[1]
            x1 = x1.type(torch.FloatTensor)
            y1 = np.reshape(np.argmax(model(x1).detach().cpu().numpy(), 1),
                            (-1, (2 * corePadSize + 1), (2 * corePadSize + 1)))
            for j in range(len(vID)):
                r = np.unravel_index(vID[j], imShape)
                V[r[0], (r[1] - corePadSize):(r[1] + corePadSize + 1),
                    (r[2] - corePadSize):(r[2] + corePadSize + 1), 0] += y1[j, ...]
    V = (V > (len(I) ** 2 // 2))
    toc = time.time()
    print("The time needed to run inference is: ", toc - tic, "seconds")
    np.save(path_save_output + 'output.npy', V)

