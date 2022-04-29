from torch.utils.data import DataLoader
import argparse
from data.datasets import Single_Image_Eval
import numpy as np
from skimage.morphology import skeletonize_3d
from skimage.morphology import ball, binary_dilation
from scipy.ndimage import convolve, median_filter
from utils import imfill
from utils import forward
from skimage.measure import label
from scipy.signal import medfilt
from skimage import io
from skimage.morphology import ball, binary_closing, remove_small_objects

#Segmentation skeletonization pipeline, including forward pass of NN and segmentation skeleton extraction

#Loading arguments
parser = argparse.ArgumentParser(description='Segmentation pipeline')
parser.add_argument('--orig-images-path', type=str, default=r'C:\Users\sonil\PycharmProjects\Semester project\DeepVess-master\202007_inhouse_data\Jacq_data\Cropped_volume.tif', metavar='N',
                   help='Original microscopy image path')
parser.add_argument('--orig-labels-path', type=str, default=r'C:\Users\sonil\PycharmProjects\Semester project\DeepVess-master\202007_inhouse_data\Jacq_data\segmentation.tif', metavar='N',
                   help='Original labels path')
parser.add_argument('--range-norm', action='store_true',
                    help='range-norm')
parser.add_argument('--validate', action='store_false',
                    help='validate')
parser.add_argument('--lr', type=float, default=1e-4, metavar='N',
                    help='Learning rate')
parser.add_argument('--resume', type=str, default=r'/itet-stor/slaguna/net_scratch/semproject/UMIS/run/InHouseSplits/UMIS_Epoch1k_inhouse/experiment_3/checkpoint.pth.tar',
                    help='path to load model from')
parser.add_argument('--checkname', type=str, default='UMIS_skeleton_inhouse',
                    help='set the checkpoint name')
parser.add_argument('--skeleton-path', type=str, default='C:\Users\sonil\PycharmProjects\Semester project\Data_output\skeleton', metavar='N',
                    help='Path to store skeletons')
args = parser.parse_args()

# Dataloader for NN forward pass
dataset_val = Single_Image_Eval(image_path=args.orig_images_path,
                                label_path=args.orig_labels_path,
                                data_shape=(32, 128, 128),
                                lables_shape=(32, 128, 128),
                                stride=(8, 16, 16),
                                range_norm=args.range_norm)
dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=2)

#NN forward pass

output_np, input_np =forward(args,dataloader_val,dataset_val)
output_np=np.squeeze(output_np)
input_np=np.squeeze(input_np)

#Saving NN original predictions
np.save(args.skeleton_path + 'segm_output.npy', output_np)
np.save(args.skeleton_path + 'segm_input.npy', input_np)


#Preparing skeleton extraction
segm = output_np > 0.9
segm = segm.astype('bool')
#topology postprocessing step
segm = remove_small_objects(segm, min_size=200, connectivity=1)
skel_holes=np.zeros(segm.shape)
boxFiltW = 3
deadEndLimit = 20

#Center line extraction
print('Starting skeleton extraction')

#1. Meadian filter previous to skel
segm_filt=medfilt(segm,([boxFiltW,boxFiltW,boxFiltW]))

#2. Thinning: Skeleton based on the algorithm developed by Lee et al. (T.-C. Lee, R.L. Kashyap and C.-N. Chu, Building skeleton models via 3-D medial surface/axis thinning algorithms. Computer Vision, Graphics, and Image Processing, 56(6):462-478, 1994.)
skel_init=skeletonize_3d(segm_filt)

#3. Dilation with spherical kernel
skel_dil=binary_dilation(skel_init, ball(5))

#4. Median filtering (3)
skel_filt=medfilt(skel_dil,([boxFiltW,boxFiltW,boxFiltW]))
skel_filt=skel_filt.astype('bool')

#5. Removing holes
for i in range(skel_filt.shape[0]):
    skel_holes[i,:]=imfill(skel_filt[i,:])
skel_holes=skel_holes/255

#6. Final thinning
skel=skeletonize_3d(skel_holes)

#7. Remove undesired branches
change=1

while change>=1:
    #Removing short dead end vessels: vessels with one end not connected to the network (i.e., dead end) and with length smaller than 11 voxels.
    new=skel
    c=convolve(skel, np.ones((boxFiltW,boxFiltW,boxFiltW)))
    c2=np.multiply(convolve(np.multiply(skel,(c==2)),np.ones((boxFiltW,boxFiltW,boxFiltW))),skel)
    cc1=np.multiply(skel,(c==3))
    cc0=label(cc1,return_num=True)
    for i in range(1,cc0[1]+1):
        if np.logical_and(np.sum(cc0[0]==i) < deadEndLimit, np.any(c2[cc0[0]==i])):
            skel[cc0[0]==i] = 0

    #Remove single pixel connected to node, Remove single voxels connected to a junction
    c=convolve(skel,np.ones((3,3,3)))
    c2=convolve(c>3,np.ones((3,3,3)))
    cc0=np.multiply((skel>0),(c==2),c2)
    skel[cc0>0]=0

    #Remove isolated pixel, single voxels with no connections.
    C = convolve(skel, np.ones((3, 3, 3)))
    skel[np.logical_and(skel, C==1)] = 0

    #Remove vessel loops with length of one or two voxels.
    #Remove single pixel
    C = convolve(skel, np.ones((3, 3, 3)))
    C2 = np.multiply(convolve(np.multiply(skel, (C>3)), np.ones((3, 3, 3))), skel)
    CC1 = np.multiply(skel, np.multiply((C==3) ,(C2==2)))
    for k in np.transpose(np.nonzero(CC1.flatten())):
        CC0 = label(skel)
        interm=skel.flatten()
        interm[k]=0
        skel=np.reshape(interm,skel.shape)
        CC1 = label(skel)
        if np.not_equal(np.amax(CC0),np.amax(CC1)):
            interm[k]=1
            skel=np.reshape(interm,skel.shape)

    #Remove double pixle loop
    C = convolve(skel, np.ones((3, 3, 3)))
    C2 =np.multiply(convolve(np.multiply(skel, (C == 2)), np.ones((3, 3, 3))), skel)
    CC1 =np.multiply(skel, (C == 3))
    CC0 = label(CC1)
    for i in range(np.amax(CC0)):
        if np.sum(CC0==i) < 3:
            skel[CC0==i] = 0
    check=np.not_equal(new,skel)
    change=np.sum(check)

io.imsave(args.skeleton_path + 'UMISskeleton.tif', skel, plugin='tifffile')
