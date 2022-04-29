import argparse
import tifffile as T
import numpy as np
from skimage.morphology import skeletonize_3d
from skimage.morphology import ball, binary_dilation
from scipy.ndimage import convolve, median_filter
from utils_cpu import imfill
from skimage.measure import label
from scipy.signal import medfilt
from skimage import io
from skimage.morphology import ball, binary_closing, remove_small_objects

#Segmentation pipeline, including skeleton extraction from prediction file

#Loading arguments
parser = argparse.ArgumentParser(description='Segmentation pipeline')
parser.add_argument('--thr', type=str, default= 0.9, metavar='N',
                   help='Threshold corresponding to each model: UMIS = 0.9, DV and TopoDV = 0.4, Unets = 0.5')
parser.add_argument('--orig-seg-path', type=str, default=r'C:\Users\sonil\PycharmProjects\Semester project\Data_output\inhousedata\umis\output_UMIS_inhouse_Epoch1k.tif', metavar='N',
                   help='Path with prediction segmentations to compute skeleton')
parser.add_argument('--skeleton-path', type=str, default='/itet-stor/slaguna/home/Documents/semproject/pipeline', metavar='N',
                    help='Path to store skeletons')
args = parser.parse_args()

#Dataloader from prediction files for skeleton extraction

#Loading output from models
segm = T.imread(args.orig_seg_path) > args.thr
segm=segm.astype('bool')
#topology postprocessing step
segm = remove_small_objects(segm, min_size=200, connectivity=1)
skel_holes=np.zeros(segm.shape)
boxFiltW = 3
deadEndLimit = 20

#Center line extraction
print('Starting skeleton extraction')

# 1. Meadian filter previous to skel
segm_filt = medfilt(segm, ([boxFiltW, boxFiltW, boxFiltW]))

# 2. Thinning: Skeleton based on the algorithm developed by Lee et al. (T.-C. Lee, R.L. Kashyap and C.-N. Chu, Building skeleton models via 3-D medial surface/axis thinning algorithms. Computer Vision, Graphics, and Image Processing, 56(6):462-478, 1994.)
skel_init = skeletonize_3d(segm_filt)

# 3. Dilation with spherical kernel
skel_dil = binary_dilation(skel_init, ball(5))

# 4. Median filtering (3)
skel_filt = medfilt(skel_dil, ([boxFiltW, boxFiltW, boxFiltW]))
skel_filt = skel_filt.astype('bool')

# 5. Removing holes
for i in range(skel_filt.shape[0]):
    skel_holes[i, :] = imfill(skel_filt[i, :])
skel_holes = skel_holes / 255

# 6. Final thinning
skel = skeletonize_3d(skel_holes)

# 7. Remove undesired branches
change = 1

while change >= 1:
    # Removing short dead end vessels: vessels with one end not connected to the network (i.e., dead end) and with length smaller than 11 voxels.
    new = skel
    c = convolve(skel, np.ones((boxFiltW, boxFiltW, boxFiltW)))
    c2 = np.multiply(convolve(np.multiply(skel, (c == 2)), np.ones((boxFiltW, boxFiltW, boxFiltW))), skel)
    cc1 = np.multiply(skel, (c == 3))
    cc0 = label(cc1, return_num=True)
    for i in range(1, cc0[1] + 1):
        if np.logical_and(np.sum(cc0[0] == i) < deadEndLimit, np.any(c2[cc0[0] == i])):
            skel[cc0[0] == i] = 0

    # Remove single pixel connected to node, Remove single voxels connected to a junction
    c = convolve(skel, np.ones((3, 3, 3)))
    c2 = convolve(c > 3, np.ones((3, 3, 3)))
    cc0 = np.multiply((skel > 0), (c == 2), c2)
    skel[cc0 > 0] = 0

    # Remove isolated pixel, single voxels with no connections.
    C = convolve(skel, np.ones((3, 3, 3)))
    skel[np.logical_and(skel, C == 1)] = 0

    # Remove vessel loops with length of one or two voxels.
    # Remove single pixel
    C = convolve(skel, np.ones((3, 3, 3)))
    C2 = np.multiply(convolve(np.multiply(skel, (C > 3)), np.ones((3, 3, 3))), skel)
    CC1 = np.multiply(skel, np.multiply((C == 3), (C2 == 2)))
    for k in np.transpose(np.nonzero(CC1.flatten())):
        CC0 = label(skel)
        interm = skel.flatten()
        interm[k] = 0
        skel = np.reshape(interm, skel.shape)
        CC1 = label(skel)
        if np.not_equal(np.amax(CC0), np.amax(CC1)):
            interm[k] = 1
            skel = np.reshape(interm, skel.shape)

    # Remove double pixle loop
    C = convolve(skel, np.ones((3, 3, 3)))
    C2 = np.multiply(convolve(np.multiply(skel, (C == 2)), np.ones((3, 3, 3))), skel)
    CC1 = np.multiply(skel, (C == 3))
    CC0 = label(CC1)
    for i in range(np.amax(CC0)):
        if np.sum(CC0 == i) < 3:
            skel[CC0 == i] = 0
    check = np.not_equal(new, skel)
    change = np.sum(check)

io.imsave(args.skeleton_path + 'UMISskeleton.tif', skel, plugin='tifffile')
