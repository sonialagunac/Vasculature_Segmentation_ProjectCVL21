# Skeleton extraction pipeline

Skeleton extraction pipeline used in the project "Vasculature extraction from two-photon microscopy", based on the scikit-image implementations. Including: 

Based on the algorithm developed by Lee et al. (T.-C. Lee, R.L. Kashyap and C.-N. Chu, Building skeleton models via 3-D medial surface/axis thinning algorithms. Computer Vision, Graphics, and Image Processing, 56(6):462-478, 1994.), together with additional morphology operations and convolutional filters. 
All the steps are labeled and explained. 

Two main files for skeleton extraction: 

## skeleton_pipeline_fromNN.py

Includes all the steps from the original microscopy image, runs the forward pass of a trained NN model and carries out the skeletonization pipeline. 
Requires a GPU as the default example is running the unsupervised UMIS.
Inputs: 

- Original microscopy image path: Microscopy image
- Original labels path: Label path for visualization purposes. If not available, repeat 'Original microscopy image path'. 
- Path to load model from: Checkpoint from trained model
- Path to store skeleton

### Prerequisites
- Python 3.6
- CUDA 10.1
- Pytorch +1.4 (i.e. 1.6), torchvision
- Numpy
- Scipy
- OpenCV
- Path
- Tqdm
- Tifffile
- Libtiff 
- Scikit-image
- Cv2

## skeleton_pipeline_frompred.py
Includes all the steps starting directly from the predicted segmentation from a NN and carries out the skeletonization pipeline. 
Can run on GPU or CPU as there is no NN invovled. 
Inputs: 

- Threshold corresponding to each model for binarization: UMIS = 0.9, DV and TopoDV = 0.4, Unets = 0.5
- Path with prediction segmentations to compute skeleton: With the output file of a previous NN model
- Path to store skeletons

### Prerequisites
- Python 3.6
- Numpy
- Scipy
- OpenCV
- Path
- Tifffile
- Scikit-image
- Scipy
- Cv2
