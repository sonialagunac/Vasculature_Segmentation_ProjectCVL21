# Unsupervised classical methods

Unsupervised classical methods used in the project "Vasculature extraction from two-photon microscopy", based on the scikit-image implementations. Including: 
- Otsu thresholding
- Li thresholding
- ISODATA thresholding
- Active Contours Without Edges (ACWE)

Additionally includes the computation of metrics: Dice score, Jaccard Index, Intersection over Union, precission, recall, sensitivity, specificity, accuracy.
##Training:
Training corresponds to the computation of thresholds. Data location is detailed in 'path_img_train' and can correspond to the test split of one of the databases. 'path_label_train' can be included as ground truth for metrics computation. 

##Testing\Inference:
Testing is generally carried out in the same files as training where the thresholds were computed. However, to run cross-inference, the crops of the opposite dataset are required. 
'path_img_test' includes the desired file location. 'path_lbl_test' includes the file location of the ground truth. 

Note: ACWE cannot be run with cross-inference as there is no threshold or parameters to be learnt. 

## Prerequisites
- Python 3.6
- tifffile
- numpy
- time
- scikit-image



# Citation
```
@article{otsu1979threshold,
  title={A threshold selection method from gray-level histograms},
  author={Otsu, Nobuyuki},
  journal={IEEE transactions on systems, man, and cybernetics},
  volume={9},
  number={1},
  pages={62--66},
  year={1979},
  publisher={IEEE}
}

@article{li1993minimum,
  title={Minimum cross entropy thresholding},
  author={Li, Chun Hung and Lee, CK},
  journal={Pattern recognition},
  volume={26},
  number={4},
  pages={617--625},
  year={1993},
  publisher={Elsevier}
}

@article{sezgin2004survey,
  title={Survey over image thresholding techniques and quantitative performance evaluation},
  author={Sezgin, Mehmet and Sankur, B{\"u}lent},
  journal={Journal of Electronic imaging},
  volume={13},
  number={1},
  pages={146--165},
  year={2004},
  publisher={International Society for Optics and Photonics}
}

@article{getreuer2012chan,
  title={Chan-vese segmentation},
  author={Getreuer, Pascal},
  journal={Image Processing On Line},
  volume={2},
  pages={214--224},
  year={2012}
}
```