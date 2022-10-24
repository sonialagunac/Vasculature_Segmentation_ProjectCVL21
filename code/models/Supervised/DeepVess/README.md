# DeepVess

Implementation of the DeepVess architecture by the authors of "Unsupervised Microvascular Image Segmentation Using an Active Contours Mimicking Neural Network" ([link](https://arxiv.org/abs/1908.01373)). Adapted by Sonia Laguna.

## Training:
- training_DV.py, training_DV_cpu.py: Scripts used to train the network using GPU and CPU/GPU available. Paths to files required: 
    - Training dataset name: Desired name to store the checkpoint
    - Training dataset images path: Path to folder containing input images    
    - Training dataset labels path: Path to folder containing input labels
    - Saving loss: Path to desired loss location    
    - Set the checkpoint name: Name of the desired saved checkpoint. Will be stored in the location 'run\Training dataset name\Checkpoint name'
  
## Testing\Inference:
- testing_DV.py, testing_DV_cpu.py: Scripts used to test the network using GPU and CPU/GPU available. Paths to files required:
    - Validation image path: Path to test image file
    - Validation label path: Path to test labels file, if not available duplicate 'Validation image path'
    - Saving output prediction path: Path to desired output prediction location
    - Put the path to resuming file if needed: Path to previously stored checkpoint.
- Also outputs the amount fo time required for inference.

## Prerequisites
- Python 3.6
- Pytorch +1.4
- Numpy
- Scipy
- OpenCV
- Path
- tqdm
- h5py
- tifffile
- libtiff

# Citation
```
@inproceedings{gur2019unsupervised,
  title={Unsupervised Microvascular Image Segmentation Using an Active Contours Mimicking Neural Network},
  author={Gur, Shir and Wolf, Lior and Golgher, Lior and Blinder, Pablo},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={10722--10731},
  year={2019}
}
```
