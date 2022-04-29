# DeepVess
[***DeepVess with topology loss***](http://openaccess.thecvf.com/content_CVPRW_2020/html/w57/Haft-Javaherian_A_Topological_Encoding_Convolutional_Neural_Network_for_Segmentation_of_3D_CVPRW_2020_paper.html) is a model based on ***DeepVess*** in addition to a topological loss term. 
Download the trained model file from this Google Drive URL: [model-epoch2000.pt](https://drive.google.com/file/d/1bRTBTkcCfSdEK4GhXbd7fgZ2Qa7uo3dL/view?usp=sharing). Adapted by Sonia Laguna.

## Training
- First, set up the topology layer with instructions in the folder 'TopologyLayer-master'
- After setting up the topology layer, Run script training_TopoDV

## Testing

- Run script testing_TopoDV

Note: The original implementation loads h5 files coming from Matlab preprocessing. In this variant we do not carry out preprocessing and .tif files are used as inputs.
Path_img takes the whole image stack previous to data spliting and path_lbl the whole label stack. Path_img_test is the test split of the aforementioned data. 
## Requirements
* [Python 3](https://www.python.org) 
* [PyTorch](https://pytorch.org/) 
* [TopologyLayer](https://github.com/bruel-gabrielsson/TopologyLayer)

## Publication
* Haft-Javaherian, M., Villiger, M., Schaffer, C. B., Nishimura, N., Golland, P., & Bouma, B. E. (2020). A Topological Encoding Convolutional Neural Network for Segmentation of 3D Multiphoton Images of Brain Vasculature Using Persistent Homology. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (pp. 990-991). [Open Access link](http://openaccess.thecvf.com/content_CVPRW_2020/html/w57/Haft-Javaherian_A_Topological_Encoding_Convolutional_Neural_Network_for_Segmentation_of_3D_CVPRW_2020_paper.html).
## Contact
* Mohammad Haft-Javaherian <mh973@cornell.edu>, <haft@csial.mit.edu>

## License
[Apache License 2.0](../LICENSE)
