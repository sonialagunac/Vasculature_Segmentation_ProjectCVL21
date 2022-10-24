This folder includes all the data, code and outputs used in the project "Vasculature Extraction from two-photon microscopy".

Presented on  July 2nd, 2021.

Work carried out by Sonia Laguna with the supervision of Gustav Bredell in the Biomedical Image Computing Group, Computer Vision Laboratory at ETH Zurich.

# Distribution of the files

## Input data folder

Includes the slices used for training and test of the DeepVess data and the InHouse data.

- inhouse_slice and deepvess_slice: folders with the test and train crops of the label and input image.
- deepvess_graph: train and test data used for training the models to predict the vessel skeleton.
- inhouse_slice_h5, deepvess_slice_h5, deepvess_graph_h5: Corresponding h5 files in the format needed for the 3DUnet and Res3DUnet models.
- additional_inhouse_data: Includes resampled files from the inhouse dataset that do not include ground truth annotations. Used to test the trained models. 

Note: The original non-cropped files can also be find in the general input_data folder.
## Code folder

- Models: Includes all the supervised and unsupervised deep learning and classical models carried out in this project. All architectures include their own README file with details on the code execution. Some READMEs have been adapted from the original files in the official paper implementations.
- Skeleton_pipeline: Includes all the steps in the skeletonization pipeline. There are two versions: one from the original microscopy image, runs the forward pass of a trained NN model and carries out the skeletonization pipeline and another one starting directly from the predicted segmentation from a NN and carries out the skeletonization pipeline. A README file is included for clarification and usage purposes.
- Topology_postprocessing: Code and README file with the topology postprocessing step carried out on the unsupervised models predictions.
- JupyterNotebooks: Includes two Notebooks with examples on how the data was splitted, stored, normalized and evaluated in the project. 


## Output data folder
  Contains all the output segmentations organized by dataset. The distributed folders are:

- DeepVessDataSet: Includes the segmentation results from every model studied in the project together with the postprocessed supervised versions. Every file is named after the dataset and model used.
- InHouseDataSet: Includes the segmentation results from every model studied in the project together with the postprocessed supervised versions. Every file is named after the dataset and model used. It also includes an 'Additional unlabeled data' folder with the results of running inference on the additional InHouse data samples with models trained with the original InHouse dataset.
- Skeleton: Results of the skeleton on InHouse data and DeepVess data using different models properly named. 
- TrainDVTestIH: Includes the segmentation results from every learning model studied in the project after training on the DeepVess dataset and running inference on the InHouse dataset.
- TrainIHTestDV: Includes the segmentation results from every learning model studied in the project after training on the InHouse dataset and running inference on the DeepVess dataset.

