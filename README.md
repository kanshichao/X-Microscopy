# X-Microscopy
Deep learning enables STORM-like superresolution image reconstruction from conventional microscopy
## Overview
This project is to produce the results of our paper: [Deep learning enables STORM-like superresolution image reconstruction from conventional microscopy](). We provide two training methods with fixed input size and flexible input size, as well as flexible input size test. The following is the instructions for using it.

# System Requirements

## Hardware requirements
This code requires a standard computer with enough RAM to support the in-memory operations and the GeForce GTX 1080 GPU (The NVIDIA Inc.) to support GPU computing.


## Software requirements
### OS Requirements
This package is supported for Linux. The package has been tested on the following systems:
+ Linux: Ubuntu 16.04

### Python Dependencies
This package mainly depends on the Python-3.6.4 scientific stack.
```
numpy
scipy
scikit-learn
pandas
tensorflow-1.13.1
```
### GPU Dependencies
This package mainly depends on CUDA 10.0 and cudnn 7.

# Installation Guide:

### Install from Github
```
git clone https://github.com/kanshichao/X-Microscopy
cd X-Microscopy
```

## Setting up the development environment
* The folder of UR-Net-8 contains the MU-SRM reconstruction code, and change to this folder to perform wf->MU-SRM.
* The folder of X-Net contains the F-SRM reconstruction code, and change to this folder to perform wf+MU-SRM->F-SRM.


## Instructions
When you use this code, please change the corresponding folder in the code to yours, i.e., the data path in the train, evaluate, and test functions of the train.py, and in the main function of the main.py. 

The parameter of --phase is to alternative the state of training or test, set as train for training and set as test for test.
+ For training:
```bash
python main.py --phase train
```
+ For test: 
```bash
python main.py --phase test
```
The parameter of --same_input_size is to alternative fixed input size or flexible input size during the training stage. If you want to run the code with fixed input size during the training stage, you shold set the value of --same_input_size as True, otherwise, set the value of --same_input_size as False.
+ For training or fine-tuning with fixed input size: 
```bash
python main.py --phase train --same_input_size True
```
+ For training or fine-tuning with flexible input size: 
```bash
python main.py --phase train --same_input_size False
```
The script of evaluate.py is used to evaluate the performances of SRM reconstruction, which is based on the realized verison of python. When you use it, please change the corresponding folder to yours.
## Pretrained models
We provide the trained models to reproduce the results that presented in our paper. 

+ [Google Drive](https://drive.google.com/drive/folders/1tVtzCAihlcHfcBRTTPW4S3fWuTT1xVEU?usp=drive_link)

For detailed technical details, please see our paper and the released code.

### Citation

If you use this method or this code in your research, please cite as:

    @inproceedings{XuleiKanshichao-2022,
    title={Deep learning enables STORM-like superresolution image reconstruction from conventional microscopy},
    author={Lei Xu, Shichao Kan, Xiying Yu, Yuxia Fu, Yiqiang Peng, Yanhui Liang, Yigang Cen, Changjun Zhu, Wei Jiang},
    booktitle={},
    pages={},
    year={2023}
    }

### Acknowledgments
This code is written based on the tensorflow framework of pix2pix. 

### License
This code is released for academic research / non-commercial use only. This project is covered under the MIT License.
