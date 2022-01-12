# X-Microscopy
X-Microscopy: Multicolor Super Resolution Image Reconstruction from Conventional Microscopy with Deep Learning
## Introduction
This project is to produce the results of our paper: [X-Microscopy: Multicolor Super Resolution Image Reconstruction from Conventional Microscopy with Deep Learning](). We provide two training methods with fixed input size and flexible input size, as well as flexible input size test. The following is the instructions for using it.

## Instructions
1. The training and test processes are based on python-2.7, TensorFlow1.13.1, CUDA10.0, cudnn7.
2. Clone the code to your system.
* The folder of UR-Net-8 is the proposed sparse SRM reconstruction model.
* The folder of X-Net is the proposed perfect SRM reconstruction model.
3. Copy your training samples into the folder of ./datasets/train/ and corresponding ground truth samples into the folder of ./datasets/train_gt/.
4. Copy your validation samples into the folder of ./datasets/val/ and corresponding ground truth samples into the folder of ./datasets/val_gt/.
5. The parameter of --phase is to alternative the state of training or test, set as train for training and set as test for test.
* For training:
```bash
python main.py --phase train
```
* For test: 
```bash
python main.py --phase test
```
6. The parameter of --same_input_size is to alternative fixed input size or flexible input size during the training stage. If you want to run the code with fixed input size during the training stage, you shold set the value of --same_input_size as True, otherwise, set the value of --same_input_size as False.
* For training or fine-tuning with fixed input size: 
```bash
python main.py --phase train --same_input_size True
```
* For training or fine-tuning with flexible input size: 
```bash
python main.py --phase train --same_input_size False
```
7. If you want to test your trained model, please change the test path in test() of train.py, i.e., the path to obtain sample_files.
8. The script of evaluate.py is used to evaluate the performances of SRM reconstruction, which is based on the realized verison of python. When you use it, please change the corresponding folder to yours.
## Pretrained models
We provide the trained models to reproduce the results that presented in our paper. 

* [UR-Net-8]() Extract Code: 
* [X-Net]() Extract code:  

For detailed technical details, please see our paper and the released code.

### Citation

If you use this method or this code in your research, please cite as:

    @inproceedings{XuleiKanshichao-2022,
    title={X-Microscopy: Multicolor Super Resolution Image Reconstruction from Conventional Microscopy with Deep Learning},
    author={Lei Xu, Shichao Kan, Xiying Yu, Yuxia Fu, Yiqiang Peng, Yanhui Liang, Yigang Cen, Changjun Zhu, Wei Jiang},
    booktitle={},
    pages={},
    year={2022}
    }

### Acknowledgments
This code is written based on the tensorflow framework of pix2pix. 

### License
This code is released for academic research / non-commercial use only. If you wish to use for commercial purposes, please contact [Wei Jiang]() by email wjiang6138@cicams.ac.cn, and [shichao kan](https://faculty.csu.edu.cn/kanshichao/zh_CN/index.htm) by email kanshichao10281078@126.com
