Instructions
Clone the code to your system with GeForce GTX 1080 GPU.
The folder of X-Net contains the perfect SRM reconstruction code.
The training and test processes are based on python-2.7, TensorFlow1.13.1, CUDA10.0, cudnn7.
Set the dataset name and checkpoint_dir to train and test the model.
The parameter of --phase is used to alternative the state of training or test, set as train for training and set as test for test.
For training:
python main.py --phase train
For test:
python main.py --phase test
The parameter of --same_input_size is to alternative fixed input size or flexible input size during the training stage. If you want to run the code with fixed input size during the training stage, you shold set the value of --same_input_size as True, otherwise, set the value of --same_input_size as False.
For training or fine-tuning with fixed input size:
python main.py --phase train --same_input_size True
For training or fine-tuning with flexible input size:
python main.py --phase train --same_input_size False
The script of evaluate_test.py and evaluate.py is used to evaluate the performances of SRM reconstruction, which is based on the realized verison of python. When you use it, please change the corresponding folder to yours.