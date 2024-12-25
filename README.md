# visual-audio cross modal retrieval task

Code for Paper: Deep Supervised with Fine-grained Feature Fusion Network for Cross-modal Retrieval. Our code runs in the Windows 11 environment.
# Installation
## 1. Clone the repository
```bash
$ git clone https://github.com/zhangjiwei-japan/icaiic-2024-code.git
```
## 2. Requirements
#### （1） Install python from website ：https://www.python.org/downloads/windows/
#### （2） Our program runs on the GPU. Please install the cuda, cudnn, etc as follows : 
- CUDA Toolkit Archive ：https://developer.nvidia.com/cuda-toolkit-archive
- cuDNN Download | NVIDIA Developer ：https://developer.nvidia.com/login
- PYTORCH : https://pytorch.org/
#### （3） Libraries required to install the program ：
```bash
pip install -r requirements.txt
```
## 3. Prepare the datasets
### (1) AVE Dataset 
- AVE dataset can be downloaded from https://drive.google.com/open?id=1FjKwe79e0u96vdjIVwfRQ1V6SoDHe7kK.
- Scripts for generating audio and visual features: https://drive.google.com/file/d/1TJL3cIpZsPHGVAdMgyr43u_vlsxcghKY/view?usp=sharing
#### You can also download our prepared [AVE](https://drive.google.com/file/d/14Qdprd8_9cdih3QDN726kJTzaoo9Y8Y-/view?usp=sharing) dataset.
- Please create a folder named 'ave' to place the downloaded dataset and set the dataset base path in the code (train_model.py and test_model.py): `base_dir` = "./datasets/ave/"
- Place the downloaded dataset in the 'ave' file and load a dataset path in the code (train_model.py and test_model.py): `load_path` = `base_dir` + "your downloaded dataset"
### (2) VEGAS Dataset 
- The Raw dataset from: https://arxiv.org/abs/1712.01393.
#### You can also download our prepared [VEGAS](https://drive.google.com/file/d/142VXU9-3P2HcaCWCQVlezRGJguGnHeHD/view?usp=sharing) dataset. 
- Please create a folder named 'veags' to place the downloaded dataset and set the dataset base path in the code (train_model.py and test_model.py): `base_dir` = "./datasets/vegas/"
- Place the downloaded dataset in the 'veags' file and load a dataset path in the code (train_model.py and test_model.py): `load_path` = `base_dir` + "your downloaded dataset"
## 4. Execute train_model.py to train and evaluate the model as follows :
```bash
python train_model.py
```
#### Only the following parameters need to be modified when running train_model.py : 
- `batch_size`: train batch size.
- `dataset`: dataset name "vegas or ave".
- `num_epochs`: set training epoch.
- `class_dim`: vegas dataset class_dim = 10, ave dataset class_dim = 15. 
# Example
## 1. Training :
The model in the paper can be tested as follows :
```bash
python trian_model.py
```
- `dataset`: dataset name "vegas".
- `num_epochs`: set training epoch 40.
- `class_dim`: vegas dataset class_dim = 10.
## 1. Testing :
The model in the paper can be tested as follows :
```bash
python test_model.py
```
- `dataset`: dataset name "vegas".
#### Only the following parameters need to be modified when running test_model.py :
- `save_path`: load trained model path.
- `dataset`: dataset name "vegas or ave".
- `test_size`: batch size of the test set.
- `class_dim`: vegas dataset class_dim = 10, ave dataset class_dim = 15. 
## 2. Evelation : 
we use mAP as metrics to evaluate our architecture, when the system generates a ranked list in one modality for a query in another modality. Those documents in the ranked list with the same class are regarded as relevant or correct.
|Datasets    | Audio2Visual| Visual2Audio  | mAP |
| --------   | -----    | -----  |  -----  |
|#AVE      | ~0.371  | ~0.368 | ~0.370| 
|#VEGAS  | ~0.881 | ~0.8895  | ~0.888 | 
## Contact
If you have any questions, please email s210068@wakayama-u.ac.jp
