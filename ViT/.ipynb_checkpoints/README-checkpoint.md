# Pytorch Project for Plant Pathology with Advanced ViT Model 

本文件仅描述了文件架构以及训练和测试的流程，创新点及其他在报告的pdf文件中。请阅读本说明文档后进行测试或训练。

## Abstract

为实现对植物病理学数据集进行训练，从而完成对植物是否得病以及对得病种类进行分类。我搭建了网络对植物的特征进行抽取和训练。由于经费和资源有限，对于类似Vision Transformer的大模型，使用单卡的ascend910难以训练，而多卡又过于昂贵。所以我使用pytorch架构，搭建了经过改进的ViT模型，使用四张Nvidia 2080Ti 进行训练，完成了500个Epoch的训练，得到了效果优异的模型。本项目提供了测试接口和训练接口。根据相关指令，在修改数据集相关路径后便可直接运行，最终结果输出测试集的准确率和误差。

## Run
1. 安装运行ViT所需的python package:  
`pip install -r requirements.txt`  
torch和对应torch_vison请根据cuda版本进行安装.下方提供了cuda 11.1对应的pytorch＝1.8.1版本和torchvision=0.9.1版本的pip install方式：  
`pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html`
2. 训练流程

  请将./datasets/read_data.py文件中的data_dir修改为数据集现在的路径。

  完整数据集训练 `bash run.sh`

  由于训练数据集过于庞大，照片过大导致加载数据会消耗大量显存，从而需要用四张2080Ti方可完成训练数据加载。为方便使用jupyter notebook进行展示，本项目提供train_small_batch接口，且已经使用单卡完成小批量数据集的加载和网络的运行。请打开train.ipynb文件查看。

测试流程
`python test.py`

![ViT](pics/ViT.png)
## demo file structure
```base
.
|-- README.md
|-- requirements.txt
|-- configs
|-- pics
|   |-- train.png
|   |-- ViT.png
|-- datasets
|   `-- data_utils
|       |-- data_transform             # 数据预处理
|   |-- dataset.py									 # 定义dataset类
|   |-- read_data.py				# 从csv中读取数据
|-- models
|   |-- SoftTargetCrossEntropy.py       # 定义损失函数
|   |-- vision_transformer.py         # 定义ViT模型
|-- results
|   |-- best_loss.pth        # 最优的模型文件
|-- run.py   								# 训练文件
|-- train.py               # 训练相关设置
|-- run.sh                # 多卡训练命令行，运行即可训练
|-- test.py   					     	# 测试
```

## Tips
1. 提供GPU和CPU训练的接口，自动检测CUDA是否可用，可用则使用GPU进行训练。
3. 训练过程产生的训练模型为在测试集中表现最优的模型，保存在result文件下，测试时可通过更改训练模型来检验效果。

## 效果

如下图所示为测试得到的示意图。

![ViT](pics/train.png)
