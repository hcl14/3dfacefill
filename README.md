
# *3DFaceFill: An Analysis-by-Synthesis Approach for Face Completion*

## Abstract
Existing face completion solutions are primarily driven by end-to-end models that directly generate 2D completions of 2D masked faces. By having to implicitly account for geometric and photometric variations in facial shape and appearance, such approaches result in unrealistic completions, especially under large variations in pose, shape, illumination and mask sizes. To alleviate these limitations, we introduce 3DFaceFill, an analysis-by-synthesis approach for face completion that explicitly considers the image formation process. It comprises three components, (1) an encoder that disentangles the face into its constituent 3D mesh, 3D pose, illumination and albedo factors, (2) an autoencoder that inpaints the UV representation of facial albedo, and (3) a renderer that resynthesizes the completed face. By operating on the UV representation, 3DFaceFill affords the power of correspondence and allows us to naturally enforce geometrical priors (e.g. facial symmetry) more effectively. Quantitatively, 3DFaceFill improves the state-of-the-art by up to 4dB higher PSNR and 25% better LPIPS for large masks. And, qualitatively, it leads to demonstrably more photorealistic face completions over a range of masks and occlusions while preserving consistency in global and component-wise shape, pose, illumination and eye-gaze.

## Overview
3DFaceFill is an iterative inpainting approach where the masked face is disentangled into its 3D shape, pose, illumination and partial albedo by the 3DMM module, following which the partial albedo, represented in the UV space is inpainted using symmetry prior, and finally the completed image is rendered.
![alt text](images/overview.png "Architecture")

## Contributions
* Face completion using explicit 3D priors leading to geometrically and photometrically better results
* We leverage facial symmetry efficiently for face completion
* Qualitative and quantitative improvement in face completion under diverse conditions of shape, pose, illumination, etc

## Data Preparation
3DFaceFill is trained using the CelebA dataset. We preprocess the CelebA dataset to align and crop the face images. To do this, edit the process_celeba.py file to point the celeba_dir variable to the path of the CelebA dataset. Then, run the script as python process_celeba.py

## Setup
Set up a conda environment with Python 3.6 and PyTorch 1.4.0. Use the attached env.yml file to create the environment with all the required libraries conda env create -f env.yml

In addition, install the zbuffer cuda library using the following steps:
1. cd zbuffer/
2. python setup.py install

Download the pretrained 3DMM and Face Segmentation models from https://drive.google.com/drive/folders/1FTb11d0PAQP77pkpmPsgK0v7MDgKANTG?usp=sharing to the ./checkpoints/ directory.

Download 3DMM_definition.zip from https://drive.google.com/file/d/1EO6Mb0ELVb82ucGxRmO8Hb_O1sb1yda9/view?usp=sharing and extract to datasets/

## Usage
To train the network from scratch, run:
python main.py --config args.txt --ngpu 1

To evaluate the network, run:
python main.py --config args.txt --ngpu 1 --resolution [256,256] --eval

## Results
Quantitative results on the CelebA, CelebA-HQ and MultiPIE datasets.
![alt text](images/quantitative.png "Quantitative Results")

Qualitative results with artificial masks.
![alt text](images/qualitative.png "Qualitative Results on Artificial Masks")

Qualitative results with real occlusions.
![alt text](images/occlusions.png "On Real Occlusions")

<!-- ****PyTorchNet**** is a Machine Learning framework that is built on top of [PyTorch](https://github.com/pytorch/pytorch). And, it uses [Visdom](https://github.com/facebookresearch/visdom) and [Plotly](https://github.com/plotly) for visualization. -->

<!-- PyTorchNet is easy to be customized by creating the necessary classes:
 1. **Data Loading**: a dataset class is required to load the data.
 2. **Model Design**: a nn.Module class that represents the network model.
 3. **Loss Method**: an appropriate class for the loss, for example CrossEntropyLoss or MSELoss.
 4. **Evaluation Metric**: a class to measure the accuracy of the results.

# Structure
PyTorchNet consists of the following packages:
## Datasets
This is for loading and transforming datasets.
## Models
Network models are kept in this package. It already includes [ResNet](https://arxiv.org/abs/1512.03385), [PreActResNet](https://arxiv.org/abs/1603.05027), [Stacked Hourglass](https://arxiv.org/abs/1603.06937) and [SphereFace](https://arxiv.org/abs/1704.08063).
## Losses
There are number of different choices available for Classification or Regression. New loss methods can be put here.
## Evaluates
There are number of different choices available for Classification or Regression. New accuracy metrics can be put here.
## Plugins
There are already three different plugins available:
1. **Monitor**:
2. **Logger**: 
3. **Visualizer**:
## Root
 - main
 - dataloader
 - train
 - test

# Setup
First, you need to download PyTorchNet by calling the following command:
> git clone --recursive https://github.com/human-analysis/pytorchnet.git

Since PyTorchNet relies on several Python packages, you need to make sure that the requirements exist by executing the following command in the *pytorchnet* directory:
> pip install -r requirements.txt

**Notice**
* If you do not have Pytorch or it does not meet the requirements, please follow the instruction on [the Pytorch website](http://pytorch.org/#pip-install-pytorch).

Congratulations!!! You are now ready to use PyTorchNet!

# Usage
Before running PyTorchNet, [Visdom](https://github.com/facebookresearch/visdom#usage) must be up and running. This can be done by:
> python -m visdom.server -p 8097

![screenshot from 2018-02-24 19-10-44](https://user-images.githubusercontent.com/24301047/36636554-1474092e-1997-11e8-8faa-cb4ee1fe9a85.png)

PyTorchNet comes with a classification example in which a [ResNet](https://arxiv.org/abs/1512.03385) model is trained for the [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.
> python [main.py](https://github.com/human-analysis/pytorchnet/blob/dev/main.py)

![screenshot from 2018-02-24 18-53-13](https://user-images.githubusercontent.com/24301047/36636539-abe73688-1996-11e8-83ea-c43318f24048.png)

![screenshot from 2018-02-24 18-58-03](https://user-images.githubusercontent.com/24301047/36636483-05f60038-1996-11e8-806e-895638396986.png)
# Configuration
PyTorchNet loads its parameters at the beginning via a config file and/or the command line.
## Config file
When PyTorchNet is being run, it will automatically load all parameters from [args.txt](https://github.com/human-analysis/pytorchnet/blob/master/args.txt) by default, if it exists. In order to load a custom config file, the following parameter can be used:
> python main.py --config custom_args.txt
### args.txt
> [Arguments]
>  
> port = 8097\
> env = main\
> same_env = Yes\
> log_type = traditional\
> save_results = No\
> \
> \# dataset options\
> dataroot = ./data\
> dataset_train = CIFAR10\
> dataset_test = CIFAR10\
> batch_size = 64


## Command line
Parameters can also be set in the command line when invoking [main.py](https://github.com/human-analysis/pytorchnet/blob/master/main.py). These parameters will precede the existing parameters in the configuration file.
> python [main.py](https://github.com/human-analysis/pytorchnet/blob/master/main.py) --log-type progressbar

$ occlusion3d -->
