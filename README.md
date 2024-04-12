# Weight Initialization for Semantic Segmentation

This project was created by Justin, Ayleen, Jacky, Hanson and Harry (Group 12) for our MIE424 Final Project. We aim to assess the impact of different weight initializations, namely Zero Initialization, Kaiming Uniform and Normal, Xavier Uniform and Normal, and GradInit on the performance of an FCN32s model architecture for semantic segmentation.

## Environment Dependencies
To install the necessary dependencies, you can use the following list as a guide. Some packages require specific versions or higher:

- `fcn >= 6.1.5`
- `numpy`
- `Pillow`
- `pytz`
- `scipy`
- `torch >= 0.2.0`
- `torchvision >= 0.1.8`
- `tqdm`

## Dataset 
The Pascal VOC 2012 Dataset was used for this project, which can be downloaded from this link: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit

Click the "training/validation" data (2GB tar) to download.

## Usage
To train the model, run the `train_fcn32s.py` script from the command line. You can specify various arguments to customize the training process. For example, to specify which GPU to use and to set the maximum number of iterations, you can use the following command:

```bash
python train_fcn32s.py -g 0 -max_iter 10000
```

Please read through the train_fcn32s.py file to see all available arguments you can specify.

## Acknowledgments
Our FCN32s architecture was referenced from the [Fully Convolutional Networks for Semantic Segmentation](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf) paper by Long et. al. We also used the [GradInit code by Chen Zhu](https://github.com/zhuchen03/gradinit) for GradInit weight initialization.

We would like to thank Prof. Elias Khalil for teaching such an amazing course, and the MIE424 TAs as well for supporting us!!
