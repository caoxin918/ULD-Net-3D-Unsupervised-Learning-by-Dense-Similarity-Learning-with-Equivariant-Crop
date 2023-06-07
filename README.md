# ULD-Net: 3D Unsupervised Learning by Dense Similarity Learning with Equivariant-Crop

*[Yu Tian](https://tianyutienyu.github.io/)*<sup>1,2</sup>,
*Da Song*<sup>1,2</sup>,
*Mengna Yang*<sup>1,2</sup>,
*Jie Liu*<sup>1,2,3</sup>,
*[Guohua Geng](https://ist.nwu.edu.cn/info/1016/1503.htm)*<sup>1,2</sup>,
*[Mingquan Zhou](https://www.researchgate.net/profile/Mingquan-Zhou)*<sup>1,2</sup>,
*[Kang Li](https://faculty.nwu.edu.cn/KangLi/en/index.htm)*<sup>1,2</sup>,
*[Xin Cao](https://caoxin918.github.io/)*<sup>1,2,4</sup>

<sup>1</sup> Northwest University, Xi’an, Shaanxi, China,
<sup>2</sup> National and Local Joint Engineering Research Center for Cultural Heritage Digitization, Xi’an, China,
<sup>3</sup> [e-mail](mailto:jieliu2017@126.com),
<sup>4</sup> [e-mail](mailto:xin_cao@163.com)

This repository is the official implementation of [ULD-Net: 3D Unsupervised Learning by Dense Similarity Learning with Equivariant-Crop](https://opg.optica.org/josaa/abstract.cfm?URI=josaa-39-12-2343), JOSA.A. 2022. 

Please feel free to reach out for any questions or discussions!

## Setup
Setting up for this project involves installing dependencies and preparing the datasets. 

### Installing dependencies
To install all the dependencies, please run the following:
~~~
pip install -r requirements.txt
~~~

### Preparing Dataset 
For the details in the data setup, please see [data/readme.md](data/readme.md).
## Pre-training via ULD-Net
Below are training and testing commands to train ULD-Net. 
### Training
Below line will run the training code with default setting on ShapeNet55. 
~~~
sh ./sh_files/pretrain_shapenet.sh
~~~
Below line will run the training code with default setting on ModelNet40. 
~~~
sh ./sh_files/pretrain_modelnet.sh
~~~

### Testing 
Below line will run the testing code with default setting. 
~~~
sh ./sh_files/test.sh
~~~

## Pretrained Models 
We have [pretrained weights](https://drive.google.com/drive/folders/1kl9UWgsY9lLY-iRfffDNocJKfzsfhUN7). 

## Downstream
### Classification on ModelNet40
~~~
sh ./sh_files/finetune_modelnet.sh
~~~
### Part Segmentation on ShapeNet Part
~~~
sh ./sh_files/finetune_shapenetpart.sh
~~~
### Semantic Segmentation on S3DIS
~~~
sh ./sh_files/finetune_S3DIS.sh
~~~
## Citation
If you find ULD-Net useful in your research, please consider citing:
```
@article{tian2022uld,
  title={ULD-Net: 3D unsupervised learning by dense similarity learning with equivariant-crop},
  author={Tian, Yu and Song, Da and Yang, Mengna and Liu, Jie and Geng, Guohua and Zhou, Mingquan and Li, Kang and Cao, Xin},
  journal={JOSA A},
  volume={39},
  number={12},
  pages={2343--2353},
  year={2022},
  publisher={Optica Publishing Group}
}
```
## Acknowledgements 
We would like to thank and acknowledge referenced codes from the following repositories:

https://github.com/WangYueFt/dgcnn

https://github.com/charlesq34/pointnet

https://github.com/charlesq34/pointnet2

https://github.com/AnTao97/dgcnn.pytorch

https://github.com/hansen7/OcCo

https://github.com/yichen928/STRL

https://github.com/raoyongming/PointGLR
