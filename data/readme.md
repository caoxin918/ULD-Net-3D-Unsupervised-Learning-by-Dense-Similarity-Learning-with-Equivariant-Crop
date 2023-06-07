## Data Setup

#### Pre-training

For pre-training, we use the following datasets:

- `ModelNet40`[[link](https://drive.google.com/drive/folders/1eYdZUpAwJ5k-BKxVzdaavJjYR1sIH3QA)] used in the [PointGLR](https://github.com/raoyongming/PointGLR).

- `ShapeNet55`[[link](https://drive.google.com/uc?id=1sJd5bdCg9eOo3-FYtchUVlwDgpVdsbXB)]

#### Classification

we use the data [[link](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip)] used in the [PointGLR](https://github.com/raoyongming/PointGLR).


#### Semantic Segmentation

We use the provided S3DIS [data](https://github.com/charlesq34/pointnet/blob/master/sem_seg/download_data.sh) from PointNet, which is also used in DGCNN.

Please see [here](https://github.com/charlesq34/pointnet/blob/master/sem_seg/download_data.sh) for the download details, it is worth mentioning that if you download from the original S3DIS and preprocess via <a href="../data/prepare_data/collect_indoor3d_data.py">data/prepare_data/collect_indoor3d_data.py </a>and <a href="../data/prepare_data/gen_indoor3d_h5.py">/data/prepare_data/gen_indoor3d_h5.py</a>, you need to delete an extra symbol in the raw file ([reference](https://github.com/charlesq34/pointnet/issues/45)).



#### Part Segmentation

we use the ShapeNet Part [data](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip) provided in the [PointNet](https://github.com/yanx27/Pointnet_Pointnet2_pytorch), which is also used in DGCNN.

#### Make sure to put the files in the following structure:

```
|-- ROOT
|	|-- ULD-Net
|		|-- dataset
|			|-- modelnet40
|			|-- shapenet57448xyzonly.npz
|			|-- shapenetcore_partanno_segmentation_benchmark_v0_normal
|			|-- indoor3d_sem_seg_hdf5_data
```