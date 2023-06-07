import os
import torch.utils.data as data
from data.custom_transforms_random import *


class TrainShapeNet(data.Dataset):
    def __init__(self, root, labeldir, mode, args=None, split='train', res1=4096, res2=2048, inv_list=[], eqv_list=[], npoints=2048,scale=(0.5, 1), class_choice=None, normal_channel=False):
        self.root  = root
        self.split = split
        self.res1  = res1
        self.res2  = res2
        self.mode  = mode
        self.scale = scale
        self.view  = -1

        assert split == 'train', 'split should be [train].'
        self.inv_list = inv_list
        self.eqv_list = eqv_list
        self.labeldir = labeldir

        self.npoints = npoints
        self.normal_channel = normal_channel
        self.class_choice = class_choice

        self.load_imdb(args)
        self.reshuffle(npoints)

    def load_imdb(self, args):
        self.train_npy = os.path.join(self.root, "shapenet57448xyzonly.npz")
        self.td = dict(np.load(self.train_npy))
        self.points = self.td["data"]

    def __getitem__(self, index):
        # load data
        index = self.shuffled_indices[index]
        data = self.points[index].copy()
        if not self.normal_channel:
                image = data[:, 0:3]
        else:
                image = data[:, 0:6]
        # transform data
        if self.mode=='compute' and self.view == -1:
            image, choice = self.transform_image(index, image)
            return (None,) + (image,)
        else:
            image = self.transform_image(index, image)
            return (index,) + image

    def reshuffle(self, npoints):
        """
        Generate random floats for all image data to deterministically random transform.
        This is to use random sampling but have the same samples during clustering and 
        training within the same epoch. 
        """
        self.shuffled_indices = np.arange(self.points.shape[0])
        np.random.shuffle(self.shuffled_indices)
        self.init_transforms(npoints)

    def sample(self, points, res):

        # fps_ids = fps(points, res)
        # return points[fps_ids]
        points = self.transform_tensor(points)
        return index_points(points.unsqueeze(0), farthest_point_sample(points.unsqueeze(0),res)).squeeze(0)
    def transform_image(self, index, image):

        if self.mode == 'compute':
            if self.view == 1:
                image = self.transform_inv(index, image, 0)
                image = self.transform_tensor(image)
            elif self.view == 2:
                image = self.transform_inv(index, image, 1)
                image = self.sample(image, self.res1)
                image = self.transform_tensor(image)
            else:
                image = self.random_normalize[0](image)
                image = self.transform_tensor(image)
                return image, None
            return (image, )
        elif 'train' in self.mode:
            # Invariance transform.
            image1 = self.transform_inv(index, image, 0)
            image1 = self.transform_tensor(image1)

            if self.mode == 'baseline_train':
                return (image1, )

            image2 = self.transform_inv(index, image, 1)
            image2 = self.transform_tensor(image2)

            return (image1, image2)
        else:
            raise ValueError('Mode [{}] is an invalid option.'.format(self.mode))


    def transform_inv(self, index, image, ver):
        """
        Hyperparameters same as MoCo v2. 
        (https://github.com/facebookresearch/moco/blob/master/main_moco.py)
        """
        if 'scale' in self.inv_list:
            image = self.random_scale[ver](index, image)
        if 'rotateperturbation' in self.inv_list:
            image = self.random_rotate_perturbation[ver](index, image)
        if 'jitter' in self.inv_list:
            image = self.random_jitter[ver](index, image)
        if 'normalize' in self.inv_list:
            image = self.random_normalize[ver](image)
        if 'translate' in self.inv_list:
            image = self.random_translate[ver](index, image)
        
        return image



    def transform_eqv(self, indice, image, feature):
        if 'randomcrop' in self.eqv_list:
            image = self.random_crop(indice, image, feature)
        return image


    def init_transforms(self,npoints):
        N = self.points.shape[0]
        
        # Base transform.
        self.transform_base = BaseTransform(self.res2)
        
        # Transforms for invariance. 
        # Color jitter (4), gray scale, blur. 
        self.random_scale = [PointcloudScale(p=1,N=N) for _ in range(2)] # Control this later (NOTE)]
        self.random_rotate_perturbation  = [PointcloudRotatePerturbation(p=1,N=N) for _ in range(2)] # Control this later (NOTE)
        self.random_jitter = [PointcloudJitter(p=1,N=N,points_num=self.npoints) for _ in range(2)] # Control this later (NOTE)
        self.random_translate = [PointcloudTranslate(p=1,N=N) for _ in range(2)]      # Control this later (NOTE)
        self.random_normalize    = [PointcloudNormalize() for _ in range(2)]

        # Transforms for equivariance.
        self.random_crop  = PointcloudRandomCrop(p=1,N=N,res1=self.res1,res2=self.res2)
        # Tensor transform. 
        self.transform_tensor = PointcloudToTensor()

    def __len__(self):
        return self.points.shape[0]
        

  
            
       
