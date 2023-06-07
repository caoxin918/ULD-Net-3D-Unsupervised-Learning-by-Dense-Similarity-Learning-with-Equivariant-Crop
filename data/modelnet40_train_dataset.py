import os
import torch.utils.data as data
from data.custom_transforms_random import *


class TrainModelNet(data.Dataset):
    def __init__(self, root, labeldir, mode, args=None, split='train', res1=4096, res2=2048, inv_list=[], eqv_list=[], npoints=2048,scale=(0.5, 1), class_choice=None, normal_channel=False):
        self.root  = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.split = split
        self.res1  = res1
        self.res2  = res2
        self.npoints = npoints
        self.mode  = mode
        self.scale = scale
        self.args = args
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
        self.self_supervision = args.self_supervision
        root = self.root
        if args.subset10:
            if args.self_supervision:
                self.points = np.load(root + 'ModelNet10_normal_2048_train_points.npy')
                self.labels = None
            elif self.split == 'train':
                self.points = np.load(root + 'ModelNet10_normal_2048_train_points.npy')
                self.labels = np.load(root + 'ModelNet10_normal_2048_train_label.npy')
            else:
                self.points = np.load(root + 'ModelNet10_normal_2048_test_points.npy')
                self.labels = np.load(root + 'ModelNet10_normal_2048_test_label.npy')
        else:
            if args.self_supervision:
                self.points = np.load(root + 'ModelNet40_normal_2048_train_points.npy')
                self.labels = None
            elif self.split == 'train':
                self.points = np.load(root + 'ModelNet40_normal_2048_train_points.npy')
                self.labels = np.load(root + 'ModelNet40_normal_2048_train_label.npy')
            else:
                self.points = np.load(root + 'ModelNet40_normal_2048_test_points.npy')
                self.labels = np.load(root + 'ModelNet40_normal_2048_test_label.npy')

        if not args.use_normal:
            self.points = self.points[:, :, :3]

        if args.dataset_rate < 1.0:
            print('### ATTENTION: Only', args.dataset_rate, 'data of training set are used ###')
            num_instance = len(self.labels)
            index = np.random.permutation(num_instance)[0: int(num_instance * args.dataset_rate)]
            self.points = self.points[index]
            if self.labels is not None:
                self.labels = self.labels[index]

        if not args.subset10:
            print('Successfully load ModelNet40 with', self.points.shape[0], 'instances')
        else:
            print('Successfully load ModelNet10 with', self.points.shape[0], 'instances')


    def __getitem__(self, index):
        # load data
        index = self.shuffled_indices[index]
        pt_idxs = np.arange(0, self.points.shape[1])  # 2048
        data = self.points[index, pt_idxs].copy()
        if not self.self_supervision:
            cls = self.labels[index]
        if not self.normal_channel:
                image = data[:, 0:3]
        else:
                image = data[:, 0:6]
        # transform data

        if self.mode=='compute' and self.view == -1:
            image, choice = self.transform_image(index, image)
            cls = int(cls)
            return (cls,) + (image,)
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
                if self.args.noise > 0:
                    image = self.random_noise(index, image)
                image = self.sample(image, self.npoints)
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
            image2 = self.sample(image2, self.res1)
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
        self.random_scale = [PointcloudScale(p=1,N=N) for _ in range(2)] # Control this later (NOTE)]
        self.random_rotate_perturbation  = [PointcloudRotatePerturbation(p=1,N=N) for _ in range(2)] # Control this later (NOTE)
        self.random_jitter = [PointcloudJitter(p=1,N=N,points_num=self.npoints) for _ in range(2)] # Control this later (NOTE)
        self.random_translate = [PointcloudTranslate(p=1,N=N) for _ in range(2)]      # Control this later (NOTE)
        self.random_normalize    = [PointcloudNormalize() for _ in range(2)]

        # Transforms for equivariance.
        self.random_crop  = PointcloudRandomCrop(p=1,N=N,res1=self.res1,res2=self.res2)

        # Tensor transform. 
        self.transform_tensor = PointcloudToTensor()
        if self.args.noise > 0:
            self.random_noise = PointcloudNoise(p=1, N=N, std=self.args.noise, points_num=self.npoints)

    def __len__(self):
        return self.points.shape[0]
        

  
            
       
