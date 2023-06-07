import os 
import torch 
import torch.nn as nn 
import torch.utils.data as data
import torchvision
# from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np 
from PIL import Image, ImageFilter
from data.custom_transforms_random import *
import json


class EvalModelNet(data.Dataset):
    def __init__(self, root, split, mode, args, res=128, npoints=2048, transform_list=['normalize'], label=True, class_choice=None, normal_channel=False):
        #, 'scale', 'rotateperturbation', 'jitter', 'translate'
        self.root  = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.split = split
        self.res  = res
        self.mode  = mode
        self.label = label

        self.npoints = npoints
        self.normal_channel = normal_channel
        self.args = args

        self.load_imdb(args)
        self.transform_list = transform_list
        self.transform_tensor = PointcloudToTensor()
        self.class_choice = class_choice

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


    def sample(self, points, res):
        if res < points.shape[0]:
            fps_ids = fps(points, res)
            return points[fps_ids]
        else:
            return points

    def __getitem__(self, index):
        # load data
        data = self.points[index].copy()
        if not self.self_supervision:
            cls = self.labels[index]
        if not self.normal_channel:
                image = data[:, 0:3]
        else:
                image = data[:, 0:6]
        image = self._image_transform(image=image, mode=self.mode, index=index)
        image = self.transform_tensor(image)
        image = self.sample(image, self.res)
        cls = int(cls)
        data = (cls, ) + self.transform_data(image,None)
        return data
    def transform_data(self,image,label):
        if not self.label:
            return (image,None)
        # label = self._label_transform(label)
        return image, None
        # return image
    def _label_transform(self, label):
        label = np.array(label)
        label = torch.LongTensor(label)                            

        return label


    def _image_transform(self, image, mode, index):
        # if self.mode == 'test':
            transform = self._get_data_transformation()
            if self.args.noise > 0:
                image = self.random_noise(index, image)
            for i,trans in enumerate(transform):
                if i==0:
                    image = trans(image)
                else:
                    image = trans(index, image)
            return image
        # else:
        #     raise NotImplementedError()


    def _get_data_transformation(self):
        trans_list = []
        N = self.points.shape[0]
        if 'normalize' in self.transform_list:
            trans_list.append(PointcloudNormalize())
        if 'scale' in self.transform_list:
            trans_list.append(PointcloudScale(p=1,N=N))
        if 'rotateperturbation' in self.transform_list:
            trans_list.append(PointcloudRotatePerturbation(p=1,N=N))
        if 'jitter' in self.transform_list:
            trans_list.append(PointcloudJitter(p=1,N=N))
        if 'translate' in self.transform_list:
            trans_list.append(PointcloudTranslate(p=1,N=N))
        if self.args.noise > 0:
            self.random_noise = PointcloudNoise(p=1, N=N, std=self.args.noise, points_num=self.npoints)

        return trans_list

    def __len__(self):
        return self.points.shape[0]
        

  
            
       
