import random
import os 
import logging
import pickle 
import numpy as np 
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import faiss 
from data.shapenetpart_train_dataset import TrainShpeNetPart
from data.shapenetpart_eval_dataset import EvalShpeNetPart
from data.modelnet40_train_dataset import TrainModelNet
from data.modelnet40_eval_dataset import EvalModelNet
from data.shapenet_train_dataset import TrainShapeNet
from modules.ContrastModel import DenseSimSiam_Region
################################################################################
#                                  General-purpose                             #
################################################################################

def str_list(l):
    return '_'.join([str(x) for x in l]) 

def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Logging to a file
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(file_handler)

    # Logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)

    return logger

class Logger(object):
    """ Class to update every epoch to keep trace of the results
    Methods:
        - log() log and save
    """

    def __init__(self, path):
        self.path = path
        self.data = []

    def log(self, train_point):
        self.data.append(train_point)
        with open(os.path.join(self.path), 'wb') as fp:
            pickle.dump(self.data, fp, -1)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_datetime(time_delta):
    days_delta = time_delta // (24*3600)
    time_delta = time_delta % (24*3600)
    hour_delta = time_delta // 3600 
    time_delta = time_delta % 3600 
    mins_delta = time_delta // 60 
    time_delta = time_delta % 60 
    secs_delta = time_delta 

    return '{}:{}:{}:{}'.format(days_delta, hour_delta, mins_delta, secs_delta)


################################################################################
#                                General torch ops                             #
################################################################################

def freeze_all(model):
    for param in model.module.parameters():
        param.requires_grad = False 


def initialize_classifier(args):
    classifier = DenseSimSiam_Region(args=args)
    classifier = nn.DataParallel(classifier, device_ids=[0])
    classifier = classifier.cuda()

    return classifier

################################################################################
#                                   Faiss related                              #
################################################################################

def get_faiss_module(args):
    res = faiss.StandardGpuResources()
    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = False 
    cfg.device = 0 #NOTE: Single GPU only.
    idx = faiss.GpuIndexFlatL2(res, args.in_dim, cfg)

    return idx

def get_init_centroids(args, K, featlist, index):
    clus = faiss.Clustering(args.in_dim, K)
    clus.seed  = np.random.randint(args.seed)
    clus.niter = args.kmeans_n_iter
    clus.max_points_per_centroid = 10000000
    clus.train(featlist, index)

    return faiss.vector_float_to_array(clus.centroids).reshape(K, args.in_dim)

def module_update_centroids(index, centroids):
    index.reset()
    index.add(centroids)

    return index 

def fix_seed_for_reproducability(seed):
    """
    Unfortunately, backward() of [interpolate] functional seems to be never deterministic. 

    Below are related threads:
    https://github.com/pytorch/pytorch/issues/7068 
    https://discuss.pytorch.org/t/non-deterministic-behavior-of-pytorch-upsample-interpolate/42842?u=sbelharbi 
    """
    # Use random seed.
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule
def worker_init_fn(seed):
    return lambda x: np.random.seed(seed + x)

################################################################################
#                               Training Pipelines                             #
################################################################################

def eqv_transform_if_needed(args, dataloader, indice, points, features):
    if args.equiv:
        input = dataloader.dataset.transform_eqv(indice, points, features)
        return input
    elif features is not None:
        return points, features
    else:
        return points, None


def get_transform_params(args):
    inv_list = []
    eqv_list = []
    if args.augment:
        # scale/rotateperturbation/jitter/normalize/translate
        if args.scale:
            inv_list.append('scale')
        if args.rotateperturbation:
            inv_list.append('rotateperturbation')
        if args.jitter:
            inv_list.append('jitter')
        if args.normalize:
            inv_list.append('normalize')
        if args.translate:
            inv_list.append('translate')
        if args.equiv:
            # dropout/randomcrop/cutout/upsample
            if args.randomcrop:
                eqv_list.append('randomcrop')
    
    return inv_list, eqv_list

def collate_train(batch):
    if batch[0][-1] is not None:
        indice = [b[0] for b in batch]
        image1 = torch.stack([b[1] for b in batch])
        image2 = torch.stack([b[2] for b in batch])
        return indice, image1, image2
    indice = [b[0] for b in batch]
    image1 = torch.stack([b[1] for b in batch])

    return indice, image1

def collate_eval(batch):
    indice = [b[0] for b in batch]
    image = torch.stack([b[1] for b in batch])
    return indice, image

def get_dataset(args, mode, inv_list=[], eqv_list=[]):
    if args.dataset == 'modelnet':
        if mode == 'train':
            dataset = TrainModelNet(args.data_root,args=args, labeldir=args.save_model_path, res1=args.res1, res2=args.res2,
                                      split='train', mode='compute', inv_list=inv_list, eqv_list=eqv_list, npoints=args.num_points,scale=(args.min_scale, 1),normal_channel=args.normal_channel)
        elif mode == 'train_val':
            dataset = EvalModelNet(args.data_root,args=args, res=args.num_points, split='val', mode='test',
                                      npoints=args.num_points, normal_channel=args.normal_channel)
        elif mode == 'eval_val':
            dataset = EvalModelNet(args.data_root,args=args, res=args.res, split=args.val_type,
                                     mode='test', label=False, npoints=args.num_points, normal_channel=args.normal_channel)
        elif mode == 'eval_test':
            dataset = EvalModelNet(args.data_root,args=args, res=args.res, split='val', mode='test', npoints=args.num_points, normal_channel=args.normal_channel)
    else:
            dataset = TrainShapeNet(args.data_root,args=args, labeldir=args.save_model_path, res1=args.res1, res2=args.res2,
                                      split='train', mode='compute', inv_list=inv_list, eqv_list=eqv_list, npoints=args.num_points,scale=(args.min_scale, 1),normal_channel=args.normal_channel)
    return dataset
