import torch.nn as nn
from modules.dgcnn_dense import DGCNN_dense

class DenseSimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder=DGCNN_dense, args=None, seg_num_all=50):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(DenseSimSiam, self).__init__()
        channel = 256
        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(args=args, seg_num_all=seg_num_all)

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(channel, 512, bias=False),
                                        nn.BatchNorm1d(512),
                                        nn.LeakyReLU(negative_slope=0.2, inplace=True), # hidden layer
                                        nn.Linear(512, channel, bias=False)) # output layer

        self.classifier = nn.Sequential(
            nn.Conv1d(channel, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),  # hidden layer
            nn.Conv1d(512, channel, 1, bias=False)
        )

    def forward(self, x1, x2, get_feature=False):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        z1, ptfeature1, x_z1 = self.encoder(x1, get_feature) # NxC
        z2, ptfeature2, x_z2 = self.encoder(x2, get_feature) # NxC
        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        ptfeature1_pred = self.classifier(ptfeature1)
        ptfeature2_pred = self.classifier(ptfeature2)

        return p1, p2, z1.detach(), z2.detach(), ptfeature1_pred, ptfeature2_pred, ptfeature1.detach(), ptfeature2.detach(), x_z1, x_z2

class DenseSimSiam_Region(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, args=None):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(DenseSimSiam_Region, self).__init__()
        channel = 256
        self.project = nn.Sequential(
            nn.Conv1d(channel, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),  # hidden layer
            nn.Conv1d(512, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),  # hidden layer
            nn.Conv1d(256, channel, 1, bias=False),
            nn.BatchNorm1d(channel)
        )  # output layer
        self.predictor_region = nn.Sequential(
            nn.Conv1d(channel, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),  # hidden layer
            nn.Conv1d(512, channel, 1, bias=False)
        )

    def forward(self, x1, x2, get_feature=False):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view

        region1 = self.project(x1) # NxC
        region2 = self.project(x2) # NxC

        region1_pred = self.predictor_region(region1)
        region2_pred = self.predictor_region(region2)

        return region1_pred, region2_pred, region1.detach(), region2.detach()