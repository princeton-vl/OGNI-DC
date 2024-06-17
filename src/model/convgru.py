# adapted from https://github.com/princeton-vl/RAFT/blob/master/core/update.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthGradHead(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128):
        super(DepthGradHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class DepthDirectHead(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128):
        super(DepthDirectHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 1, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class DepthDirectEncoder(nn.Module):
    def __init__(self):
        super(DepthDirectEncoder, self).__init__()
        self.convd1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convd2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv = nn.Conv2d(64, 64-1, 3, padding=1)

    def forward(self, depth):
        dep = F.relu(self.convd1(depth))
        dep = F.relu(self.convd2(dep))
        
        out = F.relu(self.conv(dep))

        return torch.cat([out, depth], dim=1)

class ConfidenceHead(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=32):
        super(ConfidenceHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.conv2(self.relu(self.conv1(x))))

class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=64, input_dim=64+64):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q
        return h

class BasicDepthEncoder(nn.Module):
    def __init__(self):
        super(BasicDepthEncoder, self).__init__()
        self.convd1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convd2 = nn.Conv2d(64, 32, 3, padding=1)

        self.convg1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convg2 = nn.Conv2d(64, 32, 3, padding=1)

        self.conv = nn.Conv2d(64, 64-3, 3, padding=1)

    def forward(self, depth, depth_grad):
        dep = F.relu(self.convd1(depth))
        dep = F.relu(self.convd2(dep))

        gra = F.relu(self.convg1(depth_grad))
        gra = F.relu(self.convg2(gra))

        out = F.relu(self.conv(torch.cat([dep, gra], dim=1)))
        return torch.cat([out, depth, depth_grad], dim=1)

class BasicUpdateBlock(nn.Module):
    def __init__(self, hidden_dim=64, input_dim=64, mask_r=8, conf_min=0.01):
        super(BasicUpdateBlock, self).__init__()
        self.encoder = BasicDepthEncoder()
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=64+hidden_dim)
        self.depth_grad_head = DepthGradHead(input_dim=hidden_dim)

        self.conf_min = conf_min
        if self.conf_min < 1.0:
            self.confidence_head = ConfidenceHead(input_dim=hidden_dim)

        self.mask_r = mask_r
        if self.mask_r > 1:
            self.mask = nn.Sequential(
                nn.Conv2d(hidden_dim, 256, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, mask_r * mask_r * 9, 1, padding=0))

    def forward(self, net, inp, depth, depth_grad):
        # net: hidden; inp: ctx
        # depth and depth_grad should be detached before feeding into this layer
        depth_features = self.encoder(depth, depth_grad) # B x 64 x H x W
        inp = torch.cat([inp, depth_features], dim=1) # B x 128 x H x W

        net = self.gru(net, inp)

        delta_depth_grad = self.depth_grad_head(net)

        if self.conf_min < 1.0:
            confidence_depth_grad = self.confidence_head(net)
            confidence_depth_grad = self.conf_min + confidence_depth_grad * (1. - self.conf_min)
        else:
            confidence_depth_grad = torch.ones_like(delta_depth_grad)

        if self.mask_r > 1:
            mask = self.mask(net)
        else:
            mask = None
        
        return net, mask, delta_depth_grad, confidence_depth_grad

class DirectDepthUpdateBlock(nn.Module):
    def __init__(self, hidden_dim=64, input_dim=64, mask_r=8):
        super(DirectDepthUpdateBlock, self).__init__()
        self.encoder = DepthDirectEncoder()
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=64 + hidden_dim)
        self.depth_head = DepthDirectHead(input_dim=hidden_dim)

        self.mask_r = mask_r
        if self.mask_r > 1:
            self.mask = nn.Sequential(
                nn.Conv2d(hidden_dim, 256, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, mask_r * mask_r * 9, 1, padding=0))

    def forward(self, net, inp, depth):
        # net: hidden; inp: ctx
        # depth and depth_grad should be detached before feeding into this layer
        depth_features = self.encoder(depth)  # B x 64 x H x W
        inp = torch.cat([inp, depth_features], dim=1)  # B x 128 x H x W

        net = self.gru(net, inp)

        delta_depth = self.depth_head(net)

        if self.mask_r > 1:
            mask = self.mask(net)
        else:
            mask = None

        return net, mask, delta_depth