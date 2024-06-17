from .backbone import Backbone
from .convgru import BasicUpdateBlock
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.optim_layer.optim_layer import DepthGradOptimLayer

def upsample_depth(depth, mask, r=8):
    """ Upsample depth field [H/r, W/r, 2] -> [H, W, 2] using convex combination """
    N, _, H, W = depth.shape # B x 1 x H x W
    mask = mask.view(N, 1, 9, r, r, H, W)
    mask = torch.softmax(mask, dim=2)

    up_depth = F.unfold(depth, [3, 3], padding=1)
    up_depth = up_depth.view(N, 1, 9, 1, 1, H, W)

    up_depth = torch.sum(mask * up_depth, dim=2)
    up_depth = up_depth.permute(0, 1, 4, 2, 5, 3)
    return up_depth.reshape(N, 1, r * H, r * W)

class OGNIDC(nn.Module):
    def __init__(self, args):
        super(OGNIDC, self).__init__()

        self.args = args
        self.GRU_iters = self.args.GRU_iters

        self.backbone = Backbone(args, mode=self.args.backbone_mode)

        self.hdim = args.gru_hidden_dim
        self.cdim = args.gru_context_dim

        # NLSPN
        self.prop_time = args.prop_time
        if args.spn_type == "nlspn":
            from .nlspn_module import NLSPN

            self.num_neighbors = args.prop_kernel * args.prop_kernel - 1
            if self.prop_time > 0:
                self.prop_layer = NLSPN(args, self.num_neighbors, 1, 3,
                                        self.args.prop_kernel)
        elif args.spn_type == "dyspn":
            from .dyspn_module import DySPN_Module
            
            self.num_neighbors = 5
            if self.prop_time > 0:
                assert self.prop_time == 6
                self.prop_layer = DySPN_Module(iteration=self.prop_time,
                                               num=self.num_neighbors,
                                               mode='yx')
        else:
            raise NotImplementedError

        # DySPN


        self.downsample_rate = args.backbone_output_downsample_rate
        self.update_block = BasicUpdateBlock(hidden_dim=self.hdim, mask_r=self.downsample_rate,
                                             conf_min=self.args.conf_min)

    def initialize_depth(self, sparse_depth):
        log_depth_init = torch.zeros_like(sparse_depth)
        log_depth_grad_init = torch.zeros_like(sparse_depth).repeat(1, 2, 1, 1) # B x 2 x H x W

        return log_depth_init, log_depth_grad_init

    def forward(self, sample):
        rgb = sample['rgb']
        dep = torch.clone(sample['dep'])
        dep_original = torch.clone(dep)
        K = sample['K']

        valid_sparse_mask = (dep > 0.0).float()

        # sparse depth needs downsample before feeding into the optim layer
        if self.downsample_rate > 1:
            if self.args.depth_downsample_method == "mean":
                dep = F.avg_pool2d(dep, self.downsample_rate)
                valid_sparse_mask = F.avg_pool2d(valid_sparse_mask, self.downsample_rate)
                dep[valid_sparse_mask > 0.0] = dep[valid_sparse_mask > 0.0] / valid_sparse_mask[valid_sparse_mask > 0.0]
                valid_sparse_mask[valid_sparse_mask > 0.0] = 1.0
            elif self.args.depth_downsample_method == "min":
                dep[dep==0.0] = 100000.0 # set the invalid values to inf
                dep = -F.max_pool2d(-dep, self.downsample_rate) # trick to do min-pooling
                valid_sparse_mask = F.max_pool2d(valid_sparse_mask, self.downsample_rate) # mask is 1 if at least one pt in neighbor
                dep[valid_sparse_mask == 0.0] = 0.0 # set invalid value back to 0.0, for safety
            else:
                raise NotImplementedError

        if self.args.depth_activation_format == "exp":
            sparse_log_depth = torch.log(dep)
        else:
            sparse_log_depth = dep

        sparse_log_depth[valid_sparse_mask == 0.0] = 0.0

        # random masking-out dep_original during training, to make the backbone more robust to sparsity
        if self.args.training_depth_mask_out_rate > 0.0 and self.training:
            batch_size = rgb.shape[0]
            keep_prob = torch.empty(batch_size).uniform_(0, 1).cuda() # for each sample in the minibatch, we randomly mask out 0% ~ 100% of the depth pixels

            do_masking = (torch.empty(batch_size).uniform_(0, 1).cuda() < self.args.training_depth_mask_out_rate).float() # for some samples, we don't do mask out
            keep_prob = (keep_prob * do_masking) + (1.0 - do_masking) # for the samples w/o do_masking, keep_prob=1.0. Else keep_prob is sampled value.

            # sample a H*W binary mask for each sample
            keep_mask = keep_prob.reshape(batch_size, 1, 1, 1).expand(dep_original.shape) # B x 1 x H x W
            keep_mask = torch.bernoulli(keep_mask) # binary
            dep_original = dep_original * keep_mask

        # backbone
        assert self.args.pred_context_feature
        _, spn_guide, spn_confidence, context, confidence_input = self.backbone(rgb, dep_original)

        if confidence_input is None:
            confidence_input = torch.ones_like(dep) # B x 1 x H x W

        net, inp = torch.split(context, [self.hdim, self.cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        # initialization
        log_depth_pred, log_depth_grad_pred_init = self.initialize_depth(dep)
        log_depth_grad_pred = log_depth_grad_pred_init

        # dummy variable fpr recording gradients
        b_init = torch.zeros_like(dep, requires_grad=True)

        log_depth_grad_predictions = [] # record the init value also
        confidence_predictions = []
        depth_predictions_up = []
        depth_predictions_up_initial = []

        for itr in range(self.GRU_iters):
            log_depth_pred = log_depth_pred.detach()
            log_depth_grad_pred = log_depth_grad_pred.detach()

            # ideally, we should whiten the log_depth_pred, so that the input to gru is always invariant to depth scale.
            log_depth_pred_mean = torch.mean(log_depth_pred, dim=(1,2,3), keepdim=True)
            log_depth_pred_whitened = log_depth_pred - log_depth_pred_mean

            net, up_mask, delta_log_depth_grad, confidence_depth_grad = self.update_block(net, inp, log_depth_pred_whitened, log_depth_grad_pred)

            log_depth_grad_pred = log_depth_grad_pred + delta_log_depth_grad

            # numerical stability
            thres = self.args.optim_layer_input_clamp
            log_depth_grad_pred = torch.clamp(log_depth_grad_pred, min=-thres, max=thres)

            # the optimization layer use the prediction from last round to accelerate convergence
            log_depth_pred, b_init = DepthGradOptimLayer.apply(log_depth_grad_pred,
                                                               sparse_log_depth,
                                                               valid_sparse_mask,
                                                               confidence_depth_grad,
                                                               confidence_input,
                                                               log_depth_pred,
                                                               b_init,
                                                               self.args.integration_alpha,
                                                               1e-5, 5000, 1.0)

            log_depth_grad_predictions.append(log_depth_grad_pred)
            confidence_predictions.append(confidence_depth_grad)

            # convex upsample
            if self.downsample_rate > 1:
                log_depth_up = upsample_depth(log_depth_pred, up_mask, r=self.downsample_rate)
            else:
                log_depth_up = log_depth_pred

            # in case where Hrgb / downsample_rate is not integer, extra interpolation is needed
            _, _ ,Hrgb, Wrgb = rgb.shape
            _, _, Hd, Wd = log_depth_up.shape
            if Hd != Hrgb or Wd != Wrgb:
                print('warning: dim mismatch!')
                log_depth_up = F.interpolate(log_depth_up, size=(Hrgb, Wrgb), mode='bilinear', align_corners=True)

            if self.args.depth_activation_format == "exp":
                depth_pred_up_init = torch.exp(log_depth_up)
            else:
                depth_pred_up_init = log_depth_up

            depth_predictions_up_initial.append(depth_pred_up_init)

            # SPN
            if self.prop_time > 0 and (self.training or itr == self.GRU_iters-1):
                if self.args.spn_type == "dyspn":
                    spn_out = self.prop_layer(depth_pred_up_init,
                                                          spn_guide,
                                                          dep_original,
                                                          spn_confidence)
                    depth_pred_up_final = spn_out['pred']
                    dyspn_offset = spn_out['offset']
                elif self.args.spn_type == "nlspn":
                    depth_pred_up_final, _, _, _, _ = self.prop_layer(depth_pred_up_init, spn_guide, spn_confidence, None)
                    dyspn_offset = None
            else:
                depth_pred_up_final = depth_pred_up_init
                dyspn_offset = None

            depth_predictions_up.append(depth_pred_up_final)

        output = {'pred': depth_predictions_up[-1], 'pred_inter': depth_predictions_up,
                  'depth_predictions_up_initial': depth_predictions_up_initial,
                  'log_depth_grad_inter': log_depth_grad_predictions,
                  'log_depth_grad_init': log_depth_grad_pred_init,
                  'confidence_depth_grad_inter': confidence_predictions,
                  'dep_down': dep,
                  'confidence_input': confidence_input,
                  'dyspn_offset': dyspn_offset
                  }

        return output
