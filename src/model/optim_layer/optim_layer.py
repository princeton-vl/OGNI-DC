import numpy as np

import torch
import torch.nn.functional as F

# import sys
# sys.path.append('.')

from .cg_batch import cg_batch
from .helpers import (
  FastFiniteDiffMatrix,
  construct_diff_matrix_sparse,
  sparse_dense_mul,
  sparse_dense_mul_prod
)

def batched_matrix_to_trucated_flattened(batched_matrix):
    batch_size = batched_matrix.shape[0]
    x_truncated = batched_matrix[:, 0, :, 1:].reshape(batch_size, -1)  # B x (H*(W-1))
    y_truncated = batched_matrix[:, 1, 1:, :].reshape(batch_size, -1)  # B x ((H-1)*W)
    return torch.cat([x_truncated, y_truncated], dim=-1).unsqueeze(-1)  # B x num_eqns x 1

class DepthGradOptimLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, batched_image_gradients, batched_sparse_depth,
                batched_valid_sparse_mask, batched_confidence, batched_input_confidence, x_init=None,
                b_init=None, lamda=0.1, rtol=1e-5, max_iter=1000, num_scale_factor=1.0):
        """Performs Surface Snapping.
          batched_image_gradients: B x 2 x H x W, first feature channel stores x gradients
          batched_sparse_depth: B x 1 x H x W
          batched_valid_sparse_mask: B x 1 x H x W, each value is either 0 or 1
          batched_confidence: B x 2 x H x W, corresponding to batched_image_gradients.
          batched_input_confidence: B x 1 x H x W, the confidence on batched_sparse_depth.
          b_init is a dummy variable that we use to initialize the gradient of b in backward path.
          it has shape B x 1 x H x W. you can pass an all-zero tensor if you want to use the functionality
        """

        batch_size, _, H, W = batched_image_gradients.shape
        device = batched_image_gradients.device
        dtype = batched_image_gradients.dtype

        if not torch.is_tensor(lamda):
            lamda = torch.ones(batch_size, 1, 1, device=device, dtype=dtype, requires_grad=False) * lamda
        else:
            assert lamda.shape == (batch_size, 1, 1) or lamda.shape == (1, 1, 1)

        if not torch.is_tensor(num_scale_factor):
            num_scale_factor = torch.ones(batch_size, 1, 1, device=device, dtype=dtype, requires_grad=False) * num_scale_factor
        else:
            assert num_scale_factor.shape == (batch_size, 1, 1) or num_scale_factor.shape == (1, 1, 1)

        ctx.max_iter = max_iter
        ctx.rtol = rtol

        # batched_confidence = torch.clamp(batched_confidence, min=1e-5)
        batched_confidence = torch.clamp(batched_confidence, min=1e-4)
        # batched_input_confidence = torch.clamp(batched_input_confidence, min=0.1)
        # valid_sparse_mask = (batched_sparse_depth > -1e9).float()  # [0, 1]
        # batched_valid_sparse_mask = batched_valid_sparse_mask.float()

        # A = construct_diff_matrix_sparse(H, W, device=device, dtype=dtype).unsqueeze(0)  # B x num_eqns x (H*W)
        A = FastFiniteDiffMatrix(H, W, device=device, dtype=dtype)

        # Compute b = A^T @ rhs
        b_top = lamda * (batched_input_confidence * batched_valid_sparse_mask * batched_sparse_depth).reshape(batch_size, -1).unsqueeze(-1)  # B x (H*W) x 1

        rhs = batched_matrix_to_trucated_flattened(batched_image_gradients)  # B x num_eqns x 1
        confidence = batched_matrix_to_trucated_flattened(batched_confidence) # B x num_eqns x 1

        # b_bottom = A.transpose(1, 2).bmm(confidence * rhs)  # B x (H*W) x 1
        b_bottom = A.bmm_transposed(confidence * rhs)

        b = b_top + b_bottom

        def batched_RTRp(p):
            ret = A.bmm_transposed(confidence * A.bmm(p))
            ret += lamda * ((batched_input_confidence * batched_valid_sparse_mask).reshape(batch_size, -1, 1) * p)
            return ret * num_scale_factor

        if x_init is not None:
            x_init = x_init.reshape(batch_size, -1, 1)

        x, info = cg_batch(batched_RTRp, b * num_scale_factor, X0=x_init, rtol=rtol, maxiter=max_iter, verbose=False)
        # print('forward:', 'stopped in %d steps' % info['niter'], 'resolution %dx%d' % (W, H))

        ctx.save_for_backward(b, rhs, x, lamda, batched_image_gradients, batched_sparse_depth,
                              batched_valid_sparse_mask, batched_confidence, batched_input_confidence, num_scale_factor)

        return x.reshape(batched_sparse_depth.shape), b_init

    @staticmethod
    def backward(ctx, gradx, grad_b_init):
        # print(grad_b_init)

        b, rhs, x, lamda, batched_image_gradients, batched_sparse_depth, \
            batched_valid_sparse_mask, batched_confidence, batched_input_confidence, num_scale_factor = ctx.saved_tensors
        batch_size, _, H, W = batched_image_gradients.shape
        device = batched_image_gradients.device
        dtype = batched_image_gradients.dtype

        gradlamda = None
        gradx = gradx.reshape(x.shape)

        A = FastFiniteDiffMatrix(H, W, device=device, dtype=dtype)
        A_sparse = construct_diff_matrix_sparse(H, W, device=device, dtype=dtype).unsqueeze(0)  # B x num_eqns x (H*W)

        # valid_sparse_mask = (batched_sparse_depth > -1e9).float()  # [0, 1]
        confidence = batched_matrix_to_trucated_flattened(batched_confidence)

        def batched_RTRp(p):
            ret = A.bmm_transposed(confidence * A.bmm(p))
            ret += lamda * ((batched_input_confidence * batched_valid_sparse_mask).reshape(batch_size, -1, 1) * p)
            return ret * num_scale_factor

        # gradb has shape B x (H*W) x 1
        if grad_b_init is not None:
            grad_b_init = grad_b_init.reshape(batch_size, -1, 1)

        gradb, info = cg_batch(batched_RTRp, gradx * num_scale_factor, X0=grad_b_init, rtol=ctx.rtol, maxiter=ctx.max_iter,
                                     verbose=False)
        # print('backward:', 'stopped in %d steps' % info['niter'], 'resolution %dx%d' % (W, H))

        if torch.isnan(gradb).any():
            return torch.zeros_like(batched_image_gradients), None, None, torch.zeros_like(batched_image_gradients), \
                   None, None, None, None, None, None

        # Compute dL/dlamda
        if lamda.requires_grad:
            # dL/dlamda through b
            term1 = (gradb.squeeze(-1) * (batched_input_confidence * batched_valid_sparse_mask * batched_sparse_depth).reshape(batch_size, -1)).sum(-1)
            # dL/dlamda through A
            term2 = (-gradb * ((batched_input_confidence * batched_valid_sparse_mask).reshape(batch_size, -1, 1) * x)).sum(-1).sum(-1)
            gradlamda = (term1 + term2).unsqueeze(-1).unsqueeze(-1)

        # Compute dL/d(batched_input_confidence)
        # dL/dlamda through b
        # gradb has shape B x (H*W) x 1
        term1 = lamda * gradb.squeeze(-1).reshape(batched_input_confidence.shape) * batched_valid_sparse_mask * batched_sparse_depth
        # dL/dlamda through A
        # x has shape B x (H*W) x 1
        term2 = lamda * (-gradb * (batched_valid_sparse_mask.reshape(batch_size, -1, 1) * x)).squeeze(-1).reshape(batched_input_confidence.shape)
        grad_batched_input_confidence = term1 + term2

        grad_confrhs = A.bmm(gradb)
        # compute dL/d(batched_image_gradients)
        # A has shape B x num_eqns x (H*W)
        grad_rhs = confidence * grad_confrhs # B x num_eqns x 1
        grad_gradient_x_truncated = grad_rhs[:, :H * (W - 1), :].reshape(batch_size, 1, H, W-1)
        grad_gradient_y_truncated = grad_rhs[:, H * (W - 1):, :].reshape(batch_size, 1, H-1, W)

        grad_batched_image_gradients = torch.zeros_like(batched_image_gradients)
        grad_batched_image_gradients[:, 0:1, :, 1:] = grad_gradient_x_truncated
        grad_batched_image_gradients[:, 1:2, 1:, :] = grad_gradient_y_truncated

        # compute dL/d(batched_confidence)
        # dL/dconf through b
        term1 = (grad_confrhs * rhs) # B x num_eqns x 1

        # dL/dconf through A
        tmp1 = torch.sqrt(confidence) * A.bmm(-gradb)
        tmp2 = torch.sqrt(confidence) * A.bmm(x)
        Nx = sparse_dense_mul_prod(A_sparse, tmp1, tmp2, x.transpose(1, 2), gradb.transpose(1, 2))
        term2 = 0.5 / torch.sqrt(confidence) * torch.sparse.sum(Nx, -1).unsqueeze(-1).to_dense()

        grad_conf = (term1 + term2)
        grad_conf_x_truncated = grad_conf[:, :H * (W - 1), :].reshape(batch_size, 1, H, W - 1)
        grad_conf_y_truncated = grad_conf[:, H * (W - 1):, :].reshape(batch_size, 1, H - 1, W)

        grad_batched_confidence = torch.zeros_like(batched_image_gradients)
        grad_batched_confidence[:, 0:1, :, 1:] = grad_conf_x_truncated
        grad_batched_confidence[:, 1:2, 1:, :] = grad_conf_y_truncated

        grad_b_init_to_return = gradb.reshape(batched_sparse_depth.shape) if grad_b_init is not None else None

        return grad_batched_image_gradients, None, None, grad_batched_confidence, grad_batched_input_confidence, \
               None, grad_b_init_to_return, \
               gradlamda, None, None, None

class MultiresDepthGradOptimLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, batched_image_gradients, batched_sparse_depth, batched_valid_sparse_mask, batched_confidence,
                lamda=0.1, rtol=1e-5, max_iter=1000, down_level=0):
        """Performs Surface Snapping.
          batched_image_gradients: B x 2 x H x W, first feature channel stores x gradients
          batched_sparse_depth: B x 1 x H x W
          batched_valid_sparse_mask: B x 1 x H x W, each value is either 0 or 1
          batched_confidence: B x 2 x H x W, corresponding to batched_image_gradients.
        """

        batch_size, _, H, W = batched_image_gradients.shape
        device = batched_image_gradients.device
        dtype = batched_image_gradients.dtype

        if not torch.is_tensor(lamda):
            lamda = torch.ones(batch_size, 1, 1, device=device, dtype=dtype, requires_grad=False) * lamda
        else:
            assert lamda.shape == (batch_size, 1, 1) or lamda.shape == (1, 1, 1)

        ctx.max_iter = max_iter
        ctx.rtol = rtol

        down_level = int(min(down_level, np.floor(np.log2(H))-1, np.floor(np.log2(W))-1))

        ctx.down_level = down_level

        # batched_confidence = torch.clamp(batched_confidence, min=1e-5)
        batched_confidence = torch.clamp(batched_confidence, min=1e-4)
        # valid_sparse_mask = (batched_sparse_depth > -1e9).float()  # [0, 1]
        # batched_valid_sparse_mask = batched_valid_sparse_mask.float()

        # A = construct_diff_matrix_sparse(H, W, device=device, dtype=dtype).unsqueeze(0)  # B x num_eqns x (H*W)
        A = FastFiniteDiffMatrix(H, W, device=device, dtype=dtype)

        # Compute b = A^T @ rhs
        b_top = lamda * (batched_valid_sparse_mask * batched_sparse_depth).reshape(batch_size, -1).unsqueeze(-1)  # B x (H*W) x 1

        rhs = batched_matrix_to_trucated_flattened(batched_image_gradients)  # B x num_eqns x 1
        confidence = batched_matrix_to_trucated_flattened(batched_confidence) # B x num_eqns x 1

        # b_bottom = A.transpose(1, 2).bmm(confidence * rhs)  # B x (H*W) x 1
        b_bottom = A.bmm_transposed(confidence * rhs)

        b = b_top + b_bottom

        levels = list(range(down_level, -1, -1)) # e.g., 3 -> [3,2,1,0]
        # start from scratch
        x_level_up = None

        for level in levels:
            stride = 2**level
            H_level = H // stride
            W_level = W // stride

            # A_level = construct_diff_matrix_sparse(H_level, W_level, device=device, dtype=dtype).unsqueeze(0) / float(stride)
            A_level = FastFiniteDiffMatrix(H_level, W_level, factor=1./stride, device=device, dtype=dtype)

            # downsample rhs
            b_level = F.avg_pool2d(b.reshape(batch_size, 1, H, W), kernel_size=stride, stride=stride)
            b_level = b_level.reshape(batch_size, H_level*W_level, 1)

            # downsample confidence
            batched_confidence_level = F.avg_pool2d(batched_confidence, kernel_size=stride, stride=stride)
            confidence_level = batched_matrix_to_trucated_flattened(batched_confidence_level)

            # downsample valid_sparse_mask
            batched_valid_sparse_mask_level = F.avg_pool2d(batched_valid_sparse_mask, kernel_size=stride, stride=stride)

            def batched_RTRp(p):
                ret = A_level.bmm_transposed(confidence_level * A_level.bmm(p))
                # ret = A_level.transpose(1, 2).bmm(confidence_level * A_level.bmm(p))
                ret += lamda * (batched_valid_sparse_mask_level.reshape(batch_size, -1, 1) * p)
                return ret

            x_level, info = cg_batch(batched_RTRp, b_level, X0=x_level_up, rtol=rtol, maxiter=max_iter, verbose=False)
            # print('forward:', 'stopped in %d steps' % info['niter'], 'resolution %dx%d' % (W_level, H_level))

            if level > 0:
                stride_up = 2 ** (level-1)
                H_level_up = H // stride_up
                W_level_up = W // stride_up
                x_level_up = F.interpolate(x_level.reshape([batch_size, 1, H_level, W_level]),
                                           size=(H_level_up, W_level_up), mode='bilinear', align_corners=False)
                x_level_up = x_level_up.reshape([batch_size, -1, 1])

        x = x_level

        ctx.save_for_backward(b, rhs, x, lamda, batched_image_gradients, batched_sparse_depth, batched_valid_sparse_mask, batched_confidence)

        return x.reshape(batched_sparse_depth.shape)

    @staticmethod
    def backward(ctx, gradx):
        b, rhs, x, lamda, batched_image_gradients, batched_sparse_depth, batched_valid_sparse_mask, batched_confidence = ctx.saved_tensors
        batch_size, _, H, W = batched_image_gradients.shape
        device = batched_image_gradients.device
        dtype = batched_image_gradients.dtype

        gradlamda = None
        gradx = gradx.reshape(x.shape)

        A = FastFiniteDiffMatrix(H, W, device=device, dtype=dtype)
        A_sparse = construct_diff_matrix_sparse(H, W, device=device, dtype=dtype).unsqueeze(0)  # B x num_eqns x (H*W)

        # valid_sparse_mask = (batched_sparse_depth > -1e9).float()  # [0, 1]
        confidence = batched_matrix_to_trucated_flattened(batched_confidence)

        levels = list(range(ctx.down_level, -1, -1))  # e.g., 3 -> [3,2,1,0]
        # start from scratch
        gradb_level_up = None

        for level in levels:
            stride = 2 ** level
            H_level = H // stride
            W_level = W // stride

            # A_level = construct_diff_matrix_sparse(H_level, W_level, device=device, dtype=dtype)\
            #               .unsqueeze(0) / float(stride)
            A_level = FastFiniteDiffMatrix(H_level, W_level, factor=1. / stride, device=device, dtype=dtype)

            # downsample rhs
            gradx_level = F.avg_pool2d(gradx.reshape(batch_size, 1, H, W), kernel_size=stride, stride=stride)
            gradx_level = gradx_level.reshape(batch_size, H_level * W_level, 1)

            # downsample confidence
            batched_confidence_level = F.avg_pool2d(batched_confidence, kernel_size=stride, stride=stride)
            confidence_level = batched_matrix_to_trucated_flattened(batched_confidence_level)

            # downsample valid_sparse_mask
            batched_valid_sparse_mask_level = F.avg_pool2d(batched_valid_sparse_mask, kernel_size=stride, stride=stride)

            def batched_RTRp(p):
                # ret = A_level.transpose(1, 2).bmm(confidence_level * A_level.bmm(p))
                ret = A_level.bmm_transposed(confidence_level * A_level.bmm(p))
                ret += lamda * (batched_valid_sparse_mask_level.reshape(batch_size, -1, 1) * p)
                return ret

            gradb_level, info = cg_batch(batched_RTRp, gradx_level, X0=gradb_level_up, rtol=ctx.rtol, maxiter=ctx.max_iter, verbose=False)
            # print('backward:', 'stopped in %d steps' % info['niter'], 'resolution %dx%d' % (W_level, H_level))

            if level > 0:
                stride_up = 2 ** (level - 1)
                H_level_up = H // stride_up
                W_level_up = W // stride_up
                gradb_level_up = F.interpolate(gradb_level.reshape([batch_size, 1, H_level, W_level]),
                                           size=(H_level_up, W_level_up), mode='bilinear', align_corners=False)
                gradb_level_up = gradb_level_up.reshape([batch_size, -1, 1])

        gradb = gradb_level

        # gradb has shape B x (H*W) x 1

        # Compute dL/dlamda
        if lamda.requires_grad:
            # dL/dlamda through b
            term1 = (gradb.squeeze(-1) * (batched_valid_sparse_mask * batched_sparse_depth).reshape(batch_size, -1)).sum(-1)
            # dL/dlamda through A
            term2 = (-gradb * (batched_valid_sparse_mask.reshape(batch_size, -1, 1) * x)).sum(-1).sum(-1)
            gradlamda = (term1 + term2).unsqueeze(-1).unsqueeze(-1)

        grad_confrhs = A.bmm(gradb)
        # compute dL/d(batched_image_gradients)
        # A has shape B x num_eqns x (H*W)
        grad_rhs = confidence * grad_confrhs # B x num_eqns x 1
        grad_gradient_x_truncated = grad_rhs[:, :H * (W - 1), :].reshape(batch_size, 1, H, W-1)
        grad_gradient_y_truncated = grad_rhs[:, H * (W - 1):, :].reshape(batch_size, 1, H-1, W)

        grad_batched_image_gradients = torch.zeros_like(batched_image_gradients)
        grad_batched_image_gradients[:, 0:1, :, 1:] = grad_gradient_x_truncated
        grad_batched_image_gradients[:, 1:2, 1:, :] = grad_gradient_y_truncated

        # compute dL/d(batched_confidence)
        # dL/dconf through b
        term1 = (grad_confrhs * rhs) # B x num_eqns x 1

        # dL/dconf through A
        tmp1 = torch.sqrt(confidence) * A.bmm(-gradb)
        tmp2 = torch.sqrt(confidence) * A.bmm(x)
        Nx = sparse_dense_mul_prod(A_sparse, tmp1, tmp2, x.transpose(1, 2), gradb.transpose(1, 2))
        term2 = 0.5 / torch.sqrt(confidence) * torch.sparse.sum(Nx, -1).unsqueeze(-1).to_dense()

        # tmp1 = torch.sqrt(confidence) * A.bmm(-gradb)
        # tmp2 = torch.sqrt(confidence) * A.bmm(x)
        # Nx = sparse_dense_mul_prod(A, tmp1, tmp2, x.transpose(1, 2), gradb.transpose(1, 2))
        # term2 = 0.5 / torch.sqrt(confidence) * torch.sparse.sum(Nx, -1).unsqueeze(-1).to_dense()

        grad_conf = (term1 + term2)
        grad_conf_x_truncated = grad_conf[:, :H * (W - 1), :].reshape(batch_size, 1, H, W - 1)
        grad_conf_y_truncated = grad_conf[:, H * (W - 1):, :].reshape(batch_size, 1, H - 1, W)

        grad_batched_confidence = torch.zeros_like(batched_image_gradients)
        grad_batched_confidence[:, 0:1, :, 1:] = grad_conf_x_truncated
        grad_batched_confidence[:, 1:2, 1:, :] = grad_conf_y_truncated

        print(grad_batched_confidence)

        return grad_batched_image_gradients, None, None, grad_batched_confidence, gradlamda, None, None, None