"""Implements helper functions."""

import torch
import torch.nn.functional as F

def sparse_dense_mul(s, d):
    """Sparse dense element-wise mul."""
    i = s._indices()
    v = s._values()
    # get values from relevant entries of dense matrix
    dv = d[i[0, :], i[1, :], i[2, :]]
    return torch.sparse.FloatTensor(i, v * dv, s.size())

class FastFiniteDiffMatrix:
    def __init__(self, H, W, factor=1.0, device='cuda', dtype=torch.float):
        self.H = H
        self.W = W
        self.factor = factor
        # self.conv_kernel_u = factor * torch.tensor([-1.0, 1.0], dtype=dtype, device=device).reshape(1, 1, 1, 2)
        # self.conv_kernel_v = factor * torch.tensor([-1.0, 1.0], dtype=dtype, device=device).reshape(1, 1, 2, 1)

    def bmm(self, x):
        # x has shape B x (H*W) x 1
        # return value has shape B x n_eqn x 1
        batch_size = x.shape[0]
        x = x.reshape(batch_size, 1, self.H, self.W)

        # x_grad_u = F.conv2d(x, self.conv_kernel_u) # B x 1 x H x (W-1)
        # x_grad_v = F.conv2d(x, self.conv_kernel_v) # B x 1 x (H-1) x W

        x_grad_u = (x[:, :, :, 1:] - x[:, :, :, :-1]) * self.factor
        x_grad_v = (x[:, :, 1:, :] - x[:, :, :-1, :]) * self.factor

        return torch.cat([x_grad_u.reshape(batch_size, -1), x_grad_v.reshape(batch_size, -1)], dim=1).unsqueeze(-1)

    def bmm_transposed(self, x):
        # x has shape B x n_eqn x 1
        # return value has shape B x (H*W) x 1
        batch_size = x.shape[0]
        x_u = x[:, :self.H * (self.W - 1)].reshape(batch_size, 1, self.H, self.W-1)
        x_v = x[:, self.H * (self.W - 1):].reshape(batch_size, 1, self.H-1, self.W)

        # out_u = F.conv2d(x_u, -self.conv_kernel_u, padding=(0, 1))  # B x 1 x H x W
        # out_v = F.conv2d(x_v, -self.conv_kernel_v, padding=(1, 0))  # B x 1 x H x W

        out_u = (F.pad(x_u, (1, 0)) - F.pad(x_u, (0, 1))) * self.factor
        out_v = (F.pad(x_v, (0, 0, 1, 0)) - F.pad(x_v, (0, 0, 0, 1))) * self.factor

        return (out_u + out_v).reshape(batch_size, -1, 1)

def construct_diff_matrix_sparse(H, W, device='cuda', dtype=torch.float):
    total_px = H * W

    total_idx = torch.arange(total_px).reshape(H, W)

    x_ind_minus_0 = torch.arange(H * (W - 1))
    x_ind_minus_1 = total_idx[:, :-1].flatten()
    x_ind_minus = torch.stack([x_ind_minus_0, x_ind_minus_1], dim=-1)  # (H*(W-1)) x 2

    x_ind_plus_0 = torch.arange(H * (W - 1))
    x_ind_plus_1 = total_idx[:, 1:].flatten()
    x_ind_plus = torch.stack([x_ind_plus_0, x_ind_plus_1], dim=-1)  # (H*(W-1)) x 2

    y_ind_minus_0 = torch.arange((H - 1) * W) + H * (W - 1)
    y_ind_minus_1 = total_idx[:-1, :].flatten()
    y_ind_minus = torch.stack([y_ind_minus_0, y_ind_minus_1], dim=-1)  # ((H-1)*W) x 2

    y_ind_plus_0 = torch.arange((H - 1) * W) + H * (W - 1)
    y_ind_plus_1 = total_idx[1:, :].flatten()
    y_ind_plus = torch.stack([y_ind_plus_0, y_ind_plus_1], dim=-1)  # ((H-1)*W) x 2

    tmp = torch.cat([x_ind_plus, y_ind_plus, x_ind_minus, y_ind_minus], dim=0) # ... x 2
    one_m_one = torch.ones(tmp.shape[0], device=device)
    one_m_one[tmp.shape[0] // 2:] = -1

    d_diff = torch.sparse_coo_tensor(tmp.transpose(0, 1), one_m_one,
                                     [H * (W - 1) + (H - 1) * W, H * W],
                                     device=device, dtype=dtype)

    return d_diff

def sparse_dense_mul_prod(s, d1, d2, d3, d4):
    """Sparse dense element-wise mul with a lookup."""
    i = s._indices()
    v = s._values()
    # get values from relevant entries of dense matrix
    dv1 = d1[i[0, :], i[1, :], 0]
    dv2 = d2[i[0, :], i[1, :], 0]
    dv3 = d3[i[0, :], 0, i[2, :]]
    dv4 = d4[i[0, :], 0, i[2, :]]
    out = v * (dv1*dv3 - dv2*dv4)
    return torch.sparse.FloatTensor(i, out, s.size())

def normal_to_log_depth_gradient(batched_K, batched_normal_map):
    # batched_K: B x 3 x 3
    # batched_normal_map: B x 3 x H x W
    device = batched_normal_map.device

    focal_x = batched_K[:, 0, 0].reshape(-1, 1, 1)
    focal_y = batched_K[:, 1, 1].reshape(-1, 1, 1)
    principal_x = batched_K[:, 0, 2].reshape(-1, 1, 1)
    principal_y = batched_K[:, 1, 2].reshape(-1, 1, 1)

    # assert (torch.abs(focal_x - focal_y) < 1e-3).all() # only supports single focal length
    # focal = focal_x
    focal = (focal_x + focal_y) / 2.0

    batch_size, _, H, W = batched_normal_map.shape
    nx, ny, nz = batched_normal_map[:, 0], batched_normal_map[:, 1], batched_normal_map[:, 2] # B x H x W each

    # nz = torch.clamp(nz, max=-1e-2)
    #
    # print('nz:', nz.min(), nz.max(), nz.mean())

    v, u = torch.meshgrid([torch.arange(H, device=device), torch.arange(W, device=device)], indexing='ij')
    v, u = v.unsqueeze(0) + 0.5, u.unsqueeze(0) + 0.5 # 1 x H x W each

    denominator = nx * (u - principal_x) + ny * (v - principal_y) + nz * focal # B x H x W

    inv_denominator = 1.0 / denominator

    # sign_denominator = torch.sign(denominator)
    # abs_denominator = torch.abs(denominator)
    #
    # # denominator = sign_denominator * torch.clamp(abs_denominator, min=1e-1)
    # denominator = sign_denominator * torch.clamp(abs_denominator, min=1.0)
    inv_denominator = torch.clamp(inv_denominator, min=-1.0, max=1.0)

    # log_depth_gradient_x = - nx / denominator
    # log_depth_gradient_y = - ny / denominator

    log_depth_gradient_x = -nx * inv_denominator
    log_depth_gradient_y = -ny * inv_denominator

    log_depth_gradient = torch.stack([log_depth_gradient_x, log_depth_gradient_y], dim=1) # B x 2 x H x W

    # abs_log_depth_gradient = torch.abs(log_depth_gradient)
    # print('grad:', abs_log_depth_gradient.min(), abs_log_depth_gradient.max(), abs_log_depth_gradient.mean())

    return log_depth_gradient

def log_depth_gradient_to_normal(batched_K, batched_log_depth_gradients):
    # batched_K: B x 3 x 3
    # batched_log_depth_gradients: B x 2 x H x W
    device = batched_log_depth_gradients.device

    focal_x = batched_K[:, 0, 0].reshape(-1, 1, 1)
    focal_y = batched_K[:, 1, 1].reshape(-1, 1, 1)
    principal_x = batched_K[:, 0, 2].reshape(-1, 1, 1)
    principal_y = batched_K[:, 1, 2].reshape(-1, 1, 1)

    # assert (torch.abs(focal_x - focal_y) < 1e-3).all()  # only supports single focal length
    # focal = focal_x
    focal = (focal_x + focal_y) / 2.0

    batch_size, _, H, W = batched_log_depth_gradients.shape

    plogzpu = batched_log_depth_gradients[:, 0]
    plogzpv = batched_log_depth_gradients[:, 1]

    v, u = torch.meshgrid([torch.arange(H, device=device), torch.arange(W, device=device)], indexing='ij')
    v, u = v.unsqueeze(0) + 0.5, u.unsqueeze(0) + 0.5  # 1 x H x W each

    pup = torch.stack([1. / focal * ((u - principal_x) * plogzpu + 1.), 1. / focal * (v - principal_y) * plogzpu, plogzpu], dim=1)
    pvp = torch.stack([1. / focal * (u - principal_x) * plogzpv, 1. / focal * ((v - principal_y) * plogzpv + 1.), plogzpv], dim=1)

    normal_from_depth = torch.cross(pvp, pup, dim=1) # B x 3 x H x W
    normalized_normal_from_depth = normal_from_depth / torch.linalg.norm(normal_from_depth, ord=2, dim=1, keepdim=True)

    return normalized_normal_from_depth


