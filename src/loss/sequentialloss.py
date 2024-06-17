from . import BaseLoss
import torch

class SequentialLoss(BaseLoss):
    def __init__(self, args):
        super(SequentialLoss, self).__init__(args)

        self.loss_name = []
        self.t_valid = 0.0001

        for k, _ in self.loss_dict.items():
            self.loss_name.append(k)

        self.intermediate_loss_weight = args.intermediate_loss_weight

    def compute(self, sample, output):
        loss_val = []

        for idx, loss_type in enumerate(self.loss_dict):
            loss = self.loss_dict[loss_type]
            loss_func = loss['func']

            if loss_func is None:
                continue

            all_depth_pred = output['pred_inter']
            all_grad_pred = output['log_depth_grad_inter']
            gt_depth = sample['gt']
            confidence_input = output['confidence_input']

            loss_tmp = 0.0
            if loss_type in ['SeqL1', 'SeqL2']:
                loss_tmp += loss_func(all_depth_pred, gt_depth) * 1.0

                if self.intermediate_loss_weight > 0.0:
                    all_depth_pred_init = output['depth_predictions_up_initial']
                    loss_tmp += loss_func(all_depth_pred_init, gt_depth) * self.intermediate_loss_weight

            elif loss_type in ['SeqGradL1']:
                loss_tmp += loss_func(all_grad_pred, gt_depth) * 1.0

            else:
                raise NotImplementedError

            loss_tmp = loss['weight'] * loss_tmp
            loss_val.append(loss_tmp)

        loss_val = torch.stack(loss_val)

        loss_sum = torch.sum(loss_val, dim=0, keepdim=True)

        loss_val = torch.cat((loss_val, loss_sum))
        loss_val = torch.unsqueeze(loss_val, dim=0).detach()

        return loss_sum, loss_val
