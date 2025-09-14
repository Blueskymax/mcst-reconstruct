import torch
import torch.nn as nn


class LossCompute_NLL(nn.Module):
    # 极大似然
    def __init__(self):
        super(LossCompute_NLL, self).__init__()

    def forward(self, output, output_sigma, labels):
        # labels (batch, frames_num, tg_num_max, feature_size)
        # output (batch, frames_num, tg_num_max, feature_size)
        # output_sigma (batch, frames_num, tg_num_max, feature_size)
        loss = torch.sum(torch.sum(torch.sum(torch.sum(0.5 * torch.exp(-output_sigma) * pow((output - labels), 2)
                                                       + 0.5 * output_sigma + 1, dim=0), dim=0), dim=0), dim=0)
        return loss/output.shape[0]/output.shape[1]/output.shape[2]