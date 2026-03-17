import torch
import torch.nn as nn 

class MultiTaskLoss(nn.Module):
    def __init__(self, del_task_loss_fn, sat_task_loss_fn, is_regression):
        super().__init__()
        self.del_task_loss_fn = del_task_loss_fn
        self.sat_task_loss_fn = sat_task_loss_fn
        self.is_regression = is_regression
        self.n_tasks = len(is_regression)
        self.log_vars = torch.nn.Parameter(torch.zeros(self.n_tasks))

    def forward(self, del_pred, sat_pred, del_target, sat_target):
        del_loss = self.del_task_loss_fn(del_pred.view(-1), del_target)
        sat_loss = self.sat_task_loss_fn(sat_pred.view(-1), sat_target)

        losses = torch.stack([del_loss, sat_loss])
        dtype = losses.dtype
        device = losses.device
        self.is_regression = self.is_regression.to(device=device, dtype=dtype)
        std = (torch.exp(self.log_vars)**(1/2)).to(device=device, dtype=dtype)

        coeff = (1/((self.is_regression+1)*(std**2)))
        multi_task_losses = coeff*losses + torch.log(std)

        loss = torch.sum(multi_task_losses)

        return loss, del_loss, sat_loss
