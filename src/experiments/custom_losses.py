import torch.nn as nn
import torch

class IoULoss(nn.Module):
    def __init__(self, device):
        super(IoULoss,self).__init__()
        self.device = device

    def forward(self, pred,label):
        intersection = torch.mul(pred,label)
        union = torch.add(pred, label)
        union = torch.add(union,-1,intersection)

        return torch.add(torch.tensor(1.0).to(self.device),-1,torch.mean(torch.div(torch.sum(torch.sum(intersection,-1),-1),torch.sum(torch.sum(union,-1),-1))))




