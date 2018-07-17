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


#TODO: make it cope with 4 d ouputs with multiple channels
class DistanceLoss(nn.Module):
    def __init__(self, device):
        super(DistanceLoss,self).__init__()
        self.device = device

    def forward(self, pred, future_centroid):
        pred = torch.squeeze(pred,1)
        batch, rows, cols = pred.size()


        y_p = torch.arange(cols).to(self.device)
        y_p = y_p.repeat(batch,1)
        y_p = y_p.unsqueeze(2)

        mean_y = torch.bmm(pred, y_p)
        mean_y = torch.sum(mean_y,1)
        mean_y = torch.div(mean_y,torch.sum(pred))


        x_p = torch.arange(rows).to(self.device)
        x_p = x_p.repeat(batch,1)
        x_p = x_p.unsqueeze(1)

        mean_x = torch.bmm(x_p,pred)
        mean_x = torch.sum(mean_x,2)
        mean_x = torch.div(mean_x,torch.sum(pred))

        mean_centroid = torch.cat((mean_x,mean_y), 1)

        return torch.mean(torch.norm(torch.add(future_centroid,-1, mean_centroid),2,1))



class DistancePlusIoU(nn.Module):

    def __init__(self, device):
            super(DistancePlusIoU,self).__init__()
            self.device = device
            self.iou = IoULoss(device)
            self.distance_loss = DistanceLoss(device)

    def forward(self, pred,label, future_centroid):

        iou_loss = self.iou(pred,label)
        dist_loss = self.distance_loss(pred, future_centroid)

        return torch.add(iou_loss,dist_loss)
