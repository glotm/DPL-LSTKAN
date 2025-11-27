import torch

def RMSE(pred,label):
    return torch.sqrt(torch.nn.functional.mse_loss(pred,label))


def MSE(pred,label):
    return torch.nn.functional.mse_loss(pred,label)

def MAE(pred,label):
    return torch.nn.functional.l1_loss(pred,label)

def NSE(pred,label):
    return 1.0-torch.sum((pred-label)**2)/torch.sum((label-label.mean())**2)

def R(pred,label):
    return torch.sum((pred-pred.mean())*(label-label.mean())) / torch.sqrt( torch.sum((label-label.mean())**2)) / torch.sqrt(torch.sum((pred-pred.mean())**2))


def KGE(pred,label):
    return 1.0-torch.sqrt((R(pred,label)-1)**2 + (pred.mean()/label.mean() - 1)**2 + (pred.std()/label.std() - 1)**2)
