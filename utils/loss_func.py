import torch
import torch.nn as nn
import torch.nn.functional as F

def dice_loss(target, predictive, ep=1e-8):
    """dice loss for autogress.

    From paper 'V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation'. In fact, dice loss is equal to the f1 score in terms of results(both are 2TP/(2TP+FP+FN)). Referd by the article 'https://zhuanlan.zhihu.com/p/269592183'
    来自于论文'V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation', dice loss结果上等同于f1-score, 为2TP/(2TP+FP+FN).可以参考文章'https://zhuanlan.zhihu.com/p/269592183'

    Args:
      target:
        forecast traget, the groudtruth.
        预测目标
      predictive:
        forecast result
        预测结果
      ep:
        a small number, used to prevent division by zero.
        小参数,用于防止除0
    
    Returns:
      loss
    """
    intersection = 2 * torch.sum(predictive * target) + ep
    union = torch.sum(predictive) + torch.sum(target) + ep
    loss = 1 - intersection / union
    return loss



class Tripletloss(nn.Module):
    """Triplet Loss 

    From paper 'FaceNet: A Unified Embedding for Face Recognition and Clustering
'

    Attributes:


    """
    def __init__(self, margin=None):
        super(Tripletloss, self).__init__()
        self.margin = margin
        if self.margin is None:
            self.Loss = nn.SoftMarginLoss()
        else:
            self.Loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, pos, neg):
        if self.margin is None:
            num_samples = anchor.shape[0]
            y = torch.ones((num_samples, 1)).view(-1)
            if anchor.is_cuda: y = y.cuda()
            ap_dist = torch.norm(anchor-pos, 2, dim=1).view(-1)
            an_dist = torch.norm(anchor-neg, 2, dim=1).view(-1)
            loss = self.Loss(an_dist - ap_dist, y)
        else:
            loss = self.Loss(anchor, pos, neg)

        return loss 





def focal_loss():
    """

    """