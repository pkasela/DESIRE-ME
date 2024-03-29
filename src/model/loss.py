from torch import einsum, nn
from torch.nn.functional import relu
import torch

class MultipleRankingLoss(nn.Module):
    """
    Triplet Margin Loss function.
    """

    def __init__(self, device):
        super(MultipleRankingLoss, self).__init__()
        self.CELoss = nn.CrossEntropyLoss()
        self.BCELoss = nn.BCEWithLogitsLoss()
        
        self.device = device
    def forward(
        self,
        anchors_classes,
        anchors_pred,
        anchors,
        positives
    ):
        pw_similarity = torch.mm(anchors, positives.T)
        labels = torch.tensor([x for x in range(anchors.shape[0])], device=self.device)
        
        # import ipdb
        # ipdb.set_trace()
        cls_loss = self.BCELoss(anchors_pred, anchors_classes)
        pw_loss = self.CELoss(pw_similarity, labels)
        
        return pw_loss, cls_loss, .5 * pw_loss + .5 * cls_loss
    
    def val_forward(
        self,
        anchors_classes,
        anchors_pred,
        anchors,
        positives
    ):
        pw_similarity = torch.mm(anchors, positives.T)
        labels = torch.tensor([x for x in range(anchors.shape[0])], device=self.device)
        
        cls_loss = self.BCELoss(anchors_pred, anchors_classes)
        pw_loss = self.CELoss(pw_similarity, labels)
        
        return pw_loss, cls_loss, .5 * pw_loss + .5 * cls_loss, (pw_similarity.argmax(dim=1, keepdim=True).squeeze() == labels)


class MultipleRankingLossBiEncoder(nn.Module):
    """
    Triplet Margin Loss function.
    """

    def __init__(self, device, temperature=1):
        super(MultipleRankingLossBiEncoder, self).__init__()
        self.CELoss = nn.CrossEntropyLoss()
        self.temperature = temperature
        
        self.device = device
    def forward(
        self,
        anchors,
        positives
    ):
        pw_similarity = torch.mm(anchors / self.temperature, positives.T)
        labels = torch.tensor([x for x in range(anchors.shape[0])], device=self.device)
        
        pw_loss = self.CELoss(pw_similarity, labels)
        
        return pw_loss
    
    def val_forward(
        self,
        anchors,
        positives
    ):
        pw_similarity = torch.mm(anchors / self.temperature, positives.T)
        labels = torch.tensor([x for x in range(anchors.shape[0])], device=self.device)
        
        pw_loss = self.CELoss(pw_similarity, labels)
        
        return pw_loss, (pw_similarity.argmax(dim=1, keepdim=True).squeeze() == labels)


