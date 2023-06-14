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
        
        cls_loss = self.CELoss(anchors_pred, anchors_classes)
        pw_loss = self.CELoss(pw_similarity, labels)
        
        return pw_loss, cls_loss, .9 * pw_loss + .1 * cls_loss
    
    def val_forward(
        self,
        anchors_classes,
        anchors_pred,
        anchors,
        positives
    ):
        pw_similarity = torch.mm(anchors, positives.T)
        labels = torch.tensor([x for x in range(anchors.shape[0])], device=self.device)
        
        cls_loss = self.CELoss(anchors_pred, anchors_classes)
        pw_loss = self.CELoss(pw_similarity, labels)
        
        return pw_loss, cls_loss, .9 * pw_loss + .1 * cls_loss, (pw_similarity.argmax(dim=1, keepdim=True).squeeze() == labels)
    
class TripletMarginLoss(nn.Module):
    """
    Triplet Margin Loss function.
    """

    def __init__(self, margin=1.0):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin

    def forward(
        self,
        anchors,
        positives,
        negatives,
    ):
        positive_embedding_scores = einsum("xz,xz->x", anchors, positives)
        negative_embedding_scores = einsum("xz,xz->x", anchors, negatives)

        loss = relu(
            self.margin - positive_embedding_scores + negative_embedding_scores
        ).mean()

        return loss
    
class TripletMarginClassLoss(nn.Module):
    """
    Triplet Margin Loss function with query classifier.
    """

    def __init__(self, margin=1.0, alpha=0.5):
        super(TripletMarginClassLoss, self).__init__()
        
        self.alpha = alpha
        
        self.TripletLoss = TripletMarginLoss(margin)
        self.CELoss = nn.CrossEntropyLoss()
        
    def forward(
        self,
        anchors_classes,
        anchors_pred,
        anchors,
        positives,
        negatives
    ):
        triple_loss = self.TripletLoss(anchors, positives, negatives)
        ce_loss = self.CELoss(anchors_pred, anchors_classes)
        
        return self.alpha * triple_loss + (1 - self.alpha) * ce_loss
    
    def val_forward(
        self,
        anchors_classes,
        anchors_pred,
        anchors,
        positives,
        negatives
    ):
        triple_loss = self.TripletLoss(anchors, positives, negatives)
        ce_loss = self.CELoss(anchors_pred, anchors_classes)
        
        return triple_loss, ce_loss, self.alpha * triple_loss + (1 - self.alpha) * ce_loss
