from torch import einsum, nn
from torch.nn.functional import relu

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
