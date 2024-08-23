from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
from torch import nn


class MultiTask_DCCE_CE_loss(nn.Module):
    """
        loss function for multitask training (segmentation and classification), mainly for the multitask_unet
    """
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", square_dice=False, weight_ce=1, weight_dice=1, lambda_seg=1, lambda_class=0.3, log_dice=False, ignore_label=None):
        super(MultiTask_DCCE_CE_loss, self).__init__()

        self.seg_loss = DC_and_CE_loss(
            soft_dice_kwargs, 
            ce_kwargs, 
            aggregate, 
            square_dice, 
            weight_ce, 
            weight_dice,
            log_dice, 
            ignore_label)

        self.multiclass_class_loss = nn.CrossEntropyLoss()

        self.lambda_seg = lambda_seg
        self.lambda_class = lambda_class

    def forward(self, outputs_seg, outputs_class, targets_seg, targets_class):
        lseg = self.seg_loss(outputs_seg, targets_seg)
        lclass = self.multiclass_class_loss(outputs_class, targets_class)

        # Combine the segmentation and classification losses
        total_loss = self.lambda_seg * lseg + self.lambda_class * lclass

        return total_loss