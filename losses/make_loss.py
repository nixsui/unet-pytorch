from .lovasz_losses import lovasz_softmax
from .focal_loss import FocalLoss
import segmentation_models_pytorch as smp

diceloss = smp.utils.losses.DiceLoss()
bceloss = smp.utils.losses.BCELoss()

focalloss = FocalLoss()


def label_smoothing(inputs, epsilon=0.1):
    # K = inputs.get_shape().as_list()[-1]    # number of channels
    return ((1-epsilon) * inputs) + (epsilon / 2.0)

def make_loss(cfg, y_pred, y_true):
    if cfg.MODEL.LOSS_TYPE == 'lovasz':
        loss_lovasz = lovasz_softmax(y_pred, y_true, classes=[1], ignore=255)
        return loss_lovasz

    elif cfg.MODEL.LOSS_TYPE == 'focalloss':
        if cfg.MODEL.IF_LABELSMOOTH == 'on':
            y_true = label_smoothing(y_true)
            loss_focal = focalloss(y_pred, y_true)

        else:
            loss_focal = focalloss(y_pred, y_true)
        return loss_focal

    elif cfg.MODEL.LOSS_TYPE == 'bceloss':
        if cfg.MODEL.IF_LABELSMOOTH == 'on':
            y_true = label_smoothing(y_true)
            loss_bce = bceloss(y_pred, y_true)

        else:
            loss_bce = bceloss(y_pred, y_true)
        return loss_bce

    elif cfg.MODEL.LOSS_TYPE == 'diceloss':
        loss_bce = diceloss(y_pred, y_true)
        return loss_bce