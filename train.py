import random
import os
import time
import numpy as np
import torch
from config import cfg
from datasets.make_dataloader import make_dataloader
from losses.make_loss import make_loss
from utils.make_optimizer import make_optimizer
from utils.lr_scheduler import WarmupMultiStepLR
import segmentation_models_pytorch as smp
from sklearn.model_selection import KFold
import argparse

def np_dice_score(probability, mask):
    p = probability.reshape(-1)
    t = mask.reshape(-1)

    p = p > 0.5
    t = t > 0.5
    uion = p.sum() + t.sum()

    overlap = (p * t).sum()
    dice = 2 * overlap / (uion + 0.001)
    return dice


def do_validation(cfg, model, val_loader):
    val_probability, val_mask = [], []
    model.eval()
    with torch.no_grad():
        for image, target in val_loader:
            image, target = image.to(cfg.MODEL.DEVICE), target.float().to(cfg.MODEL.DEVICE)
            output = model(image)

            output_ny = output.data.cpu().numpy()
            target_np = target.data.cpu().numpy()

            val_probability.append(output_ny)
            val_mask.append(target_np)

    val_probability = np.concatenate(val_probability)
    val_mask = np.concatenate(val_mask)

    return np_dice_score(val_probability, val_mask)

def set_seeds(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
def get_model(cfg):
    ENCODER = cfg.MODEL.NAME
    ENCODER_WEIGHTS = cfg.MODEL.PRETRAIN_CHOICE
    CLASSES = ['obj']
    ACTIVATION = 'sigmoid'  # could be None for logits or 'softmax2d' for multicalss segmentation

    # create segmentation model with pretrained encoder
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=len(CLASSES),
        activation=ACTIVATION,
    )

    return model

def do_train(cfg):
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    import glob
    tiff_ids = np.array([x.split('/')[-1][:-5] for x in glob.glob('hubmap-kidney-segmentation/train/*.tiff')])
    skf = KFold(n_splits=8)
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(tiff_ids, tiff_ids)):
        train_loader, val_loader = make_dataloader(cfg, tiff_ids, train_idx, val_idx)
        if not train_loader and not val_loader:
            continue
        model = get_model(cfg)
        model.to(cfg.MODEL.DEVICE)
        optimizer = make_optimizer(cfg, model)
        scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR, cfg.SOLVER.WARMUP_EPOCHS, cfg.SOLVER.WARMUP_METHOD)

        ### Table for results
        header = r'''
                Train | Valid
        Epoch |  Loss |  Loss | Time, m
        '''
        #          Epoch         metrics            time
        raw_line = '{:6d}' + '\u2502{:7.3f}'*2 + '\u2502{:6.2f}'
        print(header)

        for epoch in range(1, cfg.SOLVER.MAX_EPOCHS+1):
            losses = []
            start_time = time.time()
            model.train()
            for image, target in train_loader:
                image, target = image.float().to(cfg.MODEL.DEVICE), target.float().to(cfg.MODEL.DEVICE)
                optimizer.zero_grad()
                output = model(image)
                loss = make_loss(cfg, output, target)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            val_dice = do_validation(cfg, model, val_loader)
            print(raw_line.format(epoch, np.array(losses).mean(), val_dice,
                                  (time.time() - start_time) / 60 ** 1))
            scheduler.step()
            outputpath = cfg.OUTPUT_DIR+'/fold'+str(fold_idx)
            if not os.path.exists(outputpath):
                os.makedirs(outputpath)

            torch.save(model.state_dict(), outputpath+'/epoch'+str(epoch)+'.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configs/senext50_unet.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    print(cfg)
    do_train(cfg)