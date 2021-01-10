from .dataset import HubDataset
import albumentations as A
import torch.utils.data as D


def make_dataloader(cfg, tiff_ids, train_idx, val_idx):
    trfm = A.Compose([
        A.Resize(cfg.INPUT.NEW_SIZE, cfg.INPUT.NEW_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),

        A.OneOf([
            A.RandomBrightnessContrast(),
            A.RandomGamma(),
            # A.RandomBrightness(),
            A.ColorJitter(brightness=0.07, contrast=0.07,
                          saturation=0.1, hue=0.1, always_apply=False, p=0.3),
        ], p=0.3),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.GridDistortion(),
            A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
        ], p=0.3),
        A.ShiftScaleRotate(),

    ])

    val_trfm = A.Compose([
        A.Resize(cfg.INPUT.NEW_SIZE, cfg.INPUT.NEW_SIZE),
    ])
    print(tiff_ids[val_idx][0])

    if tiff_ids[val_idx][0] != "e79de561c":
        return None, None
    train_ds = HubDataset(cfg, cfg.DATASETS.DATA_PATH, tiff_ids[train_idx], window=cfg.DATASETS.WINDOW, new_size=cfg.INPUT.NEW_SIZE, overlap=cfg.DATASETS.MIN_OVERLAP,
                          threshold=100, transform=trfm)
    valid_ds = HubDataset(cfg, cfg.DATASETS.DATA_PATH, tiff_ids[val_idx], window=cfg.DATASETS.WINDOW, new_size=cfg.INPUT.NEW_SIZE, overlap=cfg.DATASETS.MIN_OVERLAP,
                          threshold=100, transform=val_trfm, isvalid=True)

    print(len(train_ds), len(valid_ds))

    # define training and validation data loaders
    train_loader = D.DataLoader(
        train_ds, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=True, num_workers=12)

    val_loader = D.DataLoader(
        valid_ds, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=False, num_workers=12)

    return train_loader, val_loader