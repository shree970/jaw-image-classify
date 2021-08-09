import albumentations as A
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2


def apply_transforms(mean, std):
    """
    Image augmentations for train and test set.
    """
    train_transforms = A.Compose(
        [
            A.Sequential(
                [
                    A.PadIfNeeded(min_height=40, min_width=40, always_apply=True),
                    A.RandomCrop(width=32, height=32, p=1),
                ],
                p=0.5,
            ),
            A.Rotate(limit=5, p=0.2),
            A.CoarseDropout(
                max_holes=1,
                max_height=16,
                max_width=16,
                min_holes=1,
                min_height=16,
                min_width=16,
                fill_value=tuple((x * 255.0 for x in mean)),
                p=0.2,
            ),
            A.Normalize(mean=mean, std=std, always_apply=True),
            ToTensorV2(),
        ]
    )

    test_transforms = A.Compose(
        [
            A.Normalize(mean=mean, std=std, always_apply=True),
            ToTensorV2(),
        ]
    )

    return (
        lambda img: train_transforms(image=np.array(img))["image"],
        lambda img: test_transforms(image=np.array(img))["image"],
    )


def apply_transforms_custom_resnet(mean, std):
    """
    Image augmentations for train and test set.
    """
    train_transforms = A.Compose(
        [
            A.Sequential(
                [
                    A.PadIfNeeded(min_height=40, min_width=40, always_apply=True),
                    A.RandomCrop(width=32, height=32, p=1),
                ],
                p=0.5,
            ),
            A.HorizontalFlip(p=0.2),
            A.CoarseDropout(
                max_holes=1,
                max_height=8,
                max_width=8,
                min_holes=1,
                min_height=8,
                min_width=8,
                fill_value=tuple((x * 255.0 for x in mean)),
                p=0.2,
            ),
            A.Normalize(mean=mean, std=std, always_apply=True),
            ToTensorV2(),
        ]
    )

    test_transforms = A.Compose(
        [
            A.Normalize(mean=mean, std=std, always_apply=True),
            ToTensorV2(),
        ]
    )

    return (
        lambda img: train_transforms(image=np.array(img))["image"],
        lambda img: test_transforms(image=np.array(img))["image"],
    )
