# src/datasets_hf.py
from typing import Callable, Optional
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image

class HFImageMaskDataset(Dataset):
    """
    Expects each example to have keys:
    - 'img'  : a PIL-serializable image, path, or bytes (HF handles decoding)
    - 'mask' : same; binary or {0,1} int; will be converted to (1,H,W) float
    - 'class_name': str (used to filter by part type)
    Other keys are ignored here.
    """
    def __init__(
        self,
        hf_name_or_path: str,
        split: str,
        image_key: str = "img",
        mask_key: str = "mask",
        class_name: Optional[str] = None,
        class_key: str = "class_name",
        transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None
    ):
        self.ds = load_dataset(hf_name_or_path, split=split)
        
        # Filter by class_name if provided
        if class_name is not None:
            class_name = class_name.lower().replace(" ", "_") # normalize
            self.ds = self.ds.filter(lambda x: x[class_key] == class_name)
            print(f"Filtered dataset to {len(self.ds)} examples with class_name='{class_name}'")
        
        self.image_key = image_key
        self.mask_key = mask_key
        self.transform = transform
        self.mask_transform = mask_transform or transform  # keep sizes aligned

    def __len__(self):
        return len(self.ds)

    def _to_pil(self, v):
        if isinstance(v, Image.Image): return v
        # HF may give numpy arrays or bytes; PIL can open both
        return Image.fromarray(v) if hasattr(v, "dtype") else Image.open(v)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        img = self._to_pil(ex[self.image_key])
        msk = self._to_pil(ex[self.mask_key])  # single-object binary mask per row

        if self.transform:
            img = self.transform(img)
        else:
            import torchvision.transforms as T
            img = T.ToTensor()(img)

        if self.mask_transform:
            msk = self.mask_transform(msk)
        else:
            import torchvision.transforms as T
            msk = T.ToTensor()(msk)

        # Force mask shape = (1,H,W) and range {0,1}
        if msk.ndim == 3 and msk.shape[0] != 1:
            msk = msk[:1]
        msk = (msk > 0.5).float()

        return img, msk
