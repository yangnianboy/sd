import gc
import torch

def im_to_np(im):
    im = (im / 2 + 0.5).clamp(0, 1)
    im = im.detach().cpu().permute(1, 2, 0).numpy()
    im = (im * 255).round().astype("uint8")
    return im

def flush():
    gc.collect()
    torch.cuda.empty_cache()

