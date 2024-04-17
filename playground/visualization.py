import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
torch.cuda.is_available()
from timm import create_model
from utils.utils import make_visualization
import models.deit

device = "cuda"
model = create_model(
        "deit_small", pretrained=True, num_classes=1000, drop_rate=0.0, drop_path_rate=0.1, drop_block_rate=None, img_size=224, token_nums=[1, 0], embed_type=[0,0,0],
        use_mctf=True, activate_layer=[3,6,9], task_type=[1, 0, 0], mctf_type=[0.35, 0.0, 1, 1, 1, 1, 1, 1, 1, 1, 0])
model = model.eval().to(device)