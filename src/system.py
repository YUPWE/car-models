import json
import os

import numpy as np
import torch

from .data import get_transforms
from .model import build_resnet18, get_device

CONFIG_PATH = "results/system_config.json"
WEIGHTS_PATH = "weights/resnet18_finetuned.pt"
NAMES_PATH = "results/model_names.json"


def mix(resnet_probs, clip_probs, w):
    return w * resnet_probs + (1 - w) * clip_probs


def decide(resnet_probs, clip_probs, p_oos, w, conf_threshold, oos_threshold):
    mixed = mix(resnet_probs, clip_probs, w)
    conf = mixed.max(1)
    decisions = np.where(p_oos >= oos_threshold, "not a car",
                         np.where(conf >= conf_threshold, "auto-fill", "ask seller"))
    return decisions.tolist(), mixed


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def save_config(**values):
    cfg = load_config() if os.path.exists(CONFIG_PATH) else {}
    cfg.update(values)
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)
    return cfg


class CarRecognizer:
    def __init__(self, weights_path=WEIGHTS_PATH, config=None):
        from .clip_gate import ClipZeroShot
        self.device = get_device()
        self.cfg = config or load_config()
        self.names = json.load(open(NAMES_PATH))
        self.resnet = build_resnet18(freeze_backbone=False, pretrained=False, num_classes=len(self.names))
        self.resnet.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.resnet.to(self.device).eval()
        self.clip = ClipZeroShot(self.names, device=self.device)
        self.tf = get_transforms(train=False)

    @torch.no_grad()
    def predict(self, image, k=5):
        x = self.tf(image.convert("RGB")).unsqueeze(0).to(self.device)
        r = torch.softmax(self.resnet(x), 1).cpu().numpy()
        c, oos = self.clip([image])
        d, mixed = decide(r, c, oos, self.cfg["w"], self.cfg["conf_threshold"], self.cfg["oos_threshold"])
        top = np.argsort(-mixed[0])[:k]
        return {"decision": d[0], "top": [(self.names[i], float(mixed[0, i])) for i in top],
                "p_not_car": float(oos[0])}
