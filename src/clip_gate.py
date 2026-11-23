import numpy as np
import torch
from PIL import Image

CLIP_NAME = "openai/clip-vit-base-patch16"

OUT_OF_SCOPE_PROMPTS = [
    "a photo of a person", "a photo of a face", "a photo of an animal", "a photo of a building",
    "a photo of an empty road", "a photo of an empty parking space", "a landscape photo",
    "a photo of a motorcycle", "a photo of a bicycle", "a photo of a bus", "a photo of a semi truck",
    "a photo of a page of text", "a microscope image", "a photo of the night sky",
]


class ClipZeroShot:
    def __init__(self, class_names, device=None, model_name=CLIP_NAME):
        from transformers import CLIPModel, CLIPProcessor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.n = len(class_names)
        prompts = [f"a photo of a {n}, a type of car." for n in class_names] + OUT_OF_SCOPE_PROMPTS
        self.text_emb = torch.tensor(self.text_embeddings(prompts), device=self.device)

    @staticmethod
    def _as_tensor(out):
        return out if torch.is_tensor(out) else out.pooler_output

    @torch.no_grad()
    def text_embeddings(self, texts):
        tok = self.processor(text=list(texts), return_tensors="pt", padding=True).to(self.device)
        t = self._as_tensor(self.model.get_text_features(**tok))
        return (t / t.norm(dim=-1, keepdim=True)).cpu().numpy()

    @torch.no_grad()
    def __call__(self, images, batch_size=32):
        class_probs, p_oos = [], []
        for i in range(0, len(images), batch_size):
            batch = [im.convert("RGB") for im in images[i:i + batch_size]]
            pix = self.processor(images=batch, return_tensors="pt").to(self.device)
            e = self._as_tensor(self.model.get_image_features(**pix))
            e = e / e.norm(dim=-1, keepdim=True)
            p = (self.model.logit_scale.exp() * e @ self.text_emb.T).softmax(dim=-1).cpu().numpy()
            cars = p[:, :self.n]
            class_probs.append(cars / cars.sum(1, keepdims=True))
            p_oos.append(p[:, self.n:].sum(1))
        return np.concatenate(class_probs), np.concatenate(p_oos)


def load_pil(paths):
    return [Image.open(p).convert("RGB") for p in paths]
