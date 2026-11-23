import io
import os
import re
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from PIL import Image

HF = "https://huggingface.co/datasets/tanganke/stanford_cars/resolve/main"
PARQUETS = ["data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet",
            "data/test-00000-of-00002.parquet", "data/test-00001-of-00002.parquet"]
DATA_DIR = os.environ.get("CARS_DATA_DIR", "data/raw")
IMG_DIR = os.path.join(DATA_DIR, "images")
META = os.path.join(DATA_DIR, "meta.csv")
CACHE_ZIP = "data/cars_resized.zip"
SPLIT_DIR = "splits"
SHORT_SIDE = 256
TWO_WORD_MAKES = ("AM General", "Aston Martin", "Land Rover")


def class_names():
    text = urllib.request.urlopen(f"{HF}/README.md").read().decode()
    names = re.findall(r"'(\d+)': (.+)", text)
    return [n.strip() for _, n in names][:196]


def make_of(name):
    for m in TWO_WORD_MAKES:
        if name.startswith(m):
            return m
    return name.split()[0]


def download():
    if os.path.exists(META):
        return
    if os.path.exists(CACHE_ZIP):
        print("Unzipping cached images ...")
        with zipfile.ZipFile(CACHE_ZIP) as z:
            z.extractall(DATA_DIR)
        return
    os.makedirs(IMG_DIR, exist_ok=True)
    names = class_names()
    rows = []
    for part in PARQUETS:
        local = os.path.join(DATA_DIR, os.path.basename(part))
        print("Downloading", part, "...")
        urllib.request.urlretrieve(f"{HF}/{part}", local)
        df = pd.read_parquet(local)
        tag = "tr" if "train" in part else "te"
        shard = part.split("-")[1]
        for i, (img, label) in enumerate(zip(df["image"], df["label"])):
            im = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
            w, h = im.size
            scale = SHORT_SIDE / min(w, h)
            if scale < 1:
                im = im.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
            name = f"{tag}{shard}_{i:05d}.jpg"
            im.save(os.path.join(IMG_DIR, name), quality=90)
            rows.append({"path": name, "model_id": int(label), "model": names[label],
                         "orig_w": w, "orig_h": h})
        os.remove(local)
        del df
    meta = pd.DataFrame(rows)
    meta["make"] = meta["model"].map(make_of)
    meta["year"] = meta["model"].str.split().str[-1].astype(int)
    meta.to_csv(META, index=False)
    print("Saving a resized copy to", CACHE_ZIP, "...")
    os.makedirs(os.path.dirname(CACHE_ZIP), exist_ok=True)
    with zipfile.ZipFile(CACHE_ZIP, "w", zipfile.ZIP_STORED) as z:
        z.write(META, "meta.csv")
        for name in meta["path"]:
            z.write(os.path.join(IMG_DIR, name), f"images/{name}")


def load_meta():
    return pd.read_csv(META)


def model_names(meta=None):
    meta = load_meta() if meta is None else meta
    return meta.drop_duplicates("model_id").sort_values("model_id").model.tolist()


def perceptual_hashes(paths):
    import imagehash
    out = np.empty(len(paths), dtype=np.uint64)
    for k, p in enumerate(paths):
        bits = imagehash.phash(Image.open(os.path.join(IMG_DIR, p))).hash.flatten()
        out[k] = np.packbits(bits).view(">u8")[0]
    return out


def near_duplicate_pairs(hashes, max_dist=4, chunk=2000):
    pairs = []
    for s in range(0, len(hashes), chunk):
        dist = np.bitwise_count(hashes[s:s + chunk, None] ^ hashes[None, :])
        ii, jj = np.where(dist <= max_dist)
        ii = ii + s
        keep = ii < jj
        pairs += list(zip(ii[keep], jj[keep]))
    return pairs


def duplicate_groups(n, pairs):
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in pairs:
        parent[find(i)] = find(j)
    return [find(i) for i in range(n)]


def make_splits(df, groups, seed=42):
    from sklearn.model_selection import StratifiedGroupKFold
    df = df.copy()
    df["dup_group"] = groups
    outer = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    rest_idx, test_idx = next(outer.split(df, df["model_id"], df["dup_group"]))
    rest = df.iloc[rest_idx]
    inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)
    train_idx, val_idx = next(inner.split(rest, rest["model_id"], rest["dup_group"]))
    df["split"] = ""
    df.iloc[test_idx, df.columns.get_loc("split")] = "test"
    df.loc[rest.index[train_idx], "split"] = "train"
    df.loc[rest.index[val_idx], "split"] = "val"
    return df


def save_splits(df):
    os.makedirs(SPLIT_DIR, exist_ok=True)
    for s in ["train", "val", "test"]:
        df[df["split"] == s][["path", "model_id", "model", "make", "year"]].to_csv(f"{SPLIT_DIR}/{s}.csv", index=False)


def load_split(name):
    path = f"{SPLIT_DIR}/{name}.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found - run notebooks/01_data.ipynb first.")
    return pd.read_csv(path)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(train):
    from torchvision import transforms as T
    if train:
        return T.Compose([
            T.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.6, 1.8)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.3, 0.3, 0.3),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class CarDataset:
    def __init__(self, df, transform):
        self.paths = df["path"].tolist()
        self.labels = df["model_id"].astype(int).tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(os.path.join(IMG_DIR, self.paths[i])).convert("RGB")
        return self.transform(img), self.labels[i]
