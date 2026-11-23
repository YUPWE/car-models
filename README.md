# Which car is this? Car model recognition with ResNet-18 + CLIP

Recognises the **exact car model** (make, model and year, e.g. *BMW X5 SUV 2007*) from one photo, out of **196 models**. When it is confident it fills the answer in automatically. When it isn't, it shows the **top 3** for a person to pick from, and CLIP answers **"not a car"** for photos that aren't cars.

Course project for *Computer Vision for Developers* (SDAIA Academy, Nov 2026). Team: **Mosaad Alrubaish** and **Majed Almutairi**

---

## 1. Use case card

| | |
|---|---|
| **User** | A seller on a used-car marketplace app (like Haraj or Syarah) creating a new listing |
| **Input** | One photo of the car, taken by the seller |
| **Output** | Make, model and year filled into the listing automatically, **or** "pick your car" with the top 3 suggestions, **or** "not a car" |
| **Cost of a false positive** (a *wrong* model filled in automatically) | **High.** The listing shows the wrong car, so buyers searching for the real model never find it, buyers who do find it are misled, and the platform loses trust. |
| **Cost of a false negative** (not auto-filled → seller picks from the top 3) | **Low.** The seller taps once more, a few seconds of effort. |
| **Success metric and target** | On the sealed test set: **≥ 95 % accuracy on auto-filled listings**, while auto-filling **as many photos as possible**. Also: the right model is **inside the top 3** shown to the seller at least 90 % of the time. |

Because a wrong auto-fill costs much more than asking the seller, the confidence threshold is chosen on purpose for 95 % accuracy (see §5).

## 2. Data

- **Source:** [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) (Krause et al., 2013), downloaded as parquet files from the Hugging Face mirror [`tanganke/stanford_cars`](https://huggingface.co/datasets/tanganke/stanford_cars) (~2 GB). **Licence: non-commercial research use**, which fits this course project. Details in [`data/README.md`](data/README.md).
- **Size and classes:** 16,185 web photos of **196 car models** from 49 makes, model years 1991–2012, with 48–136 photos per model (median 84).
- **Split method:** about 70 / 15 / 15 train / val / test, **stratified by model**, so every model appears in all three. Near-duplicate photos (perceptual hash distance ≤ 4) are grouped so the same photo **never appears in two splits**. Notebook 01 checks both. The split lists are committed in `splits/`.
- **Cleaning (notebook 01):** dropped thumbnails (shorter side < 64 px) and every photo that appears twice **under two different model names**, which is a label error.
- **Known problems:**
  - **Label errors:** the same photo is filed under two different models (often the same car under two model years). See `results/label_conflicts.png`.
  - **Very fine differences:** many classes are the same car in different years (e.g. *Audi S4 Sedan 2007* vs *2012*) or different body styles of one model (coupe vs convertible vs sedan).
  - **Web photos** (ads, reviews, dealer lots): many contain dealer logos or text, unlike a seller's phone photo.
  - Only cars up to **2012**.

## 3. Model

_TODO after notebook 02._

- **Baseline:** ImageNet ResNet-18 with a **frozen backbone**; only a new 196-class head is trained.
- **Trained model:** the same ResNet-18 with **all layers fine-tuned**. This is the *only* difference from the baseline: same data, augmentation (random crop, horizontal flip, colour jitter), cross-entropy, AdamW with cosine schedule, 10 epochs, seed 42. We keep the epoch with the best **validation macro-F1**.
- **Key settings:** input 224×224 · batch 64 · lr 1e-3 (frozen) / 5e-4 (fine-tuned) · weight decay 1e-4
- **Curves:** `results/curves.png`

## 4. Foundation component: CLIP (ViT-B/16)

CLIP has learned from 400M image–text pairs from the web, and car photos with their names are very common there. It is built into the decision rule (`src/system.py`), not run as a separate demo:

1. **Second opinion:** CLIP compares the photo with the text *"a photo of a {model name}, a type of car."* for all 196 names, with no training at all. Its probabilities are mixed with the ResNet's: `p = w · ResNet + (1 − w) · CLIP`, with `w` chosen on validation by log-loss.
2. **Not-a-car gate:** CLIP also compares the photo with non-car prompts ("a person", "an empty parking space", "a motorcycle", …). If p(not a car) ≥ threshold, the answer is **not a car**. The threshold is the lowest value that rejects ≤ 1 % of validation car photos.

**Why it's needed:** a 196-class ResNet *must* pick one of its 196 cars, even for a selfie or a motorbike. CLIP recognises those without any training data. And it brings knowledge from the web about car names that our 11,000 training photos alone don't have.

_TODO after notebook 03._

## 5. Evaluation (sealed test set)

_TODO after notebook 04._

## 6. Failure analysis

_TODO after notebook 04._

## 7. Limitations

- **Only 196 models, all 2012 or older.** Any newer car, or a model not in the list, gets the closest wrong answer, or at best a low confidence. A real marketplace would need thousands of current models and regular retraining.
- **Web photos:** clean dealer and review photos, often with logos. Sellers' phone photos (night, rain, part of the car, interior shots) would score lower. Production would need to fine-tune on real listing photos.
- **Year differences** between generations that look the same can't be told apart from one photo; the seller should always be able to edit the year.
- One car per photo is assumed. Production would add a car detector first.

## 8. How to run

From a fresh Google Colab (GPU runtime: *Runtime → Change runtime type → T4 GPU*):

1. Open `notebooks/01_data.ipynb` in Colab (*File → Open notebook → GitHub*, paste this repo's URL).
2. Run the first cell. It mounts Google Drive, clones the repo to `MyDrive/car-models` (shared by all four notebooks), installs the one missing library (`imagehash`; everything else comes with Colab, and notebook 02 prints the exact versions), and keeps the images on Colab's fast local disk.
3. Run the notebooks in order:
   - `01_data` downloads ~2 GB from Hugging Face (3–5 min) and saves a resized copy to Drive, so later notebooks start in seconds.
   - Then `02_train` (~30 min on a T4) → `03_foundation` → `04_evaluate`.
4. Results appear in `results/`, and the models are saved in `weights/`.

Run locally or on Hugging Face Spaces: `pip install -r requirements.txt` (pinned versions), then `python app.py`.

**Deploy (Hugging Face Spaces):** create a Space with the **Gradio** SDK and upload `app.py`, `requirements.txt`, `src/`, `weights/resnet18_finetuned.pt`, `results/system_config.json` and `results/model_names.json`.

## Repo layout

```
notebooks/   01_data · 02_train · 03_foundation · 04_evaluate
src/         data.py (download, dedup, split) · model.py (ResNet-18 + training loop)
             clip_gate.py (CLIP zero-shot + not-a-car gate) · system.py (full decision rule)
splits/      train/val/test CSVs (created by 01)
results/     curves, coverage curve, confusion matrix, failures/
weights/     trained models (created by 02; not committed, ~45 MB each)
app.py       Gradio demo
```

Dataset credit: J. Krause, M. Stark, J. Deng, L. Fei-Fei. *3D Object Representations for Fine-Grained Categorization*, 4th IEEE Workshop on 3D Representation and Recognition (3dRR-13), 2013.
