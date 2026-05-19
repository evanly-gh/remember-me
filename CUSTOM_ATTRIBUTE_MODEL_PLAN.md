# Plan: Training a Custom Facial Attribute Classifier
# (If you have a GPU and 50 hours to spare)

## Problem Statement

**Current issue:** CLIP's zero-shot attribute classifier has high false positive rates:
- Reports receding hairline when there isn't one
- Detects accessories (glasses, earrings, necklace) when absent
- Possibly innaccurate overall (unknown baseline)

**Root cause:** CLIP is trained on generic internet data (400M image-text pairs), not optimized for binary facial attributes. It doesn't learn from your specific use case.

**Solution:** Train a custom binary classifier on labeled facial images.

---

## High-Level Strategy

Replace CLIP's ~30 binary attribute outputs with a **single fine-tuned model** that predicts all ~30 attributes end-to-end.

```
OLD PIPELINE (CLIP):
Image → CLIP → 30 separate 2-way softmax scores → 30 boolean thresholds → Results

NEW PIPELINE (Custom Classifier):
Image → ResNet/ViT (pre-trained) → Fine-tune head → 30 sigmoid outputs → Results
```

**Why this is better:**
- ✅ Single forward pass (faster: ~0.5-1s vs CLIP's 4-5s)
- ✅ Learns from YOUR labeled data
- ✅ Can achieve >95% accuracy with proper data collection
- ✅ Continuous improvement as you collect more ground-truth labels

---

## Phase 1: Data Collection (Effort: HIGH, Time: 2-4 weeks)

### 1.1 Define Your Attribute Set

You currently have ~30 CLIP attributes. Decide which are **actually important** for your use case:

**Recommended priority tiers:**

**TIER 1 (Critical - high false positive rate currently)**
- wearing_glasses
- wearing_hat
- wearing_earrings
- wearing_necklace
- wearing_necktie
- heavy_makeup
- wearing_lipstick

**TIER 2 (Important - moderate accuracy)**
- has_beard
- mustache
- goatee
- sideburns
- has_bangs
- is_bald
- receding_hairline

**TIER 3 (Nice-to-have - harder to label reliably)**
- big_nose, pointy_nose, big_lips, narrow_eyes, arched_eyebrows, bushy_eyebrows
- attractive, young, pale_skin, rosy_cheeks, bags_under_eyes, chubby, double_chin, high_cheekbones, oval_face_celeba

**Recommendation:** Start with TIER 1 + TIER 2 (~20 attributes). Drop TIER 3 for now.

### 1.2 Collect Training Data

You need **200-500 labeled images per attribute** for robust training.

**Option A: Use CelebA Dataset (Easiest)**
- CelebA: 200K celebrity images, pre-labeled for 40 attributes
- Download from: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- Includes your attributes: bearing, makeup, accessories, etc.
- **Pros:** Free, large, diverse
- **Cons:** Only celebrities (may not match your user base)
- **Effort:** ~2 hours to download + organize
- **Quality:** ~90% accurate labels (human-annotated, but 15 years old)

**Option B: Collect Your Own Data (Most Accurate)**
1. **Gather base images:**
   - Use your existing Supabase photos (if you have >500)
   - Find face image datasets: LFWA, VGGFace2, Diverse Face DB
   - Ask friends/family for labeled photos (with consent)

2. **Label via web app:**
   - Build simple labeling UI (React component with checkboxes)
   - Display image → user checks which attributes are present → save to CSV
   - Yourself or 2-3 volunteers label all images
   - Estimate: 30 sec per image × 400 images = 3-4 hours solo

3. **Multiple-annotator validation:**
   - Have 2+ people label same images independently
   - Keep only labels with 80%+ agreement
   - Disagreements = re-label or exclude

**Option C: Hybrid (Recommended)**
1. Start with CelebA (baseline: 200K labeled images)
2. Fine-tune on your own photos (50-100 of yours)
3. Both → more robust model

**Recommended Approach:**
```
Phase 1a: Download CelebA (~200K images) — 2 hours
Phase 1b: Collect 50 of your own photos — 1 hour
Phase 1c: Label your 50 photos (3× annotators) — 2-3 hours
Total: ~6-7 hours, with 200K+ training images
```

### 1.3 Data Organization

Final structure:
```
data/
├── images/
│   ├── celeba_000001.jpg
│   ├── celeba_000002.jpg
│   ├── ...
│   ├── custom_001.jpg
│   ├── custom_002.jpg
│   └── ...
├── labels.csv  # Columns: image_name, wearing_glasses, has_beard, heavy_makeup, ...
└── train_val_test_split.csv  # Columns: image_name, split (train/val/test)
```

**Labels CSV format:**
```csv
image_name,wearing_glasses,has_beard,heavy_makeup,wearing_earrings,...
celeba_000001.jpg,0,1,0,0,...
celeba_000002.jpg,1,0,1,1,...
custom_001.jpg,0,0,0,0,...
```

**Train/Val/Test split (recommended):**
- Train: 70% (140K if using CelebA)
- Val: 15% (30K)
- Test: 15% (30K) — **kept secret until final evaluation**

---

## Phase 2: Model Architecture & Setup (Effort: MEDIUM, Time: 1-2 days)

### 2.1 Choose Your Base Model

**Option A: ResNet-50 (Balanced, Recommended)**
- Pre-trained on ImageNet
- ~50M parameters
- ~2 sec inference per image
- Mature ecosystem (PyTorch, TensorFlow)
- **Recommendation: START HERE**

**Option B: Vision Transformer (ViT-B, More Modern)**
- Pre-trained on ImageNet
- ~86M parameters
- ~1-2 sec inference
- Better for complex patterns
- Slightly overkill for binary attributes

**Option C: EfficientNet-B3 (Mobile-Friendly)**
- ~10M parameters
- ~0.5 sec inference
- Good for edge deployment
- Less research code available

**Decision: Use ResNet-50 for best community support.**

### 2.2 Training Architecture

```python
Input Image (3, 224, 224)
  ↓
ResNet-50 backbone (pre-trained on ImageNet, frozen layers 1-3)
  ↓
Global Average Pooling (2048,)
  ↓
Dropout (0.5)
  ↓
Dense layer (512)
  ↓
ReLU activation
  ↓
Dropout (0.3)
  ↓
Output layer (30,)  # One neuron per attribute
  ↓
Sigmoid activation  # Binary output per neuron
  ↓
~30 confidence scores (0.0-1.0) for each attribute
  ↓
Thresholds (0.5) → Boolean yes/no
```

**Key decisions:**
- **Freeze backbone layers 1-3:** Backbone already learned general features; only fine-tune top layer
- **Sigmoid output:** Each attribute is independent binary classification
- **Loss function:** Binary Cross-Entropy (BCE) loss
- **Optimizer:** Adam (learning rate 1e-4)

### 2.3 Training Hyperparameters

```python
learning_rate = 1e-4
batch_size = 32
epochs = 20  # Early stop after val loss plateaus
dropout = 0.5
optimizer = "Adam"
loss = "BinaryCrossentropy"
class_weights = None  # Set later if data imbalance detected
```

---

## Phase 3: Training Code (Effort: MEDIUM, Time: 1-2 days)

Create new file: `face-service/train_attribute_model.py`

**Required libraries:**
```bash
pip install torch torchvision pytorch-lightning albumentations scikit-learn wandb tensorboard
```

**Script outline:**

```python
# face-service/train_attribute_model.py

import torch
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torch.nn import BCEWithLogitsLoss
import torch.optim as optim
import pandas as pd
import numpy as np
from PIL import Image
import albumentations as A
from sklearn.model_selection import train_test_split
import wandb  # For experiment tracking

class FacialAttributeDataset(Dataset):
    """Load images + binary attribute labels"""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(image=np.array(img))['image']
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label
    
    def __len__(self):
        return len(self.image_paths)

class AttributeModel(torch.nn.Module):
    """ResNet-50 with custom head for ~30 binary attributes"""
    def __init__(self, num_attributes=30):
        super().__init__()
        # Load pre-trained ResNet-50
        resnet = models.resnet50(pretrained=True)
        
        # Freeze early layers (only fine-tune top)
        for param in list(resnet.parameters())[:-10]:
            param.requires_grad = False
        
        # Remove classification head
        self.backbone = torch.nn.Sequential(*list(resnet.children())[:-1])
        
        # Add custom head
        self.head = torch.nn.Sequential(
            torch.nn.Linear(2048, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, num_attributes),
            torch.nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x

def train_epoch(model, loader, optimizer, loss_fn, device):
    """One training epoch"""
    model.train()
    total_loss = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

def validate(model, loader, loss_fn, device):
    """Validation loop"""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
    
    return total_loss / len(loader)

def main():
    # Load labels CSV
    labels_df = pd.read_csv('data/labels.csv')
    splits_df = pd.read_csv('data/train_val_test_split.csv')
    
    # Merge and split
    data = labels_df.merge(splits_df, on='image_name')
    train_data = data[data['split'] == 'train']
    val_data = data[data['split'] == 'val']
    
    # Extract image paths and labels
    train_images = ['data/images/' + name for name in train_data['image_name']]
    train_labels = train_data.drop(['image_name', 'split'], axis=1).values
    
    val_images = ['data/images/' + name for name in val_data['image_name']]
    val_labels = val_data.drop(['image_name', 'split'], axis=1).values
    
    # Data augmentation for training
    train_transform = A.Compose([
        A.RandomBrightnessContrast(p=0.2),
        A.Rotate(limit=20, p=0.5),
        A.GaussNoise(p=0.2),
        A.HorizontalFlip(p=0.5),
        A.Normalize(),
        A.pytorch.ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Normalize(),
        A.pytorch.ToTensorV2()
    ])
    
    # DataLoaders
    train_dataset = FacialAttributeDataset(train_images, train_labels, train_transform)
    val_dataset = FacialAttributeDataset(val_images, val_labels, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AttributeModel(num_attributes=train_labels.shape[1])
    model.to(device)
    
    # Training
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.BCELoss()
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    wandb.init(project='hcp-attributes', name='resnet50-attributes')
    
    for epoch in range(20):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = validate(model, val_loader, loss_fn, device)
        
        print(f"Epoch {epoch+1}/20 — Train: {train_loss:.4f}, Val: {val_loss:.4f}")
        wandb.log({"train_loss": train_loss, "val_loss": val_loss})
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    print("Training complete!")
    wandb.finish()

if __name__ == '__main__':
    main()
```

---

## Phase 4: Integration into Face Service (Effort: LOW, Time: 1 day)

### 4.1 Replace CLIP with Custom Model

**File:** `face-service/analyzers/attribute_analyzer.py`

Replace the entire file:

```python
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

MODEL_PATH = 'models/custom_attribute_model.pth'
ATTRIBUTE_NAMES = [
    'wearing_glasses', 'has_beard', 'heavy_makeup', 'wearing_earrings', 
    'wearing_necklace', 'wearing_necktie', 'mustache', 'goatee', 
    'sideburns', 'has_bangs', 'is_bald', 'receding_hairline',
    # ... all 30 attributes in order
]

class CustomAttributeAnalyzer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _load_model(self):
        """Load custom attribute model"""
        try:
            resnet = models.resnet50(pretrained=False)
            # Remove classification head
            model = torch.nn.Sequential(*list(resnet.children())[:-1])
            
            # Add custom head (same architecture as training)
            head = torch.nn.Sequential(
                torch.nn.Linear(2048, 512),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(512, len(ATTRIBUTE_NAMES)),
                torch.nn.Sigmoid()
            )
            
            full_model = torch.nn.Module()
            full_model.backbone = model
            full_model.head = head
            
            # Load weights
            full_model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            full_model.to(self.device).eval()
            return full_model
        except Exception as e:
            print(f"[CustomAttributeAnalyzer] Failed to load: {e}")
            return None
    
    def analyze(self, img_rgb: np.ndarray) -> dict:
        if self.model is None:
            return self._empty_result()
        
        try:
            pil_img = Image.fromarray(img_rgb)
            tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(tensor)[0]  # (30,)
            
            scores = outputs.cpu().numpy()
            
            # Build result dict
            result = {}
            for attr_name, score in zip(ATTRIBUTE_NAMES, scores):
                result[attr_name] = bool(score >= 0.5)
                result[f'{attr_name}_score'] = float(score)
            
            return result
        
        except Exception as e:
            print(f"[CustomAttributeAnalyzer] Inference failed: {e}")
            return self._empty_result()
    
    @staticmethod
    def _empty_result():
        return {attr: False for attr in ATTRIBUTE_NAMES}
```

### 4.2 Update app.py to use custom analyzer

```python
# In face-service/app.py

from analyzers.attribute_analyzer import CustomAttributeAnalyzer  # Changed name

# Initialize
if attribute_analyzer is None:
    logger.info("Loading custom attribute analyzer...")
    attribute_analyzer = CustomAttributeAnalyzer()  # Changed

# Usage same as before — no other changes needed!
```

---

## Phase 5: Validation & Testing (Effort: MEDIUM, Time: 1-2 days)

### 5.1 Evaluation Metrics

Run on held-out test set (15% = 30K images from CelebA):

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# For each attribute:
# - Accuracy (overall correct%)
# - Precision (of 100 "yes" predictions, how many were correct)
# - Recall (of all true "yes" cases, how many were found)
# - F1 (harmonic mean of precision & recall)
# - ROC-AUC (how well the model separates yes/no)

# Example target metrics:
# - Accuracy: >95% per attribute
# - F1: >0.90
# - ROC-AUC: >0.95
```

### 5.2 Comparison with CLIP

Create test script to run side-by-side:

```python
# face-service/compare_models.py

from analyzers.attribute_analyzer import CustomAttributeAnalyzer
from analyzers.attribute_analyzer_clip import AttributeAnalyzer as CLIPAnalyzer
import numpy as np

custom = CustomAttributeAnalyzer()
clip = CLIPAnalyzer()

# Load test images
test_images = ...  # Your test set

for img in test_images:
    custom_out = custom.analyze(img)
    clip_out = clip.analyze(img)
    
    # Compare predictions
    custom_preds = [custom_out[attr] for attr in ATTRIBUTE_NAMES]
    clip_preds = [clip_out[attr] for attr in ATTRIBUTE_NAMES]
    
    agreement = sum(c == cl for c, cl in zip(custom_preds, clip_preds)) / len(ATTRIBUTE_NAMES)
    print(f"Agreement: {agreement * 100:.1f}%")
```

### 5.3 Manual Validation

- Test on 20-30 photos you personally took
- Compare predictions vs reality
- Fix any systematic errors (if accuracy <90%, retrain)

---

## Phase 6: Deployment & Monitoring (Effort: LOW, Time: 1 day)

### 6.1 Save Model to HuggingFace Hub (Optional)

```python
from huggingface_hub import upload_folder

upload_folder(
    folder_path="checkpoints/",
    repo_id="YOUR_USERNAME/hcp-attribute-classifier",
    repo_type="model"
)
```

### 6.2 Add to Requirements

```bash
# face-service/requirements.txt
torch==2.0.0
torchvision==0.15.0
albumentations==1.3.0
scikit-learn==1.2.0
wandb==0.14.0
```

### 6.3 Monitor Drift

After deployment, log predictions + ground-truth labels. Periodically compare model accuracy on new photos to catch degradation.

---

## Timeline & Effort Estimate

| Phase | Task | Time | Effort |
|-------|------|------|--------|
| **1a** | Download CelebA | 2h | Easy |
| **1b-c** | Collect + label own data | 3-4h | Easy |
| **2** | Model architecture setup | 1 day | Medium |
| **3** | Training code | 1-2 days | Medium |
| **4** | Training (on GPU) | 8-12h | Low (just waiting) |
| **5** | Integration | 1 day | Easy |
| **6** | Validation + testing | 1-2 days | Medium |
| **7** | Deployment | 1 day | Easy |
| **TOTAL** | | **2-3 weeks** | **~50-60 hours** |

---

## Risk & Mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| CelebA labels inaccurate | Model learns wrong patterns | Manual validation; use only high-agreement samples |
| Overfitting to CelebA | Poor performance on your users | Include your own photos in training; use augmentation + dropout |
| Class imbalance (e.g., few "wearing glasses" samples) | Model biased toward majority class | Use weighted BCE loss; oversample minority class |
| Not enough training data | Can't learn robust features | Start with CelebA (200K); collect more if <90% accuracy |
| GPU out of memory | Can't train | Use batch size 16 instead of 32; or use CPU (slow) |

---

## Decision Flowchart

```
START
  ↓
Can you devote 2-3 weeks?
  ├─ NO → Use CLIP as-is; document limitations
  ├─ YES ↓
Do you have labeled data?
  ├─ NO → Download CelebA (~2h)
  ├─ YES ↓
Can you access GPU (CUDA)?
  ├─ NO → Use CPU (training takes 2-3 days instead of 8-12h)
  ├─ YES ↓
Train on Phase 1-6 (follow plan above)
  ↓
Validation: >95% accuracy on test set?
  ├─ NO → Retrain with more data/epochs
  ├─ YES ↓
Deploy to production
  ↓
END
```

---

## Recommended First Steps

1. **Week 1:** Download CelebA (~2h) + collect your own photos (4-6h)
2. **Week 1:** Set up training environment (PyTorch, GPU) + organize data (2-3h)
3. **Week 1-2:** Train model (8-12h of compute time; you just monitor)
4. **Week 2:** Validate on test set; compare with CLIP
5. **Week 3:** If accuracy >95%, integrate into face-service; if <95%, retrain

**Go/No-Go Decision:** If test accuracy <90% after first training run, consider:
- Collecting more labeled data (50-100 more photos)
- Fine-tuning on YOUR photos only (subset of CelebA)
- Trying different model (EfficientNet, ViT)

---

## Questions to Answer Before Starting

1. **Which 5-10 attributes matter most to you?** (Start small, expand later)
2. **Do you have access to GPU?** (Nvidia CUDA recommended; CPU works but slow)
3. **Can you collect + label 50-100 of your own photos?** (Recommended for accuracy)
4. **What accuracy threshold would you accept?** (>95%? >90%? >80%?)
5. **How much time can you devote?** (2-3 weeks part-time, or 1 week full-time?)

---

## Alternative: Hybrid Approach (Less Risky)

Instead of replacing CLIP entirely, **use custom model + CLIP ensemble**:

```python
# Average predictions from both models
custom_pred = custom_model.predict(image)  # 0.0-1.0
clip_pred = clip_model.predict(image)      # 0.0-1.0

ensemble_pred = (custom_pred + clip_pred) / 2
final_output = ensemble_pred > 0.5
```

**Pros:**
- ✅ If custom model is wrong, CLIP might be right
- ✅ More robust than either alone
- ✅ Can gradually shift weight toward custom model

**Cons:**
- ❌ Slower (runs both models)
- ❌ More complex

---

## Next Steps

**Choose your path:**

A) **Full replacement** → Follow phases 1-7 (2-3 weeks)  
B) **Hybrid ensemble** → Train custom model (phases 1-4), then combine (1-2 weeks)  
C) **Wait & observe** → Collect user feedback first, validate CLIP accuracy (0 weeks)  
D) **Fine-tune existing** → Use CLIP but fine-tune on your data only (1 week)

Which interests you most?
