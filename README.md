# SentimentScope IMDB Movie Review Sentiment using Transformer in PyTorch

This project builds an end-to-end **binary sentiment classifier** for movie reviews using the **IMDB Large Movie Review Dataset** and a custom **Transformer-based model** implemented in **PyTorch**. The notebook walks through loading and exploring the dataset, preparing a custom `Dataset`/`DataLoader`, training a transformer model, evaluating accuracy, and saving/loading a checkpoint for inference.

## Project Goals
- Load and preprocess the IMDB dataset (train/test splits)
- Explore the dataset with descriptive statistics and visualizations
- Implement a custom PyTorch `Dataset` for text classification
- Build and train a Transformer model for **positive vs negative** sentiment
- Achieve **>75% test accuracy** (stretch goal: **>90%** with tuning)
- Save a model checkpoint and provide a simple batch inference interface

---

## Dataset
We use the **IMDB Large Movie Review Dataset** (50,000 reviews total: 25k train, 25k test) introduced by Maas et al.

**Official Stanford dataset page:**  
https://ai.stanford.edu/~amaas/data/sentiment/

**What you need:**
- Download the archive: `aclImdb_v1.tar.gz`
- Extract it so a folder named `aclimdb/` (or `aclImdb/`) exists **next to the notebook**

Expected structure:
```
aclimdb/
  train/
    pos/
    neg/
  test/
    pos/
    neg/
```

> Note: Some environments extract the folder as `aclImdb/` (capital I). The notebook supports both if configured accordingly.

---

## Repository Structure
```
.
├── SentimentScope_starter.ipynb     # Notebook (data loading → training → evaluation)
├── aclimdb/ or aclImdb/             # Dataset folder (download + extract)
└── demogpt_imdb_checkpoint.pt       # Saved model checkpoint (created after training)
```

---

## Setup

### 1) Create/activate an environment (optional but recommended)
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows
```

### 2) Install dependencies
```bash
pip install torch numpy pandas matplotlib
```

> If you’re using Jupyter locally:
```bash
pip install notebook
```

### 3) Download and extract the dataset
From the Stanford page:  
https://ai.stanford.edu/~amaas/data/sentiment/

Example (macOS/Linux):
```bash
# From the same folder as the notebook:
tar -xvzf aclImdb_v1.tar.gz
```

Rename if needed (optional):
```bash
# if it extracted as aclImdb and you want lowercase:
mv aclImdb aclimdb
```

---

## How to Run
1. Open the notebook:
   ```bash
   jupyter notebook
   ```
2. Run all cells from top to bottom.
3. The notebook will:
   - Load IMDB text files from `train/pos`, `train/neg`, `test/pos`, `test/neg`
   - Build a vocabulary/tokenization pipeline
   - Train the Transformer model
   - Evaluate **validation accuracy** and **test accuracy**
   - Save a checkpoint (e.g., `demogpt_imdb_checkpoint.pt`)

---

## Model Overview
The model (`DemoGPT`) is a compact Transformer encoder for binary classification:
- Token embedding layer
- Positional encodings
- Multi-layer Transformer encoder (configurable depth/heads)
- Masked pooling over tokens
- Classification head outputting a single logit per review

Loss function: **Binary Cross Entropy with Logits** (`BCEWithLogitsLoss`)  
Metric: **Accuracy**

---

## Results
Final model performance from the completed run:
- **Validation Accuracy:** ~82%
- **Test Accuracy:** **~77–78%** (meets the project requirement of >75%)

Your exact result may vary slightly depending on:
- random seed
- tokenizer/vocab size
- max sequence length (truncation)
- model capacity (embedding size, layers, heads)
- learning rate / regularization

---

## Improving Test Accuracy (Suggested Experiments)
If you want to push beyond ~80% test accuracy, try:
- Increase `MAX_LENGTH` (e.g., 128 → 256)
- Increase model width (`d_embed` 128 → 256) and heads (4 → 8)
- Add/raise dropout (0.2 → 0.3)
- Use AdamW with weight decay (e.g., 0.01)
- Save the **best validation checkpoint** and evaluate test on that
- Replace mean pooling with **attention pooling**

---

## Checkpoint + Inference
After training, a checkpoint is saved to disk. The notebook includes a helper interface to:
- load the model weights
- run inference on a **batch of raw text strings**
- return predicted labels and probabilities

Example usage (in the notebook):
```python
predictor = SentimentPredictor(checkpoint_path="demogpt_imdb_checkpoint.pt", config=config)
texts = ["This movie was amazing!", "Worst film I've ever seen."]
preds, probs = predictor.predict(texts)
print(preds, probs)
```
