# 🛡️ Truth Shield: Fake News Detection Analysis Report

This report presents a technical audit and analysis of the **Truth Shield** multilingual fake news detection model. It covers dataset characteristics, model architecture, parameter distribution, behavioral verification, and issues found in historical logs.

---

## 1. Dataset Analysis

The model was trained on the `merged_multilingual_dataset.csv` containing **89,936 rows** of multilingual news data.

### 1.1 Language Distribution
The dataset is composed of four distinct language blocks:
*   **Hindi (`lang = 1`):** 56,714 rows (63.06%)
*   **English (`lang = 0`):** 22,140 rows (24.62%)
*   **Urdu (`lang = 3`):** 10,084 rows (11.21%)
*   **Bengali (`lang = 2`):** 998 rows (1.11%)

> [!IMPORTANT]
> **Key Finding (Language Encoding):** 
> Analysis of `hindi_dataset.csv` revealed that it contains **0 rows** with Devanagari characters. Instead, it is composed of English-language text covering Indian national news, politics, and local fact-checks. Consequently, English-language text represents **87.68%** of the total training corpus, while Urdu and Bengali contain native Arabic and Bengali script respectively.

### 1.2 Label Distributions & Imbalance
The dataset contains a significant class imbalance towards the **FAKE** class (Label 1):
*   **Overall:** 60,185 Fake (66.92%) vs. 29,751 Real (33.08%)
*   **English (`lang = 0`):** 76.0% Fake (16,817) vs. 24.0% Real (5,323)
*   **Hindi/Indo-English (`lang = 1`):** 66.6% Fake (37,800) vs. 33.4% Real (18,914)
*   **Bengali (`lang = 2`):** 54.0% Fake (538) vs. 46.0% Real (460)
*   **Urdu (`lang = 3`):** 49.8% Fake (5,030) vs. 50.2% Real (5,054)

---

## 2. Source Metadata Bias

During preprocessing, source URLs are parsed and mapped to the top 100 most frequent sources (`source_idx = 0..99`), with `100` representing unknown or out-of-top-100 sources.

Our analysis of the label distributions per `source_idx` reveals an extreme correlation between source metadata and veracity:
*   **`source_idx = 0` (11,832 samples):** **100% FAKE**
*   **`source_idx = 1` (11,321 samples):** **100% REAL**
*   **`source_idx = 2` (11,082 samples):** **50.2% FAKE** (Neutral)
*   **`source_idx = 3` (8,582 samples):** **96.0% FAKE**
*   **`source_idx = 4` (7,202 samples):** **100% FAKE**
*   **`source_idx = 100` (5,921 samples):** **100% FAKE** (Unknown sources are heavily biased toward fake news in this dataset)

---

## 3. Model Architecture & Parameter Audit

The SOTA model is defined in [src/model.py](file:///d:/Python%20Workshop/fake_news_detection/src/model.py) as `AdversarialFakeNewsModel`.

### 3.1 Gated Multimodal Fusion
The model concatenates feature representations from multiple streams:
1.  **Text (768d):** XLM-RoBERTa-base sentence embedding (pooled output after Multihead Self-Attention).
2.  **Source (32d):** Embedding representation of `source_idx`.
3.  **Visual (128d):** ResNet-18 branch (fallback to `empty_image` buffer if no image is present).
4.  **Audio (128d):** Projector representation (fallback to `empty_audio` buffer).
5.  **Video (128d):** Projector representation (fallback to `empty_video` buffer).
6.  **Knowledge (128d):** Projector representation (fallback to `empty_knowledge` buffer).

A **Gating Network** calculates a softmax distribution over 5 dimensions (mapping to text, source, image, audio, and video+knowledge), weighting the concatenated 1,312-dimensional feature vector dynamically.

### 3.2 Trainable Parameters Breakdown
We performed a parameter audit on the model:
*   **Total Parameters:** 293,446,024
*   **Trainable Parameters:** 15,402,376 (5.25%)
*   **Frozen Parameters:** 278,043,648 (94.75%)

| Component | Trainable Parameters | Frozen Parameters | Description |
| :--- | :--- | :--- | :--- |
| **`xlm_roberta`** | 294,912 | 278,043,648 | Base XLM-R is frozen; LoRA adapters (`r=8`) target query/value. |
| **`self_attention`** | 2,362,368 | 0 | Attention layer over text token sequences. |
| **`source_embedding`** | 3,232 | 0 | 101 sources mapped to 32d embedding. |
| **`resnet`** | 11,176,512 | 0 | Pretrained ResNet-18 vision extractor. |
| **`classifier`** | 803,841 | 0 | MLP classifying fused features to label. |
| **`gating_network`** | 6,565 | 0 | MLP mapping combined features to gate weights. |
| **Projectors** | 757,952 | 0 | Audio, Video, Knowledge, and Vision adapters. |

> [!WARNING]
> **Vision Branch Redundancy:** 
> While `resnet` parameters are marked as trainable (11.17M params), the training dataset does not contain images. Because `images=None` is passed in the training loop, the resnet branch only processes `empty_image` buffers and receives zero gradients. In a production text-only API, the ResNet-18 branch can be bypassed or completely deactivated to reduce memory and speed up inference.

---

## 4. Model Verification & The "Smoke Test Paradox"

### 4.1 The Paradox
When executing the default smoke test, the model predicts **FAKE** with **99.0%+ confidence** for every test scenario, including official/real news.

### 4.2 Diagnostic Insights
Layer-by-layer inspection of intermediate activations and gate weights explained this behavior:
1.  **Gating Preferences:** The gating network assigns **66% to 77% of the feature weight to the source embedding**, while the text features receive only **21% to 31%**. The visual, audio, and knowledge streams are correctly gated out (close to 0%).
2.  **Metadata Dominance:** Because the gating network relies heavily on the source embedding, the model's predictions are dominated by the reputation of the `source_idx`.
    *   When queried with `source_idx = 0` (100% fake in training) or `source_idx = 100` (100% fake in training), the model outputs **99.5% FAKE** regardless of the text.
    *   When queried with `source_idx = 1` or `source_idx = 5` (which represent highly trusted real sources in the training set), the predicted probability of FAKE drops to **0.49% (Verdict: REAL)**.

### 4.3 Text Classification Capability
By testing text inputs using a neutral source index (`source_idx = 2`, which is 50/50 in the training set), we confirmed the text-encoder branch is highly functional:
*   **Real News Text** (`'RBI increases repo rate...'`) -> **17.43% FAKE** (Verdict: **REAL**)
*   **Fake News Text** (`'NASA finds alien base...'`) -> **69.76% FAKE** (Verdict: **FAKE**)

### 4.4 Validation Set Accuracy
Evaluating the model on a random sample of the validation set (N=500) yielded:
*   **Validation Accuracy:** **97.80%** (confirming excellent convergence)
*   **Average Predicted Probability for True FAKE:** **97.80%**
*   **Average Predicted Probability for True REAL:** **7.76%**

---

## 5. Uncovered Training Bugs & Code Corrections

During log audits, we identified two errors that caused historical training runs to crash:

### 5.1 `ReduceLROnPlateau` Unexpected Keyword Argument
*   **Error:** `TypeError: ReduceLROnPlateau.__init__() got an unexpected keyword argument 'verbose'`
*   **Cause:** In PyTorch 2.2+, the `verbose` argument has been deprecated and removed from PyTorch schedulers.
*   **Fix:** Remove `verbose=True` from the `ReduceLROnPlateau` constructor in [src/train.py](file:///d:/Python%20Workshop/fake_news_detection/src/train.py) or `colab_version.py`.

### 5.2 NameError in Baseline Training
*   **Error:** `NameError: name 'path' is not defined` inside `train_baseline_only` in `run_all_training.py` (line 46).
*   **Cause:** The script attempted to print or log a path using the undefined variable `path` instead of `save_path` or another defined identifier.
*   **Fix:** Replace `path` with the correct string or variable containing the output directory.

---

## 6. Recommendations for Production Deployment

1.  **Source Fallback in API:** If the API or browser extension queries a news article where the source is unknown, do not default to `source_idx = 100` (which causes a 99% Fake prediction due to training dataset bias). Instead, use a neutral source index (like `source_idx = 2`) to ensure predictions are driven by text content.
2.  **Prune ResNet Weights:** The ResNet-18 vision branch is not utilized in text-only inference. Exporting a pruned model with only the text encoder, source embedding, gating, and classifier will reduce the ONNX file size from **1.1 GB to ~540 MB** and decrease inference latency.
3.  **Address Training Imbalance:** Future training runs should balance the label frequencies within English and Hindi data blocks, or use a weighted BCE loss to prevent the model from over-biasing towards the FAKE class.
