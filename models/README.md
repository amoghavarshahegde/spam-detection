# DistilBERT-Based Spam Detection System

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?logo=huggingface&logoColor=black)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-F9AB00?logo=googlecolab&logoColor=white)
![Task](https://img.shields.io/badge/Task-Binary%20Text%20Classification-informational)

> **Member 5 contribution** to the collaborative **Spam Detection** project: fine-tuning a transformer-based classifier (`distilbert-base-uncased`) to separate legitimate (ham) messages from spam.

| | |
|---|---|
| **Contributor** | Haseeba Riza |
| **Branch** | `Haseeba5-distilbert` |
| **Folder** | `models/` |
| **Notebook** | `DistilBERT_Spam_Detection.ipynb` |
| **Model** | `distilbert-base-uncased` |
| **Environment** | Google Colab (GPU / CUDA) |

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Member 5 Contribution](#2-member-5-contribution)
3. [Problem Statement](#3-problem-statement)
4. [Objectives](#4-objectives)
5. [Technologies and Tools Used](#5-technologies-and-tools-used)
6. [Dataset Description](#6-dataset-description)
7. [Data Preprocessing and Tokenization](#7-data-preprocessing-and-tokenization)
8. [DistilBERT Architecture and Working](#8-distilbert-architecture-and-working)
9. [Model Training Configuration](#9-model-training-configuration)
10. [Model Evaluation and Performance Metrics](#10-model-evaluation-and-performance-metrics)
11. [Confusion Matrix Interpretation](#11-confusion-matrix-interpretation)
12. [Custom Prediction Examples](#12-custom-prediction-examples)
13. [Project Folder Structure](#13-project-folder-structure)
14. [How to Run the Notebook in Google Colab](#14-how-to-run-the-notebook-in-google-colab)
15. [Future Improvements](#15-future-improvements)
16. [Conclusion](#16-conclusion)

---

## 1. Project Overview

Spam messages waste time, spread scams, and can expose users to phishing links. This project is a collaborative college-level machine learning effort to detect spam in text messages. Each team member implements and evaluates a different modelling approach on the same classification task.

This README documents **Member 5's part**: a spam classifier built by fine-tuning **DistilBERT**, a compact pretrained transformer, using PyTorch and Hugging Face Transformers. The model classifies each message into one of two classes:

| Label | Class | Meaning |
|:---:|---|---|
| `0` | Ham | Legitimate message |
| `1` | Spam | Unwanted message |

---

## 2. Member 5 Contribution

**Contributor:** Haseeba Riza

The work delivered in this branch (`Haseeba5-distilbert`) covers:

- Loading the training, validation and test splits of the shared dataset.
- Tokenizing messages with the DistilBERT tokenizer (maximum length 128).
- Fine-tuning `distilbert-base-uncased` for binary classification on a CUDA GPU in Google Colab.
- Evaluating the model on the held-out test set (accuracy, precision, recall, F1-score).
- Producing a confusion matrix and visualizations with Matplotlib and Seaborn.
- Testing the trained model on custom, hand-written example messages.

All work is contained in a single notebook: **`DistilBERT_Spam_Detection.ipynb`** (inside the `models/` folder).

---

## 3. Problem Statement

Given the text of a message, automatically decide whether it is **ham** (legitimate) or **spam** (unwanted). The task is a binary text classification problem. Because spam is typically much rarer than ham in real message collections, a good solution must do more than achieve high accuracy: it should also catch as much spam as possible (recall) without wrongly flagging legitimate messages (precision).

---

## 4. Objectives

- Fine-tune a pretrained transformer (DistilBERT) for spam detection.
- Evaluate the model fairly on a held-out test set that was not used during training.
- Report metrics that are meaningful for an imbalanced problem (precision, recall, F1-score, confusion matrix), not accuracy alone.
- Check model behaviour on a few custom example messages.
- Contribute a transformer-based model that the team can compare with the other members' approaches.

---

## 5. Technologies and Tools Used

| Category | Tool |
|---|---|
| Language | Python |
| Deep learning framework | PyTorch |
| Pretrained model library | Hugging Face Transformers |
| Pretrained model | `distilbert-base-uncased` |
| Evaluation | Scikit-learn |
| Data handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Environment | Google Colab with GPU (CUDA) |
| Version control | Git / GitHub |

---

## 6. Dataset Description

The dataset contains **5,085 messages** divided into three splits:

| Split | Messages | Share of total |
|---|---:|---:|
| Training | 3,631 | ~71.4% |
| Validation | 727 | ~14.3% |
| Testing | 727 | ~14.3% |
| **Total** | **5,085** | **100%** |

**Columns**

| Column | Description |
|---|---|
| `text` | Message text used as model input |
| `raw_text` | Original message text |
| `label` | Target class: `0` = Ham, `1` = Spam |

**Test-set class composition** (derived from the confusion matrix in Section 11): 642 ham messages and 85 spam messages, i.e. spam is roughly 11.7% of the test set. This imbalance is why precision, recall and F1-score are reported alongside accuracy.

> **Note:** Dataset files are **not** included in this repository. The dataset is shared across the team project and must be supplied separately when running the notebook (see [Section 14](#14-how-to-run-the-notebook-in-google-colab)).

---

## 7. Data Preprocessing and Tokenization

Neural networks cannot read raw text, so each message is converted into numbers before training:

1. **Load the splits** — training, validation and test data are read with Pandas.
2. **Tokenization** — the `distilbert-base-uncased` tokenizer splits each message into WordPiece sub-word tokens and maps them to integer IDs. The *uncased* model lowercases all text.
3. **Special tokens** — `[CLS]` is added at the start (its final representation is used for classification) and `[SEP]` at the end.
4. **Padding and truncation** — every sequence is padded or truncated to a fixed **maximum length of 128 tokens**.
5. **Attention mask** — a mask tells the model which positions are real tokens and which are padding.
6. **Batching** — tokenized data is wrapped as PyTorch tensors and fed to the model in batches of 16.

| Setting | Value |
|---|---|
| Tokenizer | `distilbert-base-uncased` |
| Maximum sequence length | 128 |
| Padding / truncation | Enabled |

---

## 8. DistilBERT Architecture and Working

**DistilBERT** is a smaller, faster version of BERT created through *knowledge distillation*: a compact "student" model is trained to reproduce the behaviour of the larger BERT "teacher". It keeps most of BERT's language understanding while using fewer layers (6 transformer layers instead of BERT-base's 12), which makes it lighter and quicker to train — well suited to a Colab GPU.

**How it works for spam detection**

```
Message text
   │
   ▼
Tokenizer  →  token IDs + attention mask
   │
   ▼
DistilBERT encoder (pretrained transformer layers with self-attention)
   │
   ▼
[CLS] representation
   │
   ▼
Classification head (2 outputs: Ham / Spam)
   │
   ▼
Softmax  →  class probabilities  →  predicted label + confidence
```

- **Self-attention** lets every token consider every other token in the message, so the model captures context (for example, "claim" near "prize" and "click here").
- **Transfer learning:** the encoder starts with general language knowledge from pretraining; **fine-tuning** adapts it to the spam/ham task using the training data.
- **Classification head:** a small layer on top of the encoder outputs two scores, converted by softmax into probabilities. The reported *confidence* is the probability of the predicted class.

---

## 9. Model Training Configuration

| Parameter | Value |
|---|---|
| Pretrained model | `distilbert-base-uncased` |
| Number of classes | 2 |
| Epochs | 3 |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Maximum sequence length | 128 |
| Framework | PyTorch (Hugging Face Transformers) |
| Hardware | Google Colab GPU (CUDA) |

The validation split is used to monitor training, and the test split is reserved for the final evaluation.

---

## 10. Model Evaluation and Performance Metrics

Final results on the **test set (727 messages)**:

| Metric | Score |
|---|---:|
| Accuracy | **98.21%** |
| Precision | **95.00%** |
| Recall | **89.41%** |
| F1-score | **92.12%** |

Precision, recall and F1-score are computed for the **spam** class.

| Metric | What it tells us |
|---|---|
| **Accuracy** | Share of all messages classified correctly. |
| **Precision** | Of the messages predicted as spam, how many really were spam. |
| **Recall** | Of the actual spam messages, how many the model caught. |
| **F1-score** | Harmonic mean of precision and recall; a single balanced measure. |

Because ham messages greatly outnumber spam, accuracy alone can look high even for a weak model. Precision, recall and F1 give a more honest picture of spam-detection quality.

---

## 11. Confusion Matrix Interpretation

| | Predicted Ham | Predicted Spam |
|---|---:|---:|
| **Actual Ham** | 638 | 4 |
| **Actual Spam** | 9 | 76 |

| Outcome | Count | Meaning |
|---|---:|---|
| True Negatives (TN) | 638 | Ham correctly identified as ham |
| False Positives (FP) | 4 | Ham wrongly flagged as spam |
| False Negatives (FN) | 9 | Spam that slipped through as ham |
| True Positives (TP) | 76 | Spam correctly identified as spam |

**Interpretation**

- Out of 642 legitimate messages, only **4** were wrongly flagged as spam, so legitimate messages are rarely blocked.
- Out of 85 spam messages, **76** were caught and **9** were missed. This is why recall (89.41%) is lower than precision (95.00%): the model is somewhat more likely to miss a spam message than to falsely flag a ham message.
- The metrics follow directly from the matrix: Precision = 76 / (76 + 4), Recall = 76 / (76 + 9), Accuracy = (638 + 76) / 727.

These results apply to this specific test split only; they should not be assumed to hold on other datasets or message types.

---

## 12. Custom Prediction Examples

The trained model was tested on a few hand-written messages:

| # | Message | Prediction | Confidence |
|:---:|---|:---:|---:|
| 1 | "Congratulations! You won a free iPhone. Click here to claim your prize!" | **SPAM** | 64.60% |
| 2 | "Mom, I will reach home by 6 pm. See you soon." | **HAM** | 99.67% |
| 3 | "Hey, are you coming to college tomorrow?" | **HAM** | 99.66% |

All three examples were classified as expected. Note that the spam example received a noticeably lower confidence (64.60%) than the ham examples, so the model was less certain on that message. Three examples are a qualitative sanity check only and are not a substitute for the test-set evaluation above.

---

## 13. Project Folder Structure

The portion of the repository that belongs to this contribution (branch `Haseeba5-distilbert`):

```
Spam-Detection/
└── models/
    ├── DistilBERT_Spam_Detection.ipynb   # Training, evaluation and custom predictions
    └── README.md                         # This documentation
```

Not included in the repository:

- Dataset files (supplied separately).
- Trained model weights/checkpoints (too large for GitHub).

Other team members' work lives in their own branches/folders of the shared project.

---

## 14. How to Run the Notebook in Google Colab

**Prerequisites:** a Google account, access to the project dataset (train / validation / test splits), and the notebook from this repository.

1. **Get the notebook**

   ```bash
   git clone -b Haseeba5-distilbert <repository-url>
   cd <repository-folder>/models
   ```

   Alternatively, download `DistilBERT_Spam_Detection.ipynb` from the `models/` folder on GitHub.

2. **Open it in Colab** — go to [colab.research.google.com](https://colab.research.google.com), choose **File → Upload notebook** (or **GitHub** tab), and open `DistilBERT_Spam_Detection.ipynb`.

3. **Enable the GPU** — **Runtime → Change runtime type → Hardware accelerator: GPU**.

4. **Provide the dataset** — upload the training, validation and test files to the Colab session (or mount Google Drive), and make sure the file paths used in the notebook's data-loading cells match where you placed them.

5. **Install dependencies** — run the installation cell in the notebook if the libraries are not already available. Colab typically includes PyTorch, Pandas, NumPy, Scikit-learn, Matplotlib and Seaborn; the Hugging Face libraries may need installing:

   ```bash
   pip install transformers
   ```

6. **Run all cells** — **Runtime → Run all**. The notebook will tokenize the data, fine-tune the model for 3 epochs, evaluate on the test set, plot the confusion matrix and run the custom predictions.

> Exact results may vary slightly between runs because of random initialization and hardware differences.

---

## 15. Future Improvements

> The items below are **ideas only** and are **not** part of the current implementation.

- Experiment with hyperparameters (learning rate, epochs, batch size, maximum length) using the validation set.
- Address class imbalance, for example with class weights, to improve spam recall.
- Compare with larger transformers (e.g. BERT, RoBERTa) and with the other team members' models under identical conditions.
- Perform error analysis on the 9 missed spam messages and 4 false alarms.
- Evaluate with cross-validation or on additional, external datasets to test generalization.
- Calibrate or threshold the output probabilities to balance precision and recall.
- Package the model as a reusable inference script or application.

---

## 16. Conclusion

This contribution fine-tuned `distilbert-base-uncased` for binary spam detection and evaluated it on a held-out test set of 727 messages, obtaining **98.21% accuracy**, **95.00% precision**, **89.41% recall** and a **92.12% F1-score**. The confusion matrix shows 4 false positives and 9 false negatives, indicating that the model rarely blocks legitimate messages but misses a small share of spam. Custom examples behaved as expected, although the spam example was classified with comparatively low confidence.

This work is a college-level academic project. The results come from a single dataset and a single train/validation/test split, so the model should be regarded as a research prototype rather than a production-ready system. It provides a transformer-based baseline that the team can compare with the other approaches in the Spam Detection project.

---

**Author:** Haseeba Riza (Member 5) · **Branch:** `Haseeba5-distilbert`
