# 🌪️ Disaster Tweet Classification (NLP Getting Started)

**Goal:** Classify tweets as real disasters or non-disasters using Natural Language Processing.<br>
**Best Validation F1:** 0.8041 (DistilBERT)

## 📌 Project Overview
This project is my implementation of Kaggle's **Natural Language Processing with Disaster Tweets** competition. The task is binary text classification: given a tweet, predict whether it refers to a real disaster (`1`) or not (`0`).

I approached the problem in stages: starting with a neural baseline, improving with pre-trained word embeddings, and then fine-tuning a transformer model for stronger language understanding.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, TensorFlow/Keras, Hugging Face Transformers, Datasets, PyTorch, Matplotlib
* **Models:** BiLSTM baseline, BiLSTM + GloVe embeddings, DistilBERT

## 📊 The Data Process (My Approach)

### 1. Data Preparation
I loaded the Kaggle training set and combined multiple text fields into one modeling input.
* Merged `keyword`, `location`, and `text` into a single `text_combined` feature.
* Dropped metadata columns after consolidation (`id`, `keyword`, `location`).
* Used a stratified train/validation split (80/20) to preserve class balance.

### 2. Lightweight Text Cleaning
I used practical cleaning rules to improve consistency without destroying meaning.
* Replaced URLs with a placeholder token (`url`).
* Replaced mentions with a placeholder token (`user`).
* Removed noisy HTML and non-ASCII artifacts.
* Kept core sentence structure intact for downstream models.

### 3. Tokenization and Sequence Setup (Neural Models)
For LSTM-based models, tweets were converted to fixed-length numeric sequences.
* Built a tokenizer vocabulary capped at 10,000 words.
* Converted tweets to integer sequences.
* Padded sequences to a fixed length of 35 tokens.

## ⚙️ Model Iteration & Results

I trained and evaluated three progressively stronger approaches using F1 as the main metric.

### 1. Baseline BiLSTM
* Architecture: Embedding -> SpatialDropout -> Bidirectional LSTM -> Dense layers
* Validation F1: **0.7539**
* Observation: Model missed context-heavy disaster tweets where vocabulary nuance mattered.

### 2. BiLSTM + GloVe Embeddings
* Initialized embeddings with pre-trained **GloVe 100d** vectors.
* Trained in two stages: frozen embedding warm-up, then low learning-rate fine-tuning.
* Validation F1: **0.7710**
* Threshold tuning improved best observed F1 to **0.7811**.

### 3. DistilBERT (Best Model)
* Fine-tuned `distilbert-base-uncased` for binary sequence classification.
* Tokenized tweets to max length 64 with transformer tokenizer.
* Trained for 5 epochs with evaluation each epoch and best-model loading.
* Final validation F1: **0.8041**

## 🧠 Key Learnings
* **Transfer learning wins in NLP:** DistilBERT outperformed LSTM variants by capturing contextual meaning and subtle phrasing.
* **Threshold selection matters:** Optimizing decision thresholds improved F1 beyond default 0.5-style cutoffs.
* **Pre-trained embeddings help, but context is king:** GloVe improved over baseline, but transformer context modeling gave the largest gain.

## 🚀 How to Run
1. Clone this repository.
2. Ensure you have the Kaggle competition files (`train.csv`, `test.csv`) for NLP Getting Started.
3. Open and run [disaster-tweet.ipynb](disaster-tweet.ipynb) cell by cell.
4. Generate predictions and export `submission.csv` from the final section.

## 📦 Output
The notebook produces a Kaggle-ready submission file:
* `submission.csv` with columns:
  * `id`
  * `target`
