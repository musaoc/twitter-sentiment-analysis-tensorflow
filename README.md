# Twitter Sentiment Analysis with TensorFlow & BiLSTM

A deep learning natural language processing pipeline trained on 1.6 million tweets to classify sentiment with over 75% accuracy using TensorFlow, custom word embeddings, and Bidirectional LSTMs.

[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/lazer999/twitter-sentiment-analysis-tensorflow-0-75)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Field](https://img.shields.io/badge/Field-Natural%20Language%20Processing%20/%20Deep%20Learning-brightgreen)](#)

---

## Table of Contents
- [Project Overview](#project-overview)
- [Key Highlights & Results](#key-highlights--results)
- [System Architecture & Workflow](#system-architecture--workflow)
- [Repository Structure](#repository-structure)
- [Quickstart & Reproduction](#quickstart--reproduction)
- [Dataset Details](#dataset-details)
- [Author & Acknowledgments](#author--acknowledgments)

---

## Project Overview

This repository provides the complete, production-structured implementation of the **[Twitter Sentiment Analysis with TensorFlow & BiLSTM](https://www.kaggle.com/code/lazer999/twitter-sentiment-analysis-tensorflow-0-75)** project originally published on Kaggle. 

The primary focus of this work is translating complex data into actionable machine learning solutions using disciplined data engineering, rigorous validation strategies, and clean, leak-free preprocessing pipelines.

---

## Key Highlights & Results

- Engineered full NLP preprocessing: regex cleaning, mention removal, URL stripping, and NLTK stopword handling.
- Tokenized vocabulary with sequence length optimization and zero-padding.
- Built Bidirectional LSTM deep neural network capturing long-range contextual sentiment dependencies.
- Achieved 75%+ out-of-fold validation accuracy on notoriously noisy social media text.
- Integrated training diagnostics: loss/accuracy learning curves and custom phrase inference.

---

## System Architecture & Workflow

The pipeline follows a structured, modular execution path:

```mermaid
flowchart LR
    A[1.6M Raw Tweets] --> B[Regex Normalization & Stopword Filter]
    B --> C[Tokenizer & Sequence Padding]
    C --> D[Dense Word Embeddings]
    D --> E[Bidirectional LSTM Layer]
    E --> F[Dropout Regularization]
    F --> G[Sentiment Prediction > 75% Acc]
```

---

## Repository Structure

```plaintext
twitter-sentiment-analysis-tensorflow/
├── notebooks/
│   └── twitter-sentiment-analysis-tensorflow.ipynb      # Original Jupyter notebook with full exploratory visuals
├── src/
│   └── main.py                # Modular, executable Python pipeline
├── .gitignore                 # Standard Python/Jupyter ignores
├── LICENSE                    # MIT License
├── README.md                  # Human-friendly documentation
└── requirements.txt           # Verified Python dependencies
```

---

## Quickstart & Reproduction

### 1. Clone the Repository
```bash
git clone https://github.com/musaoc/twitter-sentiment-analysis-tensorflow.git
cd twitter-sentiment-analysis-tensorflow
```

### 2. Set Up a Virtual Environment
```bash
# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Pipeline
You can run the end-to-end script directly:
```bash
python src/main.py
```

Or open and run the interactive notebook:
```bash
jupyter lab notebooks/twitter-sentiment-analysis-tensorflow.ipynb
```

---

## Dataset Details

- **Dataset / Competition**: [Sentiment140 (1.6 Million Tweets)](https://www.kaggle.com/datasets/kazanova/sentiment140)
- **Origin Platform**: Kaggle
- For automated dataset downloading via Kaggle CLI:
  ```bash
  kaggle datasets download -d kazanova/sentiment140
  ```

---

## Author & Acknowledgments

- **Author**: **Muhammad Musa Khan** (Kaggle Master)
- **Kaggle Profile**: [@lazer999](https://www.kaggle.com/lazer999)
- **GitHub**: [@musaoc](https://github.com/musaoc)
- **Original Kaggle Solution**: [Twitter Sentiment Analysis with TensorFlow & BiLSTM](https://www.kaggle.com/code/lazer999/twitter-sentiment-analysis-tensorflow-0-75)

If you found this project helpful or insightful, please consider starring the repository ⭐!
