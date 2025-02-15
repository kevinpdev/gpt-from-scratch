# 🧠 Train a Small GPT-Style LLM from Scratch

🚀 **This repository contains a Jupyter Notebook that trains a small GPT-style, decoder-only language model from scratch using PyTorch.**

🔗 **[Open the Notebook](./llm-from-scratch.ipynb)**

## 📌 Overview

This project is an educational walkthrough of the full process of building and training a **Minimal GPT-style Decoder Only Transformer Model**. The notebook covers:

- 📖 **Tokenization** – Converting text into tokens
- 🔄 **Positional Encoding** – Adding order to input sequences
- 📈 **Self Attention Intuition** - Building intuition behind the self attention operation
- 🏗 **Transformer Decoder Blocks** – Multi-head self-attention & feedforward layers
- 🎯 **Training from Scratch** – Using a small pretraining and SFT dataset to train a language model
- 🔥 **Inference** – Generating text using the trained model

## 📂 Repository Structure

📂 gpt-from-scratch
│── 📄 README.md # Project documentation (this file)
│── 📒 llm-from-scratch.ipynb # Jupyter Notebook with full training pipeline

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/kevinpdev/gpt-from-scratch.git
cd gpt-from-scratch
```

### 2️⃣ Install Dependencies

Make sure you have Python and Jupyter installed. Install required packages:

```bash
pip install torch transformers datasets tqdm jupyter
```

### 3️⃣ Run the Notebook

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open llm-from-scratch.ipynb and run

## 🎯 Goals & Use Cases

✅ Understand dataset formats and working with Huggingface libraries
✅ Learn the process of tokenization
✅ Learn the inner workings of GPT-style models
✅ Train a small-scale Transformer on a custom dataset
✅ Understand self-attention and language modeling
✅ Experiment with fine-tuning & inference

## 🔗 Notebook & Resources

📌 Notebook: llm-from-scratch.ipynb
📖 Transformer Paper: [“Attention Is All You Need"](https://arxiv.org/pdf/1706.03762)
📖 GPT Paper: ["Improving Language Understanding by Generative Pre-Training"](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
🛠 PyTorch Documentation: [pytorch.org](https://pytorch.org/)
👐 Huggingface Documentation: [https://huggingface.co/docs](https://huggingface.co/docs)
