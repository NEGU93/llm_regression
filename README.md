# LLM Regression

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/NEGU93)

Can we actually do regression tasks with LLMs? And are they any good? Let's see...

There are two main projects here:

- **🔍 RAG-enhanced GPT-4 mini** for improved contextual understanding  
- **🦙 Fine-tuned Llama 3.1** with QLoRA optimization

## Benchmark models


| Rank | Model | Type | Status |
|------|-------|------|--------|
| 1 | Random prices | Baseline | ✅ |
| 2 | Average price of training set | Baseline | ✅ |
| 3 | Bag of Words + Linear Regression | Traditional ML | ✅ |
| 4 | Word2Vec + Linear Regression | Traditional ML | ✅ |
| 5 | Word2Vec + Linear SVR | Traditional ML | ✅ |
| 6 | GPT-4 mini | LLM | ✅ |
| 7 | **GPT-4 mini with RAG** | LLM + RAG | ✅ |
| 8 | Quantized Llama 3.1 | LLM | ✅ |
| 9 | **Fine-tuned Llama 3.1** | LLM + Fine-tuning | 🕐 *Coming soon* |


![benchmark_results](img/benchmark.png)

## ⚡ Quickstart

### 1️⃣ Environment Setup

```bash
# Install conda (if not already installed)
# Then create the environment
conda env create -f environment.yml -n llm-regression python=3.11
conda activate llm-regression
```

### 2️⃣ Configuration

Create your `.env` file with the required API keys:

```env
OPENAI_API_KEY=sk-proj-your-openai-key-here
HF_TOKEN=hf_your-huggingface-token-here
WANDB_API_KEY=your-wandb-key-here  # Optional for experiment tracking
```

### 3️⃣ RAG Regression

Enhance GPT-4 mini with **Retrieval-Augmented Generation** for better price predictions.

```bash
python -m llm_regressor.rag_regressor.create_db
```

The RAG system creates a vector database of similar products to provide relevant context during inference.

### 4️⃣ Run the Benchmark

```bash
python -m llm_regressor.benchmark
```

## 🦙 Training Llama

Normally you DON'T need to train, you will download my trained model directly from Hugging Face, but if you wish to train follow this steps:

- Request access to the [Llama 3.1 model on Hugging Face](https://huggingface.co/meta-llama/Llama-3.1-8B):
    1. Visit the [Llama 3.1 8B model page](https://huggingface.co/meta-llama/Llama-3.1-8B)
    2. Click "Request access to this model"
    3. Fill out the access request form with your intended use case
    4. Wait for Meta's approval (usually takes 1-2 business days)
    5. Once approved, your `HF_TOKEN` will have access to download the model
- Select Training parameters (Optional): Change parameters like epochs, batch size, QLoRA hyper-parameters in [llm_regressor/__init__.py](https://github.com/NEGU93/llm_regression/blob/main/llm_regressor/__init__.py).
- Then run `python -m llm_regressor.trainer`

### 📊 Dataset Information

**Source**: Curated [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)  
**Location**: [🤗 NEGU93/pricer-data](https://huggingface.co/datasets/NEGU93/pricer-data)  
**Auto-download**: Via your `HF_TOKEN` 🔄

### Dataset Statistics
- **Training examples**: 400,000
- **Test examples**: 200,000  
- **Features**: Item text description + price
- **Price range**: $1 - $1,000
- **Categories**: Electronics, Home & Garden, Sports, Books, and more
