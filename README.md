# LLM Fine-Tuning Demo

A complete workflow for fine-tuning large language models using LoRA (Low-Rank Adaptation) and QLoRA (Quantized LoRA) techniques. This project demonstrates how to fine-tune models like Llama 3.2 on the SAMSum dataset for dialogue summarization tasks.

## Features

- **QLoRA Fine-tuning**: 4-bit quantized LoRA for memory-efficient training
- **End-to-end Pipeline**: Training, evaluation, and model deployment
- **Hugging Face Integration**: Easy authentication and model upload to Hub
- **Configurable**: YAML-based configuration for easy experimentation
- **Evaluation Metrics**: ROUGE scores for summarization quality assessment

## Project Structure

```
llm-fine-tuning-demo/
├── config/
│   └── config.yaml          # Model and training configuration
├── utils/
│   ├── config_utils.py      # Configuration loading utilities
│   ├── data_utils.py        # Dataset loading and preprocessing
│   ├── model_utils.py       # Model setup and LoRA configuration
│   ├── inference_utils.py   # Text generation and evaluation
│   └── hf_utils.py          # Hugging Face authentication
├── Demo/
│   └── TrainAndEvaluate.ipynb  # Complete training and evaluation notebook
├── train_qlora.py           # Training script
├── evaludate_qlora.py       # Evaluation script
└── paths.py                 # Project path configuration
```

## Quick Start

### 1. Setup

Install required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Configure

Edit `config/config.yaml` to set:
- Base model (e.g., `meta-llama/Llama-3.2-1B-Instruct`)
- Dataset configuration
- Training hyperparameters (learning rate, batch size, epochs, etc.)
- LoRA parameters (r, alpha, target modules)

### 3. Authenticate

Set your Hugging Face token:
```bash
export HUGGINGFACE_TOKEN="your_token_here"
```

Or use the authentication function in the notebook.

### 4. Train and Evaluate

**Option A: Using Jupyter Notebook (Recommended)**
```python
# Open Demo/TrainAndEvaluate.ipynb
# Run all cells to:
# - Authenticate with Hugging Face
# - Load configuration
# - Train the model
# - Evaluate with ROUGE metrics
# - Upload to Hugging Face Hub
```

**Option B: Using Python Scripts**
```bash
# Train
python train_qlora.py

# Evaluate
python evaludate_qlora.py
```

## Configuration

Key configuration options in `config/config.yaml`:

- **Model**: Base model identifier from Hugging Face
- **Dataset**: Dataset name and split configuration
- **LoRA**: Rank (r), alpha, dropout, and target modules
- **Training**: Learning rate, batch size, epochs, sequence length
- **Quantization**: 4-bit quantization settings for QLoRA

## Outputs

- **Trained Adapters**: Saved to `data/outputs/lora_samsum/lora_adapters/`
- **Evaluation Results**: ROUGE scores saved as JSON
- **Predictions**: Model predictions saved as JSONL

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- Hugging Face account with access to gated models (if using Llama)

## License

This project is for educational and demonstration purposes.

