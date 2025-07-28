## Quick Navigation

- [Level-Up Giants: 8-bit Training for Massive Models](#level-up-giants-8-bit-training-for-massive-models)
- [Task-Focused Training: Aim for Better Learning - Part 1](#task-focused-training-aim-for-better-learning-part-1)
- [Task-Focused Training: Aim for Better Learning - Part 2](#task-focused-training-aim-for-better-learning-part-2)
- [Edge of Hardware Limits: Scaling Inputs with Flash Attention 2](#edge-of-hardware-limits-scaling-inputs-with-flash-attention-2)
- [Edge of Hardware Limits: Reaching 4bit Training with QLoRA](#edge-of-hardware-limits-reaching-4bit-training-with-qlora)

---

## Level-Up Giants: 8-bit Training for Massive Models



## 📌 Quick Navigation

- [Training Phi-3: Scaling to Larger LLMs](#training-phi-3-scaling-to-larger-llms)
- [Memory Challenges and 8-Bit Quantization](#memory-challenges-and-8-bit-quantization)
- [Performance, Trade-offs, and Results](#performance-trade-offs-and-results)
- [Using the SQuAD Dataset for LLM QA](#using-the-squad-dataset-for-llm-qa)
- [Phi-3 Configuration: `specialised_phi.yml`](#phi-3-configuration-specialised_phiyml)
- [Gemma 2-27B Configuration: `specialised_gemma.yml`](#gemma-2-27b-configuration-specialised_gemmayml)
- [Colab Notebook](#colab-notebook)
- [References & Further Reading](#references--further-reading)

---

## Training Phi-3: Scaling to Larger LLMs

This session demonstrates the training of Microsoft’s Phi-3 Medium, a 14B parameter model with a 128k context window, on a single 24GB GPU using Axolotl and LoRA. Techniques such as gradient accumulation, checkpointing, and 8-bit quantization enable training without sacrificing batch size or sequence length.

[Back to Top](#quick-navigation)

---

## Memory Challenges and 8-Bit Quantization

Despite tuning batch size and sequence length, the Phi-3 model exceeded available memory due to its ~30GB weight size. To overcome this:

- **Quantization to 8-bit** (`load_in_8bit: true`) was applied.
- This halved memory usage, allowing training to proceed.
- Axolotl supports this via a simple YAML flag.

💡 *Note: Precision is traded for capacity; however, the gain in model size (14B vs. 7B) outweighs the loss from quantization.*

[Back to Top](#quick-navigation)

---

## Performance, Trade-offs, and Results

- Training succeeded with **no OOM errors** and **room to spare**.
- Sequence length and batch size compromises were reversed.
- Training was slower than LLaMA 8B due to the model's size.
- Despite quantization, Phi-3 handled the dataset well and generated fluent outputs.

[Back to Top](#quick-navigation)

---

## Using the SQuAD Dataset for LLM QA

The model was fine-tuned on the **SQuAD** (Stanford Question Answering Dataset), adapted for generative LLMs.

### Prompt Format:
```plaintext
<|user|>
 {input} {instruction} </s>
<|assistant|>
```

- Context and question are combined into a system-level prompt.
- Model learns to generate precise answers based on instruction tuning.

[Back to Top](#quick-navigation)

---

## Phi-3 Configuration: `specialised_phi.yml`

```yaml
base_model: microsoft/Phi-3-medium-128k-instruct
datasets:
  - path: TheFuzzyScientist/squad-for-llms
    type:
      system_prompt: "Read the following context and concisely answer my question."
      field_system: system
      field_instruction: question
      field_input: context
      field_output: output
      format: "<|user|>
 {input} {instruction} </s>
<|assistant|>"
      no_input_format: "<|user|> {instruction} </s>
<|assistant|>"
output_dir: ./models/Phi3_Storyteller
sequence_length: 8172
bf16: auto
tf32: false
micro_batch_size: 4
num_epochs: 1
optimizer: adamw_bnb_8bit
learning_rate: 0.0002
logging_steps: 1
adapter: lora
lora_r: 32
lora_alpha: 16
lora_dropout: 0.05
lora_target_linear: true
gradient_accumulation_steps: 1
gradient_checkpointing: true
load_in_8bit: true
```

[Back to Top](#quick-navigation)

---

## Gemma 2-27B Configuration: `specialised_gemma.yml`

```yaml
base_model: unsloth/gemma-2-27b-it
datasets:
  - path: Yukang/LongAlpaca-12k
    type: alpaca
output_dir: ./models/gemma-LongAlpaca
sequence_length: 1024
bf16: auto
tf32: false
micro_batch_size: 1
num_epochs: 1
optimizer: adamw_bnb_8bit
learning_rate: 0.0002
logging_steps: 1
adapter: qlora
lora_r: 32
lora_alpha: 16
lora_dropout: 0.05
lora_target_linear: true
gradient_accumulation_steps: 1
gradient_checkpointing: true
load_in_8bit: false
load_in_4bit: true
flash_attention: true
```

[Back to Top](#quick-navigation)

---

## Colab Notebook

👉 [Open in Colab](https://colab.research.google.com/drive/1M2EJd20gCBta8f85zLb56ta0oMmn7TND?usp=sharing)  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1M2EJd20gCBta8f85zLb56ta0oMmn7TND?usp=sharing)

[Back to Top](#quick-navigation)

---

## References & Further Reading

- [Microsoft Phi-3 Medium Model Card](https://huggingface.co/microsoft/Phi-3-medium-128k-instruct)
- [Phi-3 Technical Report (Microsoft)](https://www.microsoft.com/en-us/research/project/phi-3/)
- [Quantization in Transformers (Hugging Face)](https://huggingface.co/docs/transformers/perf_train_gpu_one#quantization)
- [Gradient Checkpointing – PyTorch Docs](https://pytorch.org/docs/stable/checkpoint.html)
- [LoRA: Low-Rank Adaptation (arXiv)](https://arxiv.org/abs/2106.09685)
- [Axolotl Fine-Tuning Framework](https://github.com/OpenAccess-AI-Collective/axolotl)
- [Unsloth – Efficient Training](https://github.com/unslothai/unsloth)
- [Flash Attention 2 – Paper](https://arxiv.org/abs/2309.17453)

---

🗓 Generated on: July 27, 2025


[Back to Top](#quick-navigation)

---

## Task-Focused Training: Aim for Better Learning - Part 1

## 📌 Quick Navigation

- [Dataset Overview](#dataset-overview)
- [Data Preprocessing](#data-preprocessing)
- [Training Configuration](#training-configuration)
- [Loss Calculation Challenges](#loss-calculation-challenges)
- [Colab Notebook](#colab-notebook)
- [References & Further Reading](#references--further-reading)

---

## Dataset Overview

The lesson transitions from a simple dataset to the more complex **SQuAD** dataset, aiming to train a decoder-only large language model (LLM).  
SQuAD (Stanford Question Answering Dataset) consists of:

- **Context**: A passage of text
- **Question**: Related to the context
- **Answer**: A short, extractable string from the context

**Dataset Statistics**:
- **Training Set**: ~87,000 samples
- **Validation Set**: ~10,000 samples

🔗 [SQuAD on Hugging Face](https://huggingface.co/datasets/squad)

[Back to Top](#quick-navigation)

---

## Data Preprocessing

### 🔧 Tools Used
- Python
- Hugging Face Datasets
- Pandas
- Parquet file storage

### 🧩 Steps:
1. **Load** SQuAD using `datasets` library.
2. **Convert** to pandas DataFrame.
3. **Simplify Answers**: Use only the first answer string.
4. **Drop** unnecessary columns.
5. **Save** the processed dataset as a `.parquet` file.
6. **Update** the YAML configuration to use the local dataset.
7. **Adjust Prompts**:
   - Instruction: The question
   - Input: The context
   - Output: The answer
   - System prompt: “Read the following context and concisely answer my question.”

[Back to Top](#quick-navigation)

---

## Training Configuration

### 🧠 Model: `microsoft/Phi-3-medium-128k-instruct`

- Direction: Decoder-only
- Max Token Sequence: 8,172 (extended due to load_in_8bit optimization)
- Optimizer: `adamw_bnb_8bit`
- Precision: `bf16`, `8bit`, `tf32: false`

### 🛠️ LoRA Setup
- `adapter: lora`
- `r: 32`, `alpha: 16`, `dropout: 0.05`
- Target Linear Layers

### 🌀 YAML Config Snippet

```yaml
base_model: microsoft/Phi-3-medium-128k-instruct
datasets:
  - path: TheFuzzyScientist/squad-for-llms
    type: 
      system_prompt: "Read the following context and concisely answer my question."
      field_system: system
      field_instruction: question
      field_input: context
      field_output: output
      format: "<|user|>\n {input} {instruction} </s>\n<|assistant|>"
output_dir: ./models/Phi3_Storyteller
sequence_length: 8172
micro_batch_size: 4
learning_rate: 0.0002
adapter: lora
gradient_checkpointing: true
load_in_8bit: true

📄 [Model on Hugging Face](https://huggingface.co/microsoft/Phi-3-medium-128k-instruct)  
📜 [ArXiv Paper (if available)](https://arxiv.org/abs/2404.14219)  
[Back to Top](#quick-navigation)

---

## Task-Focused Training: Aim for Better Learning - Part 2


## 📌 Quick Navigation

- [Output-Focused Training](#output-focused-training)
- [Training Behavior Comparison](#training-behavior-comparison)
- [YAML Configuration Updates](#yaml-configuration-updates)
- [Colab Notebook](#colab-notebook)
- [References & Further Reading](#references--further-reading)

---

## Output-Focused Training

In this lesson, we explore how to modify our training setup so that the model is rewarded and punished based **only** on its ability to generate the output — not the input.

This is especially important for tasks like **question answering**, where:
- The **input (context + question)** is significantly longer than the **output (answer)**.
- The model may waste learning capacity modeling irrelevant parts of the input.

### ✅ Key Change:
- In the Axolotl framework, set:

  ```yaml
  train_on_inputs: false
  ```

This disables gradient calculation over the input portion of the sequence.

### 📌 When to Use
- Ideal for tasks with short, deterministic outputs (e.g., QA, summarization).
- **Not recommended** for conversational datasets or tasks with intertwined dialog.

[Back to Top](#quick-navigation)

---

## Training Behavior Comparison

Two identical models are trained:
- **Left**: `train_on_inputs = false` (loss computed only on output)
- **Right**: `train_on_inputs = true` (loss computed on full input + output)

### Observations:
- Early performance is similar due to SQuAD's simplicity.
- Models with `train_on_inputs = false` tend to:
  - Converge **faster**
  - Achieve **higher final accuracy**
  - Require **fewer steps**
  - Be **more stable** (less prone to divergence)

This adjustment improves focus and prevents overfitting to irrelevant tokens.

[Back to Top](#quick-navigation)

---

## YAML Configuration Updates

Below is the modified configuration snippet used for this lesson:

```yaml
base_model: microsoft/Phi-3-medium-128k-instruct
datasets:
  - path: TheFuzzyScientist/squad-for-llms
    type: 
      system_prompt: "Read the following context and concisely answer my question."
      field_system: system
      field_instruction: question
      field_input: context
      field_output: output
      format: "<|user|>
 {input} {instruction} </s>
<|assistant|>"
      no_input_format: "<|user|> {instruction} </s>
<|assistant|>"
train_on_inputs: false
output_dir: ./models/Phi3_Storyteller
sequence_length: 8172
bf16: auto
tf32: false
micro_batch_size: 4
num_epochs: 1
optimizer: adamw_bnb_8bit
learning_rate: 0.0002
adapter: lora
lora_r: 32
lora_alpha: 16
lora_dropout: 0.05
lora_target_linear: true
gradient_accumulation_steps: 1
gradient_checkpointing: true
load_in_8bit: true
```

[Back to Top](#quick-navigation)

---

## Colab Notebook

👉 [Open in Colab](https://colab.research.google.com/drive/1PN6MyU6WubUbRqjkq8mtBMmxRvjzHxxV?usp=sharing)  
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1PN6MyU6WubUbRqjkq8mtBMmxRvjzHxxV?usp=sharing)

[Back to Top](#quick-navigation)

---

## References & Further Reading

- [Phi-3 Model on Hugging Face](https://huggingface.co/microsoft/Phi-3-medium-128k-instruct)
- [ArXiv: Phi-3 Model Paper](https://arxiv.org/abs/2404.14219)
- [Training on Outputs Only - Hugging Face Discussion](https://discuss.huggingface.co/t/efficient-loss-masking/41053)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Transformers Docs – Hugging Face](https://huggingface.co/docs/transformers/index)
- [Axolotl GitHub](https://github.com/OpenAccess-AI-Collective/axolotl)
- [Google Colab Guide](https://research.google.com/colaboratory/)

[Back to Top](#quick-navigation)

[Back to Top](#quick-navigation)

---

## Edge of Hardware Limits: Scaling Inputs with Flash Attention 2


## 📌 Quick Navigation

- [Scaling Model Training](#scaling-model-training)
- [Flash Attention: Theory](#flash-attention-theory)
- [Implementing Flash Attention](#implementing-flash-attention)
- [Long Context Dataset Training](#long-context-dataset-training)
- [Colab Notebook](#colab-notebook)
- [References & Further Reading](#references--further-reading)

---

## Scaling Model Training

In this section, the focus shifts to pushing the boundaries of what is possible in training large language models (LLMs) **without upgrading hardware**.

Key goals:
- Train on longer sequences
- Reduce GPU memory footprint
- Improve training throughput and stability

Challenges:
- Transformer attention is **quadratic** in complexity
- Long sequences demand more compute and memory

[Back to Top](#quick-navigation)

---

## Flash Attention: Theory

Flash Attention is a highly efficient attention mechanism designed to optimize:
- Memory transfers
- Computation efficiency

### 🧠 Key Optimizations:
- Minimizes memory I/O by loading queries, keys, and values once
- Operates on GPU SRAM instead of relying on frequent memory access
- Enables longer sequence training with same hardware

### 🔍 Outcomes:
- Up to **1GB memory saved**
- Up to **12% training speed improvement**
- Higher max sequence lengths and batch sizes achievable

📘 Flash Attention Paper: [ArXiv: 2205.14135](https://arxiv.org/abs/2205.14135)

[Back to Top](#quick-navigation)

---

## Implementing Flash Attention

In the `Axolotl` framework, enabling Flash Attention is very simple:

```yaml
flash_attention: true
```

Once enabled, the attention mechanism becomes more memory-efficient without any change to model architecture or tokenization.

🛠️ Other Configuration Highlights:
- Base Model: `unsloth/gemma-2-27b-it`
- Adapter: `qLoRA`
- Mixed precision: `bf16`, `load_in_4bit`

```yaml
# model params
base_model: unsloth/gemma-2-27b-it

# dataset params
datasets:
  - path: Yukang/LongAlpaca-12k
    type: alpaca

output_dir: ./models/gemma-LongAlpaca

sequence_length: 1024
flash_attention: true
adapter: qlora
load_in_4bit: true
```

[Back to Top](#quick-navigation)

---

## Long Context Dataset Training

To test Flash Attention and longer contexts, the model was fine-tuned using:

🗂️ **Dataset**: `LongAlpaca-12k`  
🧾 **Type**: Instruction-following conversations with long inputs and outputs  
🔢 **Max Sequence Length**: 16,000 tokens (extended from 8k)  
📦 **Micro Batch Size**: 1  
🧪 **Padding**: Applied to ensure all sequences are uniform  
📉 **Observation**:
- 5% lower memory usage
- 10–12% training speedup

[Back to Top](#quick-navigation)

---

## Colab Notebook

👉 [Open in Colab](https://colab.research.google.com/drive/1CM_tNCws5wqW6Mmww7N94FQ0K7E-yOXT?usp=sharing)  
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1CM_tNCws5wqW6Mmww7N94FQ0K7E-yOXT?usp=sharing)

[Back to Top](#quick-navigation)

---

## References & Further Reading

- [Flash Attention (ArXiv)](https://arxiv.org/abs/2205.14135)
- [LongAlpaca Dataset – Hugging Face](https://huggingface.co/datasets/Yukang/LongAlpaca-12k)
- [Gemma-2-27B Model – Hugging Face](https://huggingface.co/unsloth/gemma-2-27b-it)
- [Axolotl Fine-Tuning GitHub](https://github.com/OpenAccess-AI-Collective/axolotl)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of LLMs](https://arxiv.org/abs/2305.14314)
- [Transformers Documentation – Hugging Face](https://huggingface.co/docs/transformers/index)

 

[Back to Top](#quick-navigation)

---

## Edge of Hardware Limits: Reaching 4bit Training with QLoRA

## 📌 Quick Navigation

- [Overview of 4-bit and qLoRA Training](#overview-of-4-bit-and-qlora-training)
- [Memory Efficiency via Quantization](#memory-efficiency-via-quantization)
- [Gemma-2 27B Model Training](#gemma-2-27b-model-training)
- [YAML Configuration](#yaml-configuration)
- [Colab Notebook](#colab-notebook)
- [References & Further Reading](#references--further-reading)

---

## Overview of 4-bit and qLoRA Training

As we push the boundaries of training ever-larger models on limited hardware, the next evolution involves precision optimization:

### 🔍 Topics Covered:
- **4-bit training** via double quantization
- **qLoRA** (quantized Low-Rank Adaptation)
- Using `Gemma-2-27B` on a 24GB GPU

These techniques allow us to maximize model size without scaling hardware further.

[Back to Top](#quick-navigation)

---

## Memory Efficiency via Quantization

### 📉 4-bit Quantization
- Reduces model weights from 16/32 bits to **4 bits**
- Uses **double quantization**: 8-bit quantization followed by 4-bit compression
- Implements **NF4 (Normalized Float)** for high precision retention
- Memory savings allow fitting massive models on smaller GPUs

### 🧠 qLoRA
- A variant of LoRA integrating quantization into adapter training
- Propagates gradients only through **low-rank matrices**, while freezing the backbone
- Enables training models like `Gemma-2-27B` efficiently

🚀 Outcome:
- Fits **27B parameters** on a single **24GB GPU**
- Matches or exceeds performance of full-precision fine-tuning

[Back to Top](#quick-navigation)

---

## Gemma-2 27B Model Training

We moved from Microsoft's Phi-3 to Google's `Gemma-2-27B` for long-instruction fine-tuning.

### 🧾 Training Setup:
- **Model**: `unsloth/gemma-2-27b-it`
- **Dataset**: `LongAlpaca-12k`
- **Precision**: 4-bit (`load_in_4bit: true`)
- **Adapter**: `qlora`
- **Flash Attention**: Enabled
- **Hardware**: Single 24GB GPU

### ⚠️ Challenges:
- Training failed at 8-bit due to OOM
- 4-bit loading + qLoRA enabled training to proceed
- Achieved full convergence and speed on low-resource setup

[Back to Top](#quick-navigation)

---

## YAML Configuration

```yaml
# model params
base_model: unsloth/gemma-2-27b-it

# dataset params
datasets:
  - path: Yukang/LongAlpaca-12k
    type: alpaca

output_dir: ./models/gemma-LongAlpaca

# training setup
sequence_length: 1024
bf16: auto
tf32: false
micro_batch_size: 1
num_epochs: 1
optimizer: adamw_bnb_8bit
learning_rate: 0.0002
logging_steps: 1

# adapter and quantization
adapter: qlora
lora_r: 32
lora_alpha: 16
lora_dropout: 0.05
lora_target_linear: true
load_in_8bit: false
load_in_4bit: true
gradient_accumulation_steps: 1
gradient_checkpointing: true
flash_attention: true
```

[Back to Top](#quick-navigation)

---

## Colab Notebook

👉 [Open in Colab](https://colab.research.google.com/drive/1Nf1R7g1fFejNiKVF-e5FRMqlohYHtJJ4?usp=sharing)  
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Nf1R7g1fFejNiKVF-e5FRMqlohYHtJJ4?usp=sharing)

[Back to Top](#quick-navigation)

---

## References & Further Reading

- [Gemma 2-27B on Hugging Face](https://huggingface.co/unsloth/gemma-2-27b-it)
- [LongAlpaca Dataset – Hugging Face](https://huggingface.co/datasets/Yukang/LongAlpaca-12k)
- [QLoRA: Efficient Finetuning of LLMs](https://arxiv.org/abs/2305.14314)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [NF4 Quantization Paper](https://arxiv.org/abs/2309.02369)
- [Transformers Documentation – Hugging Face](https://huggingface.co/docs/transformers/index)
- [Flash Attention (ArXiv)](https://arxiv.org/abs/2205.14135)
- [Axolotl Fine-Tuning GitHub](https://github.com/OpenAccess-AI-Collective/axolotl)

[Back to Top](#quick-navigation)


---
⬅️ **Previous:** [Advanced LLM Training](10-llm-advanced.md) | ➡️ **Next:** [Final Deployment](12-llm-deployment.md)  
 