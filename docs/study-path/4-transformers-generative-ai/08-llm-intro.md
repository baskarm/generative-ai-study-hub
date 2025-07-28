# 📌 Quick Navigation

- [Course Overview](#course-overview)
- [What is a Large Language Model?](#what-is-a-large-language-model)
- [Decoder-Only Architecture](#decoder-only-architecture)
- [Chat Templates & Structured Inputs](#chat-templates--structured-inputs)
- [Model Selection on Hugging Face](#model-selection-on-hugging-face)
- [Code Demonstration: TinyLlama](#code-demonstration-tinyllama)
- [References & Further Reading](#references--further-reading)

---

## Course Overview

This section explores Large Language Models (LLMs) built on the transformer architecture, their training procedures, deployment challenges, and how they are applied in real-world interactive systems. The goal is to bridge conceptual understanding with hands-on implementation.

- Core Focus:
  - Decoder-only transformer models
  - Tokenization & input formatting
  - Reinforcement Learning from Human Feedback (RLHF)
  - Chat templates
  - Model selection and generation parameters

[Back to Top](#-quick-navigation)

---

## What is a Large Language Model?

LLMs refer to powerful NLP models capable of generating complex, human-like text. They’re built using **decoder-only transformer** architectures and trained at scale using massive datasets.

- **Scale**:
  - Models like LLaMA-3 and GPT-4 have up to 70+ billion parameters.
  - Small LLMs (e.g., 2–7B) are optimized for consumer hardware.
- **Architecture**:
  - Modern LLMs are generally **decoder-only** models.
- **Capabilities**:
  - High factual recall
  - Scalable deployment
  - Robust contextual understanding
- **Challenges**:
  - Hallucination
  - Deployment complexity
  - High compute requirements

🧠 Despite limitations, they’ve surpassed average human factual knowledge.

[Back to Top](#-quick-navigation)

---

## Decoder-Only Architecture

### Key Properties

- LLMs process inputs as a **single concatenated sequence**.
- Interaction is simulated using **autoregession**, where the model predicts the next token.
- Requires clever input formatting to mimic input-output behavior.

### Fine-tuning Techniques

- **Supervised Fine-Tuning**: Uses input-response pairs to guide expected outputs.
- **Reinforcement Learning from Human Feedback (RLHF)**:
  - Multiple responses are generated.
  - Human annotators **rank responses**.
  - Used to improve contextual accuracy and helpfulness.

📊 **Illustration**:

![Decoder-only Architecture](https://jalammar.github.io/images/gpt2/gpt2-large-transformer.png)  
*Source: Jay Alammar’s GPT2 visual guide*

### Applications:

- Chatbots  
- Code generation  
- Instruction following  
- Document summarization

[Back to Top](#-quick-navigation)

---

## Chat Templates & Structured Inputs

LLMs simulate dialogue using **chat templates** that structure user-assistant messages.

### Input Structure

- A "conversation" is a series of messages with:
  - `role`: Identifies speaker (user/assistant)
  - `content`: Message text

### Model-specific Template Examples

| Model       | Structure Type     | Special Tokens | Role Awareness | Instruction Capable |
|-------------|--------------------|----------------|----------------|---------------------|
| Blenderbot  | Basic concat       | ❌             | ❌             | ❌                  |
| Mistral     | Instruction tokens | ✅             | ⚠️ (Partial)   | ✅                  |
| Gemma       | Turn-based format  | ✅✅           | ✅             | ✅✅                |
| LLaMA 3     | Header tokens      | ✅✅✅         | ✅             | ✅✅✅              |

🧩 These templates are essential during fine-tuning to teach models interaction patterns.

[Back to Top](#-quick-navigation)

---

## Model Selection on Hugging Face

🛠️ Choosing the right LLM impacts performance, cost, and resource needs.

### What to Look For:

- **Model Family**: LLaMA, Mistral, Phi, Gemma, etc.
- **Size (Parameters)**:
  - Small: 2B–7B
  - Medium: 13B–34B
  - Large: 70B+
- **Instruction-Following**:
  - Look for `instruct` or `chat` variants
- **Context Length**:
  - Defined via `max_position_embeddings` in `config.json`
  - Affects how much prompt+response can be handled

💡 **Hugging Face Model Hub**:  
🔗 [Browse Models](https://huggingface.co/models)

[Back to Top](#-quick-navigation)

---

## Code Demonstration: TinyLlama

👉 [Open in Colab](https://colab.research.google.com/drive/1gpMMTuwRR1PDJhgca7_Qui-AqruTYUvL?usp=sharing)  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1gpMMTuwRR1PDJhgca7_Qui-AqruTYUvL?usp=sharing)

We explore TinyLlama to demonstrate basic generation and parameter tuning.

### Workflow

- Load model + tokenizer  
- Prepare chat messages using templates  
- Encode as tokens  
- Generate response  
- Decode and analyze output  

### Key Generation Parameters

| Parameter         | Description                                      |
|-------------------|--------------------------------------------------|
| `max_new_tokens`  | Limits length of generated response              |
| `temperature`     | Controls creativity/randomness (higher = more)   |
| `top_p`           | Nucleus sampling: restricts to top % of prob.    |
| `do_sample`       | Enables randomness in output                     |

🟢 **Temperature Examples**:

- 1.0 → Creative, varied responses  
- 0.1 → Deterministic, factual outputs  

📎 Prompt token count impacts total input length (important for context fitting).

[Back to Top](#-quick-navigation)

---

## References & Further Reading

- [📜 Attention Is All You Need (Vaswani et al.)](https://arxiv.org/abs/1706.03762)
- [🤗 Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [🖼️ Jay Alammar’s Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [🧠 LLaMA 3 on Hugging Face](https://huggingface.co/meta-llama)
- [📘 RLHF Explained – Hugging Face Blog](https://huggingface.co/blog/rlhf)
- [📄 OpenAI: ChatGPT Fine-tuning Guide](https://platform.openai.com/docs/guides/fine-tuning)
- [📚 Gemma Tokenizer Guide – Google](https://ai.google.dev/gemma/docs/tokenization)
- [🔬 Microsoft Phi Models on Hugging Face](https://huggingface.co/microsoft)

[Back to Top](#-quick-navigation)

---

⬅️ **Previous:** [Real World Scenario with LLMs](07-real-world-scenario-llm.md) | ➡️ **Next:** [Preparing LLMs](09-llm-prep.md)  
 