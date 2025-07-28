# 📌 Quick Navigation

- [Course Overview](#course-overview)
- [Part 1: NLP & Transformer Fundamentals](#part-1-nlp--transformer-fundamentals)
- [Part 2: Practical LLM Applications](#part-2-practical-llm-applications)
- [Model Comparison Summary](#model-comparison-summary)
- [Key Takeaways](#key-takeaways)
- [References & Further Exploration](#references--further-exploration)

---

## Course Overview

This course is a hands-on introduction to transformer-based language models, combining theoretical foundations with practical implementations. The curriculum covers BERT, GPT, and T5 models, including their use in real-world NLP tasks.

---

## Part 1: NLP & Transformer Fundamentals

### Historical Phases of NLP

![NLP Evolution](https://jalammar.github.io/images/nlp-timeline.png)

- **Rule-Based Systems**: Manually defined linguistic rules
- **Statistical Methods**: Word co-occurrence and probabilistic models
- **Machine Learning**: Feature-based methods (e.g., SVM, Naive Bayes)
- **Deep Learning**: Dense vector embeddings and neural models

### Core Transformer Concepts

![Transformer Architecture](https://jalammar.github.io/images/the-transformer-architecture.png)

- **Attention Mechanism**: Enables global contextual representation
- **Tokenization**: Breaks text into subwords with positional info
- **Encoder-Decoder**: Structure used in models like T5, BART
- **Fine-tuning**: Adjusts pretrained models for specific tasks

🔗 Reference: [Illustrated Transformer – Jay Alammar](https://jalammar.github.io/illustrated-transformer/)

---

## Part 2: Practical LLM Applications

### 🟢 BERT – Extractive Question Answering

![BERT Architecture](https://miro.medium.com/v2/resize:fit:720/format:webp/1*V4IWf-J3zVHZAXzqDEtPHw.png)

- Extracts an answer span from context using start and end logits
- Ideal for closed-domain QA
- Handles context chunks using stride

👉 [Open in Colab](https://colab.research.google.com/drive/156gcOZbUQwXfiINzlzD_AIkt1Llfa-O1?usp=sharing)  
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/156gcOZbUQwXfiINzlzD_AIkt1Llfa-O1?usp=sharing)

🔗 Model: [bert-base-uncased](https://huggingface.co/bert-base-uncased)  
📄 Paper: [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)

---

### 🔵 GPT – Instruction-Following Generation

![GPT Architecture](https://jalammar.github.io/images/gpt2-architecture.jpg)

- Trained using causal language modeling
- Uses instruction + response prompts
- Fine-tuned with Open-Instruct dataset

👉 [Open in Colab](https://colab.research.google.com/drive/1OUHnyQevDJA1p_tDDUqfWdxKfwRCz1Xt?usp=sharing)  
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OUHnyQevDJA1p_tDDUqfWdxKfwRCz1Xt?usp=sharing)

🔗 Model: [DiabloGPT on Hugging Face](https://huggingface.co/TheFuzzyScientist/diabloGPT_open-instruct)  
📄 Paper: [GPT-2](https://openai.com/research/language-unsupervised)  
📄 Dataset: [Open-Instruct](https://huggingface.co/datasets/open-instruct)

---

### 🔴 T5 – Text-to-Text Product Review Generation

![T5 Architecture](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/t5-arch.png)

- Treats all tasks as text-to-text (e.g., `summarize:` or `translate:`)
- Pretrained on C4 corpus with span corruption
- Ideal for summarization, QA, translation, and generation

👉 [Open in Colab](https://colab.research.google.com/drive/1EqyVW8tmnCrGKIb67fIOiyyR5gbmdFhy?usp=sharing)  
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1EqyVW8tmnCrGKIb67fIOiyyR5gbmdFhy?usp=sharing)

🔗 Model: [T5-base](https://huggingface.co/t5-base), [Amazon Review Model](https://huggingface.co/TheFuzzyScientist/T5-base_Amazon-product-reviews)  
📄 Paper: [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)

---

## Model Comparison Summary

| Model      | Architecture     | Directionality       | Pretraining Task         | Ideal Use Cases                          | Limitations              |
|------------|------------------|----------------------|---------------------------|-------------------------------------------|--------------------------|
| **BERT**   | Encoder-only     | Bidirectional         | Masked Language Modeling  | QA, classification, embeddings            | 512-token limit          |
| **GPT-2**  | Decoder-only     | Unidirectional        | Causal Language Modeling  | Instruction generation, chatbots         | No bidirectional context |
| **T5**     | Encoder-Decoder  | Bi/Uni (input/output) | Span corruption (text-to-text) | Summarization, QA, translation        | Needs task-specific prompt |
| **Gemini** | Multi-modal      | Flexible              | MoE + RLHF + VLM          | Multimodal generation, reasoning          | Closed-source            |

---

## Key Takeaways

- Leverage pretrained models to reduce time and cost
- Use token chunking and stride for input limits
- Even small models like GPT-2 perform well when fine-tuned
- T5’s text-to-text design enables flexibility across tasks

---

## References & Further Exploration

### 🧠 Foundational Papers

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [GPT-2](https://openai.com/research/language-unsupervised)
- [T5](https://arxiv.org/abs/1910.10683)

### 🤗 Hugging Face Models

- [bert-base-uncased](https://huggingface.co/bert-base-uncased)
- [gpt2](https://huggingface.co/gpt2)
- [t5-base](https://huggingface.co/t5-base)
- [DiabloGPT](https://huggingface.co/TheFuzzyScientist/diabloGPT_open-instruct)
- [Amazon T5 Review Model](https://huggingface.co/TheFuzzyScientist/T5-base_Amazon-product-reviews)

### 🧪 Colab Notebooks

- [BERT QA](https://colab.research.google.com/drive/156gcOZbUQwXfiINzlzD_AIkt1Llfa-O1?usp=sharing)
- [GPT Instruction Tuning](https://colab.research.google.com/drive/1OUHnyQevDJA1p_tDDUqfWdxKfwRCz1Xt?usp=sharing)
- [T5 Product Review Generator](https://colab.research.google.com/drive/1EqyVW8tmnCrGKIb67fIOiyyR5gbmdFhy?usp=sharing)

---

---

⬅️ **Previous:** [Using Transformers](06-using-transformers.md) | ➡️ **Next:** [Intro to LLMs](08-llm-intro.md)  
 