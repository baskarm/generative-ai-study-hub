# 📌 Quick Navigation

- [Course Overview](#course-overview)
- [Key Concepts](#key-concepts)
- [Tokenizer and Embeddings](#tokenizer-and-embeddings)
- [Masked Language Modeling](#masked-language-modeling)
- [Semantic Search Engine](#semantic-search-engine)
- [Model Evolution Table](#model-evolution-table)

---

## Course Overview

This section focuses on transitioning from theoretical knowledge of transformer models to their practical implementation and engineering components, emphasizing real-world applications such as semantic search and embedding usage.

- Prepares learners to apply transformer embeddings for NLP tasks
- Covers tokenization, embeddings, model internals, and downstream tasks
- Includes practical hands-on coding with Hugging Face Transformers and PyTorch

[Back to Top](#quick-navigation)

---

## Key Concepts

### Transformer Engineering Focus

- **Embeddings**: Represent words/sentences as dense vectors for downstream processing
- **Tokenization**: Converts raw text to token IDs; includes handling special tokens
- **Attention Mechanism**: Key to contextual representation in transformers
- **Model Inputs**: Includes token IDs, attention masks, and token type IDs
- **Sentence Transformers**: Fine-tuned models for capturing sentence-level semantics

[Back to Top](#quick-navigation)

---

## Tokenizer and Embeddings

### Tokenization Pipeline

- Tokenizers split sentences into subword tokens
- Maintains a vocabulary of ~30k+ tokens
- Returns token IDs, attention masks, and token type IDs
- Important to use model-specific tokenizers for consistency

## Try It Yourself

Explore and run the notebook interactively using Google Colab:

[Open the Notebook in Colab](https://colab.research.google.com/drive/1wg_2Q9UkVDOhL7g9bdB6CvE_ww1UZ8WM?usp=sharing)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1wg_2Q9UkVDOhL7g9bdB6CvE_ww1UZ8WM?usp=sharing)

### Embeddings

- Token IDs are converted to high-dimensional vectors
- Two key outputs:
  - **Last Hidden State**: Embeddings for individual tokens (shape: seq_len × hidden_dim)
  - **Pooled Output**: Embedding for the entire sequence, used in classification


## Try It Yourself

You can run and explore the notebook directly in Google Colab:

[Open the Notebook in Colab](https://colab.research.google.com/drive/1bQLidcWx-dj8SH1bCOfzB6wUpl2bso4l?usp=sharing)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bQLidcWx-dj8SH1bCOfzB6wUpl2bso4l?usp=sharing)

[Back to Top](#quick-navigation)

### Semantic Distance

- Embeddings compared using cosine similarity
- Allows words with different meanings (e.g., "fly") to be distinguished contextually







---

## Masked Language Modeling

- Pretraining task for models like BERT
- Random tokens replaced with `[MASK]` and predicted by the model
- Output logits converted to probabilities via softmax
- Used to help the model build a strong language understanding foundation

### Example

- Input: `"I want to [MASK] pizza for tonight"`
- Output: `"have"`, `"get"`, `"eat"`, `"make"`, `"order"` as top predictions

## Try It Yourself

You can experiment with the code by opening the notebook in Google Colab:

[Open the Notebook in Colab](https://colab.research.google.com/drive/1XMpEYowE4RsupsSN3m0lw4Vn1G-TgGgD?usp=sharing)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XMpEYowE4RsupsSN3m0lw4Vn1G-TgGgD?usp=sharing)

[Back to Top](#quick-navigation)

---

## Semantic Search Engine

### Goal

Build a semantic search engine that finds the most relevant document to a query based on meaning, not keyword match.

### Tools & Dataset

- **Dataset**: Multi-News (2000 article summaries)
- **Model**: SentenceTransformer for lightweight sentence embeddings (384-dim)
- **Libraries**: Hugging Face Transformers, PyTorch, Pandas

### Process

- Embed all documents once
- Embed user’s query
- Compute cosine similarity between query and all document embeddings
- Retrieve top-k relevant results using `torch.topk`

### Example Queries

- "Artificial Intelligence": returned AI-related articles
- "Natural Disasters": returned disaster-related summaries
- "Law Enforcement", "Politics": worked as expected

## Try It Yourself

Give it a try by opening the interactive Google Colab notebook below:

[Open the Notebook in Colab](https://colab.research.google.com/drive/1nYoIMe3JcRfVoO-VEEOE_iWLvfVXIbir?usp=sharing)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nYoIMe3JcRfVoO-VEEOE_iWLvfVXIbir?usp=sharing)

[Back to Top](#quick-navigation)

---

## Model Evolution Table

| Model Name | Category       | Architecture     | Strengths                            | Ideal Use Cases                          | Latest Version Info |
|------------|----------------|------------------|--------------------------------------|-------------------------------------------|---------------------|
| BERT       | Encoder-only   | Transformer      | Bidirectional context, strong understanding | Text classification, Q&A, embedding generation | BERT-Base / BERT-Large |
| GPT        | Decoder-only   | Transformer      | Text generation, instruction following | Chatbots, creative writing, code generation | GPT-4o (June 2024)  |
| T5         | Encoder-Decoder| Transformer      | Unified text-to-text architecture    | Translation, summarization, Q&A           | T5.1.1, Flan-T5     |
| Gemini     | Multi-modal    | Transformer + Vision + Memory | Text + image processing, powerful LLM+VLM hybrid | Multi-modal tasks, agentic reasoning | Gemini 1.5 (June 2025) |
| SentenceTransformer | Encoder-only   | Siamese / Bi-encoder Transformer | Sentence similarity, semantic search | Embedding generation, retrieval, clustering | `all-MiniLM-L6-v2`  |

[Back to Top](#quick-navigation)

## References & Further Exploration

- 🤗 **Hugging Face Models and Tools**
  - [BERT (bert-base-uncased)](https://huggingface.co/bert-base-uncased)
  - [GPT-2 (gpt2)](https://huggingface.co/gpt2)
  - [T5 (t5-base)](https://huggingface.co/t5-base)
  - [SentenceTransformer (all-MiniLM-L6-v2)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
  - [T5-based Amazon Product Review Generator by TheFuzzyScientist](https://huggingface.co/TheFuzzyScientist/T5-base_Amazon-product-reviews)

- 📓 **Colab Notebooks (Used in This Module)**
  - [Tokenizer & Embeddings Colab](https://colab.research.google.com/drive/1bQLidcWx-dj8SH1bCOfzB6wUpl2bso4l?usp=sharing)
  - [Masked Language Modeling (MLM) Demo](https://colab.research.google.com/drive/1XMpEYowE4RsupsSN3m0lw4Vn1G-TgGgD?usp=sharing)
  - [Semantic Search with Transformers](https://colab.research.google.com/drive/1nYoIMe3JcRfVoO-VEEOE_iWLvfVXIbir?usp=sharing)
  - [Tokenizer Pipeline Walkthrough](https://colab.research.google.com/drive/1wg_2Q9UkVDOhL7g9bdB6CvE_ww1UZ8WM?usp=sharing)

- 📚 **Further Reading**
  - [Attention Is All You Need (Vaswani et al.)](https://arxiv.org/abs/1706.03762)
  - [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/)
  - [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)

➡️ [Back to Top](#quick-navigation)

---

⬅️ **Previous:** [Popular Transformer Models](05-popular-transformer-models.md) | ➡️ **Next:** [Real World Scenario with LLMs](07-real-world-scenario-llm.md)  
 