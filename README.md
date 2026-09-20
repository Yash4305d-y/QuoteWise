# ✍️ QuoteWise

> **AI-powered quote generation using LSTM-based language modeling**

QuoteWise is a deep learning project that learns word patterns and sequential language structures from a dataset of quotes and generates new quote-like text from a user-provided seed phrase.

The project compares a **SimpleRNN baseline** with an **LSTM-based language model**, using tokenization, sequence generation, word embeddings, and temperature-controlled sampling to produce varied text.

---

## 🚀 Features

* 🧠 **LSTM-based text generation**
* 🔄 **SimpleRNN baseline** for comparison
* 🔤 Text tokenization and vocabulary creation
* 🧩 Sequential training-data generation
* 📐 Pre-padding of variable-length sequences
* 🔡 Word embeddings for language representation
* 🌡️ Temperature-controlled text sampling
* 💾 Saved trained model and tokenizer
* ✍️ Generate quote-like text from custom seed phrases

---

## 🧠 How It Works

QuoteWise follows a next-word prediction pipeline:

```text
Quote Dataset
      ↓
Text Cleaning
      ↓
Tokenization
      ↓
Convert Text → Integer Sequences
      ↓
Create Input/Target Word Pairs
      ↓
Sequence Padding
      ↓
Word Embedding
      ↓
LSTM / SimpleRNN
      ↓
Next-Word Prediction
      ↓
Temperature Sampling
      ↓
Generated Quote
```

For example:

```text
Seed:
Life is

Generated:
Life is ... [model-generated continuation]
```

The model predicts one word at a time and feeds the predicted sequence back into the model to continue generating text.

---

## 🏗️ Model Architecture

### SimpleRNN Baseline

```text
Input Sequence
      ↓
Embedding
      ↓
SimpleRNN (128 units)
      ↓
Dense (10,000 units)
      ↓
Softmax
```

### LSTM Model

```text
Input Sequence
      ↓
Embedding (50 dimensions)
      ↓
LSTM (128 units)
      ↓
Dense (10,000 units)
      ↓
Softmax
```

The LSTM model is used as the primary text-generation model because LSTM networks are designed to retain relevant information across longer sequences.

---

## 🛠️ Tech Stack

| Technology       | Purpose                        |
| ---------------- | ------------------------------ |
| Python           | Core programming               |
| TensorFlow       | Deep learning framework        |
| Keras            | Model development              |
| NumPy            | Numerical computation          |
| Pandas           | Dataset handling               |
| Matplotlib       | Visualization                  |
| Seaborn          | Data visualization             |
| Pickle           | Saving tokenizer/configuration |
| Jupyter Notebook | Development environment        |

---

## 📊 Dataset Processing

The quote dataset is loaded from:

```text
qoute_dataset.csv
```

The quote text is processed by:

1. Converting text to lowercase
2. Removing punctuation
3. Building a vocabulary using Keras `Tokenizer`
4. Converting quotes into integer sequences
5. Creating multiple training examples from each sequence

For a sequence such as:

```text
life is a beautiful journey
```

the training samples become conceptually:

```text
life                  → is
life is               → a
life is a             → beautiful
life is a beautiful   → journey
```

This converts the problem into a **next-word prediction task**.

---

## 🔤 Tokenization

QuoteWise uses the Keras `Tokenizer` with a vocabulary size of:

```text
10,000 words
```

Each word is mapped to an integer index.

The resulting sequences are padded to a maximum length of:

```text
100 tokens
```

using pre-padding so that every training sample has a consistent input shape.

---

## 🧩 Training Configuration

### SimpleRNN

```text
Embedding Dimension : 50
RNN Units           : 128
Vocabulary Size     : 10,000
Maximum Sequence    : 100
Optimizer           : Adam
Loss                : Sparse Categorical Crossentropy
Batch Size          : 128
Epochs              : 10
```

### LSTM

```text
Embedding Dimension : 50
LSTM Units          : 128
Vocabulary Size     : 10,000
Maximum Sequence    : 100
Optimizer           : Adam
Loss                : Sparse Categorical Crossentropy
Batch Size          : 128
Maximum Epochs      : 100
Early Stopping      : Enabled
Model Checkpoint    : Enabled
Validation Split    : 10%
```

The LSTM model uses **EarlyStopping** with restoration of the best weights and **ModelCheckpoint** to save the best-performing model according to validation loss.

---

## 🌡️ Temperature Sampling

QuoteWise uses temperature-based sampling instead of always selecting the highest-probability word.

The temperature controls the randomness of generated text:

```text
Lower Temperature
        ↓
More predictable output

Higher Temperature
        ↓
More diverse / random output
```

For example, the generation function can use:

```python
temperature=0.7
```

to produce relatively controlled but non-deterministic text.

---

## ✍️ Generate Text

QuoteWise provides a reusable generation function:

```python
generate_quote(
    lstm_model,
    tokenizer,
    max_len,
    seed_text="Life is",
    next_words=15,
    temperature=0.7
)
```

Example:

```text
Seed:
Life is

↓
LSTM prediction
↓
Sample next word
↓
Append word
↓
Predict again
↓
Repeat
```

This allows users to experiment with different seed phrases, generation lengths, and temperatures.

---

## 💾 Saved Artifacts

The trained model and preprocessing configuration are saved for later inference:

```text
lstm_quote_model.h5
tokenizer.pkl
config.pkl
```

### `lstm_quote_model.h5`

Contains the trained LSTM model.

### `tokenizer.pkl`

Stores the fitted tokenizer and vocabulary mapping.

### `config.pkl`

Stores configuration required during inference, including:

```python
{
    "max_len": 100
}
```

Keeping the tokenizer and padding configuration alongside the model ensures that inference uses the same preprocessing setup as training.

---

## 📁 Project Structure

```text
QuoteWise/
│
├── qoute_dataset.csv
├── QuoteWise.ipynb
│
├── lstm_quote_model.h5
├── best_quote_lstm.keras
│
├── tokenizer.pkl
├── config.pkl
│
└── README.md
```

> File names can be adjusted depending on the final notebook/repository structure.

---

## 🔬 What I Learned

Through this project, I explored:

* Natural Language Processing fundamentals
* Tokenization and vocabulary construction
* Sequence-based training data generation
* Word embeddings
* Recurrent Neural Networks
* LSTM architecture
* Next-word prediction
* Sequence padding
* Temperature-based sampling
* Model checkpointing
* Early stopping
* Saving and reusing trained NLP models

---


---

## ⚠️ Limitations

QuoteWise is an experimental next-word language-generation project.

The model learns statistical patterns from the training dataset rather than understanding the meaning of quotes like a human.

Generated text may therefore contain:

* Repetitive phrases
* Grammatically inconsistent sentences
* Unusual word combinations
* Incomplete or nonsensical continuations

The quality of generated text is also strongly dependent on the size and diversity of the training dataset.

---

## 👨‍💻 Author

**Yash Vinay Kalyani**

B.Tech CSE — Guru Ghasidas Vishwavidyalaya

Interested in **AI/ML, Deep Learning, NLP, and intelligent applications**.

---

## ⭐ Project Goal

> **Learn language patterns → predict the next word → generate new quote-like text.**

QuoteWise demonstrates how recurrent neural networks can be trained to learn sequential language patterns and generate text one word at a time.
