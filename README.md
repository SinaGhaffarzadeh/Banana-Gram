# Character-Level Bigram Language Model with PyTorch

This project implements a **character-level Bigram Language Model** using PyTorch. It is trained on text data from *Dorothy and the Wizard in Oz* by L. Frank Baum (sourced from [Project Gutenberg](https://www.gutenberg.org/)).

## 📘 What is an N-gram Language Model?

An **N-gram language model** is a probabilistic model used to predict the next item (typically a word or character) in a sequence based on the previous N-1 items.

### 🔹 Types of N-gram Models

- **Unigram**: Considers only the current item (`P(w1), P(w2), ...`)
- **Bigram**: Considers the current and previous item (`P(w2|w1), P(w3|w2), ...`)
- **Trigram**: Considers the current and two previous items (`P(w3|w1, w2), ...`)
- **n-gram**: Extends this to any "n" previous items (`P(wn|wn-1, ..., wn-n+1)`)

## ✨ What is a Bigram Model?

A **Bigram model** learns the probability of a token (word or character) given the immediately preceding token. In the case of characters:

Given the sequence `I love cats`, the bigrams are:
```
I →   l  
l →   o  
o →   v  
v →   e  
e →   (space)  
(space) →   c  
...
```

The model learns how often each character follows another and uses this to generate text.

## 🧠 About This Implementation

This is a character-level Bigram Language Model implemented in PyTorch. Instead of words, we tokenize individual characters.

### Key Features

- Character-level tokenization using integer encoding.
- Vocabulary is built from unique characters in the text.
- A neural model using PyTorch’s `nn.Embedding` to learn bigram probabilities.
- Text generation using sampling from the learned distribution.

### Model Overview

- Input: Sequence of character indices
- Output: Predicted probability distribution for the next character
- Architecture: Single `nn.Embedding` layer for simplicity
- Loss: Cross-entropy loss
- Optimizer: AdamW

## 📂 Dataset

The training data comes from the book:

> **Dorothy and the Wizard in Oz** by L. Frank Baum  
> Source: [Project Gutenberg](https://www.gutenberg.org/)

The text is read from a file called `Wizard_of_Oz.txt`.

## 🛠 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/SinaGhaffarzadeh/Banana
cd bigram-language-model
```

### 2. Install Dependencies

```bash
pip install torch matplotlib numpy pylzma ipykernel jupyter
```

> ✅ For CUDA-enabled machines, make sure you install the correct version of PyTorch from [https://pytorch.org](https://pytorch.org/)

### 3. Prepare Dataset

Download the plain text version of the book from [Project Gutenberg](https://www.gutenberg.org/ebooks/15389) and rename it as:

```
Wizard_of_Oz.txt
```

Place the file in the root directory of your project.


## 📈 Training

- The model trains using randomly sampled character sequences of length 8.
- Every `eval_iters` (default: 250) steps, training and validation loss are reported.

Sample training log:
```
Step 0 and the losses {'train': 3.941, 'val': 3.955}
Step 250 and the losses {'train': 2.891, 'val': 2.947}
...
```

(It's gibberish—but statistically plausible gibberish!)

## 🚀 Future Improvements

- Add support for word-level tokenization
- Expand to Trigram or higher-order models
- Integrate a Transformer decoder block
- Add dataset preprocessing (cleaning, lowercasing, etc.)

