# 🎓 Viva Prep: IMDB Sentiment Analysis
### Text Classification using Deep Neural Network (Embedding + DNN)
> **Exam ready in 15 mins — focus on bold Q&As**

---

## 🧠 What is This Experiment About?

You built a **DNN to classify movie reviews as Positive (1) or Negative (0)** using the IMDB dataset.  
This is a **Binary Text Classification** problem using NLP + Deep Learning.

---

## 📦 The Dataset

**Q: What is the IMDB dataset?**
> A standard NLP benchmark dataset of **50,000 movie reviews** from IMDb — 25,000 for training and 25,000 for testing. Each review is labeled as **Positive (1)** or **Negative (0)**.

**Q: Is the dataset balanced?**
> Yes — perfectly balanced. Train set: **12,500 negative + 12,500 positive**. Test set same. External loves balanced datasets!

**Q: How is the data stored internally?**
> Reviews are pre-encoded as **sequences of integers** — each integer is an index into a vocabulary dictionary (word → index). The raw text is NOT stored, just word indices.

**Q: What were the review lengths?**
- Average training review: **~239 words**
- Average test review: **~231 words**
- Max training review: **2494 words**

---

## ⚙️ Preprocessing Steps

**Q: What parameters did you set?**
| Parameter | Value | Why |
|-----------|-------|-----|
| vocab_size | 10,000 | Keep only top 10K most frequent words |
| maxlen | 250 | Pad/truncate all reviews to 250 tokens |
| embedding_dim | 16 | Each word → 16-dimensional vector |
| dense_units | 16 | Hidden layer size |
| epochs | 10 | Training cycles |
| batch_size | 512 | Samples per gradient update |

**Q: What is vocab_size = 10,000?**
> Only the 10,000 most frequent words are kept. Rare words are replaced with an "unknown" token. This reduces noise and memory.

**Q: What is padding? Why is it needed?**
> Neural networks require **fixed-size inputs**. But reviews have variable lengths.  
> `pad_sequences` adds **zeros** at the end (post-padding) to make every review exactly **250 tokens** long.  
> If a review is longer than 250, it's **truncated** (from the end).

**Q: What does the padded data look like?**
> Shape = **(25000, 250)** — 25,000 reviews, each exactly 250 integers long.

**Q: Why offset word indices by 3?**
> Indices 0, 1, 2 are reserved for: `0=padding`, `1=start token`, `2=unknown word`. So actual word indices start from 3.

---

## 🏗️ Model Architecture

**Q: What was your model architecture?**
```
Input: integer sequences of length 250
    ↓
Embedding(vocab_size=10000, output_dim=16, input_length=250)
    ↓  [output shape: (batch, 250, 16)]
Flatten()
    ↓  [output shape: (batch, 4000)]
Dense(16, activation='relu')
    ↓
Dense(1, activation='sigmoid')  ← Binary classification output
```

**Q: What is an Embedding layer?**
> Converts each integer (word index) into a dense vector of fixed size (here, 16 dimensions).  
> It learns meaningful word representations during training — similar words get similar vectors.  
> Example: index `42` → `[0.2, -0.1, 0.8, ..., 0.3]` (16 numbers)

**Q: Why Embedding instead of one-hot encoding?**
> One-hot: vocab_size=10000 → each word is a 10000-dimensional sparse vector. Wasteful!  
> Embedding: each word → compact 16-dim dense vector. **Learns semantic meaning.** Much more efficient.

**Q: What does Flatten() do here?**
> After Embedding, shape is `(batch, 250, 16)` — a 3D tensor.  
> Flatten converts it to `(batch, 4000)` — a 2D tensor that Dense layers can process.

**Q: Why sigmoid on the output layer?**
> Sigmoid outputs a value between 0 and 1 — perfect for **binary classification**.  
> Output > 0.5 → Positive, Output ≤ 0.5 → Negative.

**Q: What loss function was used and why?**
> **Binary Crossentropy** — standard loss for binary classification problems.  
> It measures how far the predicted probability is from the true label (0 or 1).

**Q: What optimizer was used?**
> **Adam** with learning_rate=0.001. Adaptive optimizer — adjusts learning rates automatically per parameter.

---

## 📊 Results

**Q: What were the final results?**
| Metric | Value |
|--------|-------|
| Test Loss | 0.4434 |
| **Test Accuracy** | **84.58%** |

**Q: What did the training curves show?**
- Training accuracy reached nearly **99.78%** by epoch 10
- Validation accuracy plateaued around **84-85%**
- Training loss kept dropping but validation loss started **increasing after epoch 2-3**
- This gap is a sign of **overfitting**

**Q: Is there overfitting in this model?**
> **Yes, clearly.** Training accuracy ~99% vs validation accuracy ~84%. The model memorized training data.  
> Solutions: Add **Dropout layers**, reduce model complexity, or use regularization.

---

## ✅ Is the Experiment Correct? Will External Like It?

### ✅ What's Good:
- Proper use of IMDB dataset with `imdb.load_data` ✔️
- Balanced dataset confirmed via EDA ✔️
- Padding done correctly with shape (25000, 250) ✔️
- Embedding layer used — shows NLP understanding ✔️
- Correct loss (binary crossentropy) + activation (sigmoid) for binary classification ✔️
- Visualizations: label distribution, review length histograms, training curves ✔️
- Test accuracy of **84.58%** is solid for this architecture ✔️

### ⚠️ Points External May Push On:
- **Why no Dropout?** — Model clearly overfits. Dropout(0.5) after Dense would help.
- **Why only 10 epochs?** — Valid to say: controlled overfitting, quick experiment.
- **Why Flatten instead of GlobalAveragePooling?** — Flatten works but GlobalAveragePooling1D is more standard for text and less prone to overfitting.
- **Can you improve accuracy?** — Yes: LSTM/GRU layers, more epochs with early stopping, larger embedding dim.

### 🏆 Verdict:
> **Yes, experiment is correct.** 84.58% accuracy on a balanced binary classification task is respectable. External will appreciate the clear EDA, proper NLP pipeline (integer encoding → padding → embedding), and training visualization.

---

## 💡 Quick Concept Questions

**Q: What is Sentiment Analysis?**
> NLP task of determining the emotional tone (positive/negative/neutral) of text.

**Q: What is tokenization?**
> Splitting text into individual words/tokens. In IMDB, already done — words replaced by integer indices.

**Q: What is the difference between padding and truncation?**
- **Padding**: short sequences get zeros added to reach maxlen
- **Truncation**: long sequences get cut to maxlen

**Q: What is binary cross-entropy?**
> Loss = `-[y*log(p) + (1-y)*log(1-p)]` where y is true label, p is predicted probability.

**Q: What is the vanishing gradient problem?**
> In deep networks, gradients become very small during backpropagation through many layers — weights stop updating. ReLU helps mitigate this.

**Q: Why use batch_size=512?**
> Larger batch = faster training per epoch (parallelism). IMDB is large enough that 512 is efficient.

**Q: What is the difference between this and RNN/LSTM for text?**
> This DNN uses Embedding + Flatten — treats all words as a **bag of features**, ignoring word order.  
> RNN/LSTM captures **sequential order** of words, which is better for language understanding but slower.

**Q: What would you add to improve this model?**
> 1. Dropout layers (prevent overfitting)
> 2. LSTM/GRU (capture word order)
> 3. Early stopping (stop when val_loss stops improving)
> 4. Larger embedding_dim (e.g., 32 or 64)

---

## ⚔️ Comparison: Practical 1 vs Practical 2

| Aspect | Boston Housing (P1) | IMDB Sentiment (P2) |
|--------|--------------------|--------------------|
| Task | Regression | Classification |
| Output | Continuous price | Binary (0/1) |
| Output activation | None (linear) | Sigmoid |
| Loss | MSE | Binary Crossentropy |
| Metric | R², MSE, MAE | Accuracy |
| Input | Tabular numbers | Text (word sequences) |
| Key layer | Dense | Embedding |

---

## 🚀 One-Line Summary to Impress the External

> *"We used an Embedding layer to convert word indices into dense vectors, flattened them, and passed through Dense layers with sigmoid output to classify IMDB reviews. The model achieved 84.58% test accuracy, with visible overfitting suggesting Dropout or LSTM would further improve performance."*

---

*You're almost done! Last stretch — go crush it! 💪*