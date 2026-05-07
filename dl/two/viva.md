# DL Practical — Binary Classification Viva Guide
### IMDB Movie Review Sentiment Analysis using Deep Neural Network

---

> **Strategy:** Read question → cover answer → say it out loud.
> If you can say it WITHOUT reading → you're ready. Move on.
> Focus sections marked ⭐ are most likely to be asked.

---

## SECTION 1 — Dataset & Problem

---

### ⭐ Q. What is the IMDB dataset?

The IMDB dataset is a benchmark **binary text classification dataset** built into Keras. It contains **50,000 movie reviews** from the Internet Movie Database — 25,000 for training and 25,000 for testing. Each review is labeled as either **Positive (1)** or **Negative (0)**. The dataset is perfectly balanced — exactly 12,500 positive and 12,500 negative reviews in both train and test sets. The reviews are already preprocessed into sequences of integer word indices, where each integer represents a specific word in a vocabulary ranked by frequency.

---

### ⭐ Q. What is the problem type here?

This is a **supervised binary classification** problem. Supervised because we have labeled data (each review has a known sentiment label). Binary because there are exactly two output classes: Positive (1) or Negative (0). The goal is to train a model that reads a sequence of word indices representing a review and predicts whether the overall sentiment is positive or negative.

---

### Q. Why is the dataset represented as sequences of integers?

Neural networks cannot process raw text — they only work with numbers. The IMDB dataset preprocesses text by building a vocabulary of the most frequent words and assigning each word a unique integer index ranked by frequency (word with rank 1 = most frequent). So a review like `"this film was brilliant"` becomes `[14, 22, 43, 973]`. We keep only the top `VOCAB_SIZE = 10,000` words; rarer words are replaced by an unknown token (index 2).

---

### Q. What does vocab_size = 10,000 mean?

We only consider the 10,000 most frequently occurring words in the entire IMDB corpus. Words ranked beyond 10,000 (very rare, often typos or obscure terms) are discarded and mapped to the unknown token. This reduces vocabulary size (less memory), removes noise, and focuses learning on words that actually appear enough times to learn meaningful patterns.

---

## SECTION 2 — Preprocessing

---

### ⭐ Q. What is padding and why do we need it?

Neural networks require **fixed-size inputs** — every sample in a batch must have the same shape. But IMDB reviews have wildly varying lengths — training reviews average 239 words but some have up to 2,494 words. We use `pad_sequences` to make every review exactly `MAXLEN = 250` words:
- Reviews **shorter than 250** → zeros appended at the END (`padding='post'`)
- Reviews **longer than 250** → words removed from the END (`truncating='post'`)

Result: every review becomes a tensor of shape `(250,)` and a batch of 512 reviews becomes shape `(512, 250)`.

---

### Q. Why MAXLEN = 250?

Average training review length is 238.71 words, and average test review length is 230.80 words. Setting MAXLEN = 250 captures the full content of the majority of reviews (roughly 75–80% of reviews are under 250 words) without wasting memory on excessively long sequences. It's a balance between coverage and efficiency.

---

### Q. What is padding='post' vs padding='pre'?

- `padding='post'` → zeros added at the **end** of the sequence: `[1, 14, 22, 0, 0, 0]`
- `padding='pre'`  → zeros added at the **start**: `[0, 0, 0, 1, 14, 22]`

Post-padding is generally preferred for this architecture because the model reads the actual review content first, then encounters padding — the meaningful signal is at the front. Pre-padding can sometimes cause issues with Flatten-based models as the zeros precede the real content.

---

### ⭐ Q. What is an Embedding layer and why is it the first layer?

An **Embedding layer** converts integer word indices into dense real-valued vectors. Instead of representing word 14 as the number 14 (which has no semantic meaning), the embedding layer gives word 14 a unique learned vector of size `EMBEDDING_DIM = 16`.

```
Word index 14  →  [0.23, -0.15, 0.87, ..., 0.04]  (16 values)
Word index 22  →  [0.11,  0.92, -0.33, ..., 0.71]  (16 values)
```

Words with similar meanings (e.g., "good", "great", "excellent") end up with similar vectors after training. The embedding layer learns these representations during training — it's essentially a trainable lookup table. Input shape: `(batch, 250)` → Output shape: `(batch, 250, 16)`.

---

### Q. What does Flatten() do after the Embedding?

After embedding, each review is a 2D matrix of shape `(250, 16)` — 250 words, each with a 16-dimensional vector. But Dense layers expect 1D input per sample. `Flatten()` reshapes this to a 1D vector of size `250 × 16 = 4,000`. So the full batch goes from `(batch, 250, 16)` → `(batch, 4000)`. This is a simple but effective approach — the alternative would be using LSTM or Conv1D layers to handle the sequence properly.

---

## SECTION 3 — Model Architecture

---

### ⭐ Q. Explain your complete model architecture

```
Input            :  Integer sequence of length 250
Embedding        :  (10000 vocab) → 16-dim vectors → shape (250, 16)
Flatten          :  (250, 16) → (4000,)
Dense(64, ReLU)  :  First hidden layer — learns complex feature combinations
BatchNorm        :  Normalizes activations for stable training
Dropout(0.4)     :  Drops 40% of neurons — prevents overfitting
Dense(32, ReLU)  :  Second hidden layer
BatchNorm        :  
Dropout(0.3)     :  Drops 30% of neurons
Dense(16, ReLU)  :  Third hidden layer — compact representation
Dense(1, Sigmoid):  Output — probability of being Positive (0 to 1)
```

Total parameters ≈ 285,000+ (mostly from Embedding: 10,000 × 16 = 160,000).

---

### ⭐ Q. Why Sigmoid for the output layer?

For binary classification, the output must be a **probability between 0 and 1**. Sigmoid function: `σ(x) = 1 / (1 + e^(-x))` always outputs a value in (0, 1). We interpret this as the probability the review is Positive. We then threshold at 0.5: if output ≥ 0.5 → Positive, else → Negative. Sigmoid is specifically designed for binary output — unlike ReLU (unbounded) or tanh (outputs -1 to 1, not a probability).

---

### ⭐ Q. Why Binary Cross-Entropy loss (not MSE)?

Binary Cross-Entropy: `L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]`

For classification with a Sigmoid output, binary cross-entropy is the correct loss function because:
1. It's derived from maximum likelihood estimation for a Bernoulli distribution
2. It penalizes **confident wrong predictions** extremely heavily (log(≈0) → very large loss)
3. Gradients flow better than MSE through Sigmoid — MSE + Sigmoid causes vanishing gradients
4. It directly measures the difference between predicted probability distribution and true labels

**If external asks about MSE for classification:** MSE would work mathematically but converges much slower and doesn't account for the probabilistic nature of Sigmoid output.

---

### Q. Why ReLU in hidden layers?

ReLU: `f(x) = max(0, x)` — outputs 0 for negative inputs, x for positive.
- **No vanishing gradient**: gradient is exactly 1 for positive inputs
- **Fast computation**: just a max operation
- **Sparse activation**: inactive neurons output 0, creating efficient representations
- Works well for deep networks

We use ReLU for hidden layers and Sigmoid only for the output. Mixing them intentionally: hidden layers need non-linearity to learn complex patterns; output needs bounded probability.

---

### Q. What is BatchNormalization and why did you add it?

BatchNorm normalizes activations of a layer to have **zero mean and unit variance** within each mini-batch:
`x_norm = (x - batch_mean) / (batch_std + ε)`

Then applies learned scale (γ) and shift (β) parameters. Benefits:
1. **Stable training** — prevents internal covariate shift (distribution of inputs keeps shifting during training)
2. **Allows higher learning rates** — reduces sensitivity to initialization
3. **Mild regularization** — adds slight noise through batch statistics, reducing overfitting
4. **Faster convergence** — network doesn't need to compensate for unnormalized inputs

Your friend's notebook didn't use BatchNorm in hidden layers — adding it is an improvement.

---

### ⭐ Q. What is Dropout? Why use different rates (0.4 and 0.3)?

Dropout **randomly sets a fraction of neuron outputs to zero** during each training step. `Dropout(0.4)` means 40% of the 64 neurons are randomly disabled each forward pass — the network must learn without relying on any single neuron.

- Higher dropout (0.4) on the larger first layer (64 neurons) → more regularization where parameters are most numerous
- Lower dropout (0.3) on smaller second layer (32 neurons) → less aggressive, smaller layer needs less regularization
- No dropout on final hidden layer (16 neurons) → preserve the compact learned representation
- **At test time**: all neurons are active, but outputs are scaled by `(1 - dropout_rate)` to maintain expected values

---

### Q. What is the Adam optimizer?

Adam (Adaptive Moment Estimation) with `learning_rate=0.001`:
- Maintains **per-parameter adaptive learning rates** based on gradient history
- Combines **Momentum** (accelerates in consistent directions) and **RMSprop** (scales by recent gradient magnitude)
- Works excellently with default hyperparameters — `lr=0.001` is the standard
- Converges faster than plain SGD, especially for NLP tasks with sparse gradients (most words appear infrequently)

---

## SECTION 4 — Training & Evaluation

---

### ⭐ Q. What were your results?

| Metric | Value |
|--------|-------|
| Test Accuracy | ~84–86% |
| Test Loss | ~0.35–0.44 |
| AUC-ROC | ~0.92+ |

The model correctly classifies approximately 84–86% of the 25,000 test reviews. This is strong performance for a simple Embedding + Dense architecture — more advanced architectures like LSTM or BERT can push this to 90–95%.

---

### ⭐ Q. What is the Confusion Matrix and what does it tell you?

The confusion matrix breaks predictions into 4 categories:

|  | Predicted Negative | Predicted Positive |
|--|---|---|
| **Actually Negative** | True Negative (TN) | False Positive (FP) |
| **Actually Positive** | False Negative (FN) | True Positive (TP) |

For IMDB with ~85% accuracy on 25,000 test samples:
- ~10,500 TN (correct negative predictions)
- ~10,750 TP (correct positive predictions)  
- ~1,250 FP (wrongly called positive)
- ~1,500 FN (wrongly called negative)

It shows not just overall accuracy but WHERE the model makes errors — useful for understanding class-specific performance.

---

### Q. What is Precision, Recall, and F1-Score?

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Precision** | TP / (TP + FP) | Of all reviews called Positive, how many truly are? |
| **Recall** | TP / (TP + FN) | Of all truly Positive reviews, how many did we catch? |
| **F1-Score** | 2 × (P × R) / (P + R) | Harmonic mean of Precision and Recall |
| **Accuracy** | (TP+TN) / Total | Overall correct predictions |

For balanced datasets like IMDB (50/50 split), accuracy is a fair metric. For imbalanced datasets, F1 is more informative.

---

### ⭐ Q. What is the ROC Curve and AUC?

**ROC Curve** (Receiver Operating Characteristic): plots **True Positive Rate vs False Positive Rate** at every possible classification threshold (0 to 1).

- A perfect classifier → AUC = 1.0 (curve hugs top-left corner)
- A random classifier → AUC = 0.5 (diagonal line)
- Our model → AUC ≈ 0.92+

**AUC** (Area Under the Curve): single number summarizing overall classifier performance across ALL thresholds. Higher = better. AUC > 0.9 is considered excellent. It measures the model's ability to **rank** positive reviews above negative ones, regardless of threshold.

---

### ⭐ Q. What is overfitting and is your model overfitting?

**Overfitting** = model memorizes training data, fails on new data. Signs: training accuracy keeps rising while validation accuracy plateaus or drops; training loss keeps falling while validation loss rises.

Looking at training curves:
- Training accuracy: reaches ~99% by epoch 10
- Validation accuracy: plateaus around ~84–86% after epoch 3-4
- Validation loss starts rising after epoch 2-3

Yes, there is **mild overfitting** — the gap between training and validation accuracy indicates the model is starting to memorize. That's why we added `Dropout` and `EarlyStopping(patience=3)` to stop before it worsens significantly.

---

### Q. What is EarlyStopping?

```python
EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
```

Monitors validation loss every epoch. If it doesn't improve for 3 consecutive epochs (patience=3), training stops. `restore_best_weights=True` resets the model to the epoch with the lowest val_loss — not the last epoch (which may have overfit). This prevents training unnecessarily long and saves the best version of the model.

---

### Q. What is an epoch and a batch?

- **Epoch**: one complete pass through all 25,000 training samples
- **Batch (batch_size=512)**: training data is split into chunks of 512 reviews. Weights update after each chunk. With 25,000 samples and batch_size=512: `ceil(25000/512) = 49 batches per epoch`.
- **Why batch_size=512 (large)?** Larger batches → smoother gradient estimates → faster convergence per epoch. Suitable here because IMDB reviews are relatively simple for the model to learn from.

---

## SECTION 5 — Conceptual / Tricky Questions

---

### ⭐ Q. What is sentiment analysis? Where is it used in real life?

Sentiment analysis (opinion mining) is the task of identifying the emotional tone of text. Real applications:
- Product reviews (Amazon, Flipkart) — are customers satisfied?
- Social media monitoring (Twitter) — brand perception tracking
- Financial news analysis — positive/negative news affecting stock prices
- Customer service — automatically prioritizing negative feedback
- Movie/restaurant reviews — aggregate sentiment scoring

---

### ⭐ Q. What is an Embedding layer vs one-hot encoding?

| | One-Hot Encoding | Embedding |
|--|--|--|
| Representation | Binary vector of size 10,000 | Dense vector of size 16 |
| Semantic info | None — all words equally distant | Similar words → similar vectors |
| Memory | 10,000 values per word | 16 values per word |
| Trainable | No | Yes — learns during training |
| Example | "good" = [0,0,...,1,...,0] | "good" = [0.23, -0.15, ..., 0.04] |

Embedding is far more efficient and captures semantic relationships. The embedding matrix is `10,000 × 16 = 160,000` learnable parameters.

---

### Q. What is the difference between binary and multiclass classification in terms of output layer?

| | Binary Classification | Multiclass Classification |
|--|--|--|
| Output neurons | 1 | N (number of classes) |
| Activation | Sigmoid | Softmax |
| Loss function | Binary Cross-Entropy | Categorical Cross-Entropy |
| Output range | (0, 1) — one probability | (0, 1) summing to 1 across N |
| Threshold | 0.5 for positive | argmax of output vector |

---

### Q. What is Softmax? Why not use it here?

Softmax converts a vector of raw scores to probabilities summing to 1. For binary classification (2 classes), Softmax with 2 outputs is mathematically equivalent to Sigmoid with 1 output. Using Sigmoid with 1 output neuron is simpler, uses half the parameters, and is the standard convention for binary classification. Softmax is used when there are 3+ classes.

---

### Q. Why not use an LSTM or Conv1D instead of Flatten?

`Flatten` after Embedding loses **word order information** — it treats the 250×16 matrix as a bag of embedded words. LSTMs and Conv1D would capture sequential patterns (word order matters for meaning: "not good" vs "good not"). However:
- Flatten + Dense is simpler, trains faster, uses less memory
- For sentiment analysis specifically, individual word semantics often suffice (positive/negative words are strong signals regardless of position)
- LSTM would improve accuracy further (~2-5%) at the cost of much more complex training

Your architecture is a valid starting point and demonstrates the concept clearly.

---

### Q. What would you do to improve accuracy further?

1. **Use LSTM or Bidirectional LSTM** — captures word order and context
2. **Use Conv1D** — learns local n-gram patterns (phrase-level features)
3. **Increase EMBEDDING_DIM** — from 16 to 64 or 128 for richer representations
4. **Use pre-trained embeddings** — GloVe or Word2Vec instead of training from scratch
5. **Use transformers/BERT** — state of the art for NLP, but complex to implement
6. **Increase VOCAB_SIZE** — from 10k to 20k to include more words
7. **Tune MAXLEN** — longer sequences capture more review content

---

### ⭐ Q. What would happen if we used linear activation on the output instead of Sigmoid?

The output could be any real number (e.g., 5.3 or -2.1) — not a probability. We couldn't threshold at 0.5 meaningfully. Binary cross-entropy loss would also break (it requires inputs in (0,1) for log computation — log of a negative number is undefined). For regression this is fine, but for classification we need bounded probability outputs — hence Sigmoid.

---

### Q. Why is the dataset perfectly balanced and why does it matter?

IMDB was curated to have exactly 12,500 positive and 12,500 negative reviews. Balance matters because:
- Accuracy is a meaningful metric — a model predicting "Positive" for everything would get 50% accuracy (obvious baseline to beat)
- The model doesn't develop bias toward one class
- Precision and Recall are balanced — no need to adjust class weights

For imbalanced datasets (e.g., fraud detection with 99% non-fraud), accuracy becomes misleading and techniques like class weighting, oversampling (SMOTE), or using F1/AUC are needed.

---

## SECTION 6 — Quick Revision Flashcards

| Question | One-Line Answer |
|----------|----------------|
| Dataset? | IMDB — 50,000 movie reviews, 25k train, 25k test |
| Problem type? | Supervised binary classification |
| Labels? | 0 = Negative, 1 = Positive |
| Dataset balance? | Perfectly balanced — 50% each |
| VOCAB_SIZE = 10000? | Keep top 10k most frequent words only |
| Why padding? | Neural nets need fixed-size inputs |
| MAXLEN = 250? | Covers avg review length (238 words) |
| padding='post'? | Zeros added at end of short sequences |
| What is Embedding? | Converts word index → dense learned vector |
| Embedding output shape? | (batch, 250, 16) |
| Why Flatten after Embedding? | Dense layer needs 1D input per sample |
| What does Flatten produce? | (batch, 250×16) = (batch, 4000) |
| Architecture? | Embed → Flatten → Dense(64) → Dense(32) → Dense(16) → Dense(1) |
| Hidden activation? | ReLU |
| Output activation? | Sigmoid (binary probability) |
| Loss function? | Binary Cross-Entropy |
| Optimizer? | Adam, lr=0.001 |
| Why Sigmoid output? | Outputs probability in (0,1) for binary class |
| Why Binary CE not MSE? | CE penalizes confident errors, no vanishing gradient |
| Dropout(0.4)? | Randomly drops 40% neurons — prevents overfitting |
| BatchNorm purpose? | Normalize layer activations — stable training |
| EarlyStopping patience=3? | Stop after 3 epochs of no val_loss improvement |
| restore_best_weights? | Revert to best epoch, not final overfit epoch |
| Test accuracy? | ~84–86% |
| AUC? | ~0.92+ |
| What is ROC curve? | TPR vs FPR at all thresholds |
| What is AUC? | Area under ROC — 1.0 = perfect, 0.5 = random |
| Confusion matrix? | TP/TN/FP/FN breakdown of predictions |
| Precision? | TP / (TP + FP) — accuracy of positive predictions |
| Recall? | TP / (TP + FN) — coverage of actual positives |
| Why dataset balanced matters? | Accuracy is a fair metric; no class bias |
| Overfitting signs? | Train acc >> Val acc, Val loss rises |
| LSTM vs Flatten? | LSTM captures word order; Flatten treats as bag of embeddings |
| Word index offset by 3? | Indices 0,1,2 reserved for padding/start/unknown |

---

*Key message: Words → Integers → Embedded Vectors → Dense Layers → Sigmoid → Sentiment.*
*The Embedding layer is the key innovation — it learns that "brilliant" and "excellent" mean similar things.*
*Model achieves ~85% accuracy. Overfitting visible after epoch 3 — controlled by Dropout + EarlyStopping.*