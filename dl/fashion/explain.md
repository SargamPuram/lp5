# 🎓 Viva Prep: Fashion MNIST Image Classification
### Convolutional Neural Network (CNN)
> **Exam ready in 15 mins — focus on bold Q&As**

---

## 🧠 What is This Experiment About?

You built a **CNN to classify clothing images into 10 categories** using the Fashion MNIST dataset.  
This is a **Multi-Class Image Classification** problem using Computer Vision + Deep Learning.

---

## 📦 The Dataset

**Q: What is Fashion MNIST?**
> A dataset of **70,000 grayscale images** of clothing items — 60,000 training + 10,000 test.  
> Each image is **28×28 pixels**, belonging to one of **10 classes**.  
> It's a drop-in replacement for the original MNIST (handwritten digits), but harder.

**Q: What are the 10 classes?**
| Index | Class |
|-------|-------|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

**Q: Is the dataset balanced?**
> Yes — perfectly balanced. Each class has exactly **6,000 training images**.

**Q: What does the raw data look like?**
> x_train shape: **(60000, 28, 28)** — 60K images, 28×28 pixels  
> y_train shape: **(60000,)** — integer labels 0–9

---

## ⚙️ Preprocessing Steps

**Q: What preprocessing was done?**
Three steps:

**1. Reshape (add channel dimension)**
> CNN expects 4D input: `(batch, height, width, channels)`  
> Reshaped from `(60000, 28, 28)` → `(60000, 28, 28, 1)`  
> Channel = 1 because images are **grayscale** (color images would be 3 for RGB)

**2. Normalize pixel values**
> Divided by 255.0 → pixel values go from **[0–255]** to **[0.0–1.0]**  
> Why? Neural networks train faster and better on small, uniform-scale inputs.

**3. One-Hot Encode labels**
> Label `9` → `[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]`  
> Why? Output layer has 10 neurons (softmax) — needs vector targets, not integer labels.  
> y_train_cat shape: **(60000, 10)**

**Q: Why do we add a channel dimension?**
> Conv2D layers need to know the number of channels to apply filters correctly. Grayscale = 1 channel.

---

## 🏗️ Model Architecture

**Q: What was your CNN architecture?**
```
Input: (28, 28, 1)
    ↓
Conv2D(32, kernel_size=3×3, activation='relu', padding='same')
    → Output: (28, 28, 32)
MaxPooling2D(pool_size=2×2)
    → Output: (14, 14, 32)
    ↓
Conv2D(64, kernel_size=3×3, activation='relu', padding='same')
    → Output: (14, 14, 64)
MaxPooling2D(pool_size=2×2)
    → Output: (7, 7, 64)
    ↓
Flatten()
    → Output: (3136,)   [7×7×64 = 3136]
    ↓
Dense(128, activation='relu')
Dropout(0.5)
    ↓
Dense(10, activation='softmax')  ← 10-class output
```

**Total parameters: 421,642**

**Q: What is a Convolutional layer (Conv2D)?**
> Applies learnable **filters** (kernels) that slide over the image to detect features like edges, shapes, textures.  
> Each filter produces a **feature map**.  
> First Conv: 32 filters → learns 32 different features  
> Second Conv: 64 filters → learns more complex features

**Q: What is a kernel/filter?**
> A small matrix (3×3 here) of learnable weights that performs element-wise multiplication with image patches.  
> Detects patterns — early layers detect edges, deeper layers detect shapes/objects.

**Q: What is MaxPooling2D?**
> Reduces spatial dimensions by taking the **maximum value** in each 2×2 window.  
> 28×28 → 14×14 after first pooling. Benefits: reduces computation, adds translation invariance, prevents overfitting.

**Q: What is padding='same'?**
> Adds zero-padding around the input so the output has the **same spatial dimensions** as input.  
> Without it (padding='valid'), the output shrinks after each conv layer.

**Q: Why Flatten() before Dense?**
> Conv layers output 3D feature maps `(7, 7, 64)`. Dense layers need 1D input.  
> Flatten reshapes `(7, 7, 64)` → `(3136,)`.

**Q: What is Dropout(0.5)?**
> Randomly sets 50% of neurons to zero during training.  
> Forces the network not to rely on any single neuron → **prevents overfitting**.  
> Only active during training, disabled during inference.

**Q: Why softmax on output layer?**
> For **multi-class classification** (10 classes). Softmax converts raw scores into probabilities that sum to 1.  
> The class with the highest probability is the prediction.  
> Binary → Sigmoid. Multi-class → Softmax.

**Q: What loss function was used?**
> **Categorical Crossentropy** — used when labels are one-hot encoded.  
> (If labels were integers, you'd use `sparse_categorical_crossentropy`)

---

## 📊 Results

**Q: What were the final results?**
| Metric | Value |
|--------|-------|
| Test Loss | 0.2200 |
| **Test Accuracy** | **92.31%** |

**Q: What did the training curves show?**
- Training accuracy: **~94.45%** at epoch 15
- Validation accuracy: **~92.31%** — very close to training accuracy
- Both loss curves converge smoothly
- **Minimal overfitting** — validation accuracy closely tracks training accuracy

**Q: Is there overfitting here?**
> Minimal. The gap between training (94%) and validation (92%) is small. The Dropout(0.5) layer worked effectively.  
> Compare this to P2 (IMDB) where training was 99% vs validation 84% — CNN generalized much better here.

---

## ✅ Is the Experiment Correct? Will External Like It?

### ✅ What's Great:
- Correct reshape to 4D for CNN ✔️
- Normalization to [0,1] ✔️
- One-hot encoding with `to_categorical` ✔️
- Two Conv+Pool blocks — standard CNN structure ✔️
- Dropout used for regularization ✔️
- Correct loss (categorical crossentropy) + activation (softmax) for multi-class ✔️
- **92.31% accuracy** — excellent for this architecture ✔️
- Training curves show healthy learning with minimal overfitting ✔️
- Good EDA: sample images displayed, label distribution verified ✔️

### ⚠️ Points External May Push On:
- **Why 2 conv blocks and not more?** — 2 blocks is sufficient for 28×28 images; more would reduce spatial dims too much.
- **Why no Dropout after Conv layers?** — Common practice; Dropout commented out after Conv blocks in your code.
- **What is the receptive field?** — After 2 MaxPoolings, each neuron "sees" a large portion of the original image.
- **How is CNN better than a plain DNN for images?** — CNNs are translation-invariant, parameter-efficient (weight sharing), and preserve spatial structure.

### 🏆 Verdict:
> **Excellent experiment. External will be very impressed.** 92.31% accuracy with minimal overfitting on a 10-class problem is strong. Architecture is clean and follows industry-standard CNN design.

---

## 💡 Quick Concept Questions

**Q: What is CNN? How is it different from DNN?**
> CNN = Convolutional Neural Network. Uses Conv layers that exploit **spatial structure** of images.  
> DNN treats every pixel independently (no spatial awareness). CNN shares weights via filters — far fewer parameters for image tasks.

**Q: What is feature extraction in CNN?**
> Early layers extract low-level features (edges, corners). Later layers combine them into high-level features (shapes, objects). This hierarchy is learned automatically.

**Q: What is translation invariance?**
> CNN can recognize a shoe whether it's in the top-left or bottom-right of the image — because pooling removes exact position information.

**Q: What is the difference between sigmoid, softmax, and ReLU?**
| Activation | Used For | Output Range |
|-----------|----------|-------------|
| ReLU | Hidden layers | [0, ∞) |
| Sigmoid | Binary classification output | (0, 1) |
| Softmax | Multi-class output | (0,1), sums to 1 |

**Q: What is the difference between categorical and sparse categorical crossentropy?**
> - `categorical_crossentropy` → labels are **one-hot encoded** (used here)
> - `sparse_categorical_crossentropy` → labels are **integers** (no one-hot needed)

**Q: Why 32 filters in first Conv and 64 in second?**
> Common practice — increase filters as spatial size decreases. More filters = richer feature representations at deeper layers.

**Q: What is weight sharing in CNN?**
> The same filter (same weights) is applied across the entire image. This drastically reduces parameters compared to a fully connected layer — and makes the model generalize better.

---

## ⚔️ All 3 Practicals — Quick Comparison

| Aspect | P1: Boston Housing | P2: IMDB Sentiment | P3: Fashion MNIST |
|--------|-------------------|-------------------|------------------|
| Task | Regression | Binary Classification | Multi-Class Classification |
| Input | Tabular (13 features) | Text (sequences) | Images (28×28) |
| Key Layer | Dense | Embedding | Conv2D |
| Output activation | None (linear) | Sigmoid | Softmax |
| Loss | MSE | Binary Crossentropy | Categorical Crossentropy |
| Metric | R², MSE, MAE | Accuracy | Accuracy |
| Result | R²=0.842 (DNN) | 84.58% | **92.31%** |
| Overfitting? | Moderate | Yes (high) | Minimal |

---

## 🚀 One-Line Summary to Impress the External

> *"We built a 2-block CNN with Conv2D, MaxPooling, and Dropout layers to classify Fashion MNIST into 10 clothing categories. The model achieved 92.31% test accuracy with minimal overfitting, demonstrating that CNNs are highly effective for image classification due to weight sharing and spatial feature extraction."*

---

*That's all 3 done! Go sleep, you're ready. Crush it tomorrow! 🚀*