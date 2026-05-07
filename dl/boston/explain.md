# 🎓 Viva Prep: Boston Housing Price Prediction
### Linear Regression vs Deep Neural Network
> **Exam ready in 20 mins — focus on bold Q&As**

---

## 🧠 What is This Experiment About?

You predicted **house prices in Boston** using two models:
1. **Linear Regression** (classic ML)
2. **Deep Neural Network / DNN** (deep learning)

Then you **compared** which model performs better.

---

## 📦 The Dataset

**Q: What is the Boston Housing dataset?**
> A classic regression dataset from sklearn (via OpenML). It has **506 samples** and **13 features** used to predict median house price (MEDV → renamed PRICE).

**Q: What are the features (inputs)?**
| Feature | Meaning |
|--------|---------|
| CRIM | Crime rate per capita |
| ZN | Proportion of residential land |
| INDUS | Proportion of non-retail business acres |
| CHAS | Charles River dummy variable (1 if tract bounds river) |
| NOX | Nitric oxide concentration |
| RM | Average number of rooms per dwelling |
| AGE | Proportion of owner-occupied units built before 1940 |
| DIS | Distance to employment centres |
| RAD | Accessibility to radial highways |
| TAX | Property tax rate |
| PTRATIO | Pupil-teacher ratio |
| B | Proportion of Black population |
| LSTAT | % lower status of the population |

**Q: What is the target variable?**
> **PRICE** (originally MEDV) — Median value of owner-occupied homes in $1000s.

---

## ⚙️ Preprocessing Steps

**Q: How did you preprocess the data?**
1. Loaded using `fetch_openml`
2. Renamed `MEDV` → `PRICE`
3. Checked for null values → **No missing values** (all zeros)
4. Split into X (features) and y (PRICE)
5. **80-20 train-test split**
6. Applied **StandardScaler** — mean=0, std=1

**Q: Why do we need StandardScaler?**
> Neural networks are sensitive to scale. Features like TAX (hundreds) vs CHAS (0 or 1) are on very different scales. Standardization ensures no single feature dominates and helps gradient descent converge faster.

**Q: Why 80-20 split?**
> 80% training data gives the model enough examples to learn. 20% held out for unbiased evaluation.

---

## 📈 Model 1: Linear Regression

**Q: What is Linear Regression?**
> A model that fits a straight line (hyperplane in multiple dimensions): `y = w1x1 + w2x2 + ... + b`  
> It finds the best weights (coefficients) by minimizing Mean Squared Error.

**Q: What were the Linear Regression results?**
| Metric | Value |
|--------|-------|
| MSE | 24.291 |
| MAE | 3.189 |
| R² Score | 0.669 |

**Q: What does R² = 0.669 mean?**
> The model explains **66.9% of the variance** in house prices. Decent but not great.

**Q: What is MSE and MAE?**
- **MSE (Mean Squared Error)** = average of squared differences between actual and predicted. Penalizes large errors more.
- **MAE (Mean Absolute Error)** = average of absolute differences. More interpretable (in same unit as target).

---

## 🧬 Model 2: Deep Neural Network (DNN)

**Q: What is a DNN / Neural Network?**
> A model inspired by the brain — composed of layers of neurons. Each neuron computes a weighted sum of inputs and applies an activation function.

**Q: What was your DNN architecture?**
```
Input (13 features)
    ↓
Dense(64, activation='relu')
    ↓
Dense(32, activation='relu')
    ↓
Dense(16, activation='relu')
    ↓
Dense(1)  ← Output (regression, no activation)
```

**Q: Why ReLU activation?**
> ReLU (Rectified Linear Unit) = `max(0, x)`. It introduces non-linearity, is computationally efficient, and avoids the vanishing gradient problem better than sigmoid/tanh.

**Q: Why no activation on the output layer?**
> Because this is **regression** (predicting a continuous value). For classification you'd use sigmoid (binary) or softmax (multi-class).

**Q: What optimizer and loss did you use?**
- Optimizer: **Adam** (Adaptive Moment Estimation — adjusts learning rate automatically)
- Loss: **MSE** (standard for regression)
- Metric: **MAE**

**Q: What is Early Stopping?**
> A callback that stops training when validation loss stops improving. Used here with `patience=20` — stops if no improvement for 20 epochs. Prevents **overfitting**.

**Q: What were the DNN results?**
| Metric | Value |
|--------|-------|
| MSE | 11.622 |
| MAE | 2.272 |
| R² Score | 0.842 |

---

## ⚔️ Comparison: Linear Regression vs DNN

**Q: Which model performed better and why?**

| Model | MSE | MAE | R² |
|-------|-----|-----|----|
| Linear Regression | 24.291 | 3.189 | 0.669 |
| **Deep Neural Network** | **11.622** | **2.272** | **0.842** |

> **DNN wins.** Lower MSE, lower MAE, higher R².  
> DNN can capture **non-linear relationships** between features and price — which linear regression cannot.

**Q: Why can't Linear Regression capture non-linearity?**
> It only models a straight-line relationship. Real-world data (like housing prices) has complex interactions (e.g., crime rate AND distance to employment together affect price in a non-linear way).

---

## 🔍 EDA (Exploratory Data Analysis)

**Q: What did the correlation heatmap show?**
> - **RM** (rooms) has strong positive correlation with price
> - **LSTAT** (lower status %) has strong negative correlation with price
> - **NOX** (pollution) also negatively correlated

**Q: What did the Average Feature Values bar chart show?**
> Features like TAX and B have much higher average values compared to others — which is why scaling is important.

---

## 🧪 Is the Experiment Correct? Will Externals Like It?

### ✅ What's Good:
- Proper data loading using OpenML
- Null value check done ✔️
- StandardScaler applied correctly ✔️
- Both models trained and compared ✔️
- Visualizations: correlation heatmap, actual vs predicted plots, training loss curves ✔️
- Early stopping to prevent overfitting ✔️
- Clear comparison summary printed at the end ✔️

### ⚠️ Minor Points External May Ask:
- **Why not normalize instead of standardize?** — StandardScaler is preferred for NNs; normalization works too but StandardScaler handles outliers better.
- **Why not cross-validation?** — Simple train-test split was used. K-fold would be more robust but acceptable for this experiment.
- **Could you improve DNN further?** — Yes: dropout layers, more epochs, hyperparameter tuning.

### 🏆 Verdict:
> **Yes, experiment is correct and well-structured.** External will be satisfied — it covers EDA, preprocessing, two model comparison with metrics and visualizations. The DNN significantly outperforming Linear Regression shows proper understanding.

---

## 💡 Quick Concept Questions

**Q: What is overfitting?**
> Model performs well on training data but poorly on test data. It "memorizes" instead of learning.

**Q: What is underfitting?**
> Model is too simple to capture the pattern. Both train and test errors are high.

**Q: What is a loss function?**
> Measures how wrong the model's predictions are. The model minimizes this during training.

**Q: What is backpropagation?**
> Algorithm to compute gradients of the loss with respect to weights, flowing backward through the network. Used to update weights via gradient descent.

**Q: What is the difference between batch_size and epochs?**
- **Epoch** = one full pass through the entire training dataset
- **Batch size** = number of samples processed before weights are updated (here: 16)

**Q: Why use validation_split=0.2 during training?**
> To monitor model performance on unseen data during training — helps detect overfitting early and is required for Early Stopping to work.

**Q: What is Adam optimizer?**
> Combines the benefits of RMSProp and Momentum. Maintains adaptive learning rates per parameter. Generally works well without much tuning.

---

## 🚀 One-Line Summary to Impress the External

> *"We compared Linear Regression (R²=0.669) with a 3-hidden-layer DNN (R²=0.842) on the Boston Housing dataset. The DNN outperformed linear regression because it captured non-linear feature interactions, while Early Stopping prevented overfitting."*

---

*Good luck tomorrow! You've got this *