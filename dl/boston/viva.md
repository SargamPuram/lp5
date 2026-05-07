# Boston Housing Price Prediction — Complete Viva Guide
### Deep Learning Practical 1 | Linear Regression vs DNN

---

> **How to use:** Read question, cover answer, say it out loud.
> If you can say it confidently without reading — you're ready.

---

## SECTION 1 — Dataset & Problem

---

### Q. What is the Boston Housing Dataset?

The Boston Housing Dataset is a classic regression dataset from the UCI Machine Learning Repository, originally published in 1978. It contains 506 samples of housing data from different towns around Boston, Massachusetts. Each sample has 13 input features (like crime rate, number of rooms, tax rate) and one target variable — MEDV (Median Value of owner-occupied homes in $1000s), which we rename to PRICE. The goal is to predict the price of a house given its neighborhood characteristics. It's a supervised regression problem.

---

### Q. What are the features in the Boston dataset?

| Feature | Description |
|---------|-------------|
| CRIM | Per capita crime rate by town |
| ZN | Proportion of residential land zoned for large lots |
| INDUS | Proportion of non-retail business acres per town |
| CHAS | Charles River dummy variable (1 = bounds river) |
| NOX | Nitric oxide concentration (air pollution) |
| RM | Average number of rooms per dwelling |
| AGE | Proportion of old owner-occupied units |
| DIS | Weighted distances to Boston employment centres |
| RAD | Accessibility to radial highways |
| TAX | Property-tax rate per $10,000 |
| PTRATIO | Pupil-teacher ratio (school quality) |
| B | Proportion of Black residents (legacy feature) |
| LSTAT | Percentage lower-status population |
| **PRICE** | **Target — Median home value in $1000s** |

The most important features: **RM** (rooms, positive correlation ~0.70) and **LSTAT** (lower status %, negative correlation ~-0.74).

---

### Q. Why was Boston dataset removed from sklearn?

Boston dataset was deprecated in sklearn 1.2 and removed in 1.4 due to ethical concerns — specifically the feature B (proportion of Black residents by town), which was included in the original 1978 paper in a racially biased manner. We use `fetch_openml` which still hosts it as a historical dataset. The modern recommended alternative is the **California Housing Dataset** (`fetch_california_housing()`), which has 8 features and 20,640 samples.

---

### Q. Should I use Boston or California housing dataset?

The question specifically says "Boston Housing Dataset", so we use Boston via `fetch_openml`. In terms of accuracy, both are fine for demonstrating the concept. California housing is larger (20,640 vs 506 samples) and would train more robustly. If the external asks, say: *"We used Boston as per the question requirement; California is the modern alternative."*

---

### Q. What is the problem type here?

This is a **supervised regression** problem. Supervised because we have labeled training data (each house has a known price). Regression because the target variable (price) is a continuous numerical value — not a class or category. The goal is to learn a mapping function f(X) → y where X is the 13 features and y is the house price.

---

## SECTION 2 — EDA & Preprocessing

---

### Q. What EDA did you perform?

1. **Shape & Info** — 506 rows, 14 columns, no null values
2. **Statistical Summary** — describe() to check ranges, mean, std
3. **Null Check** — data.isnull().sum() — all zeros, no missing data
4. **Correlation Heatmap** — identifies which features correlate with price
5. **Target Distribution** — histogram + boxplot of PRICE (slight right skew, mean ~$22.5k)
6. **Average Feature Values** — bar chart of mean of each feature
7. **Scatter Plots** — RM vs Price, LSTAT vs Price, NOX vs Price — shows linear/non-linear relationships

Key EDA finding: LSTAT (-0.74) and RM (+0.70) are strongest predictors of price.

---

### Q. Why do we standardize the features?

We use `StandardScaler` which transforms each feature to have **mean = 0 and standard deviation = 1** using the formula: `z = (x - mean) / std`.

Reasons:
- Features have very different scales — TAX can be ~400 while CHAS is 0 or 1. Without scaling, features with large values dominate gradient updates
- Gradient descent converges much faster with normalized input
- Neural networks are especially sensitive to feature scale — unscaled input causes unstable training
- We fit the scaler **only on training data** (`fit_transform`) and only transform test data (`transform`) — to prevent data leakage from test set influencing scaling parameters

---

### Q. What is train-test split and what ratio did you use?

We split the 506 samples into 80% training (404 samples) and 20% testing (102 samples) using `train_test_split(X, y, test_size=0.2, random_state=42)`.

- `random_state=42` ensures reproducibility — same split every run
- Training data is used to fit both the scaler and the models
- Test data is held out and used only for final evaluation — the model never sees it during training
- This prevents overly optimistic evaluation (the model might memorize training data)

---

## SECTION 3 — Linear Regression

---

### Q. What is Linear Regression?

Linear Regression finds a linear relationship between input features and target. It fits a line (or hyperplane in multiple dimensions):

```
PRICE = w₁×CRIM + w₂×ZN + ... + w₁₃×LSTAT + bias
```

It finds weights (coefficients) `w` that minimize the Mean Squared Error between predicted and actual prices. It uses the **Ordinary Least Squares** method — a closed-form mathematical solution. It assumes a linear relationship between features and target.

---

### Q. What were your Linear Regression results?

| Metric | Value |
|--------|-------|
| MSE | 24.291 |
| RMSE | ~4.93 |
| MAE | 3.189 |
| R² | 0.669 |

Interpretation: R² = 0.669 means the model explains 66.9% of the variance in house prices. The remaining 33.1% is unexplained — because the true relationship is not perfectly linear. Average prediction error is ~$3,189 (MAE).

---

### Q. What are MSE, RMSE, MAE, and R²?

| Metric | Formula | Meaning |
|--------|---------|---------|
| **MSE** | mean((y - ŷ)²) | Mean of squared errors — penalizes large errors heavily |
| **RMSE** | √MSE | Same units as target — interpretable as average error |
| **MAE** | mean(\|y - ŷ\|) | Mean absolute error — robust to outliers |
| **R²** | 1 - SS_res/SS_tot | Proportion of variance explained — 1.0 is perfect, 0 is baseline |

Lower MSE/MAE = better. Higher R² = better (max 1.0).

---

## SECTION 4 — Deep Neural Network

---

### Q. What is a Deep Neural Network?

A DNN is a multi-layer neural network with an input layer, multiple hidden layers, and an output layer. Each layer contains neurons (nodes) connected by weights. Data flows forward through the layers (forward propagation). During training, the error is propagated backward (backpropagation) to update weights using gradient descent. "Deep" refers to having multiple hidden layers — enabling the network to learn complex non-linear patterns that linear regression cannot capture.

---

### Q. Explain your DNN architecture

```
Input Layer:    13 neurons (one per feature)
                ↓
Hidden Layer 1: 64 neurons, ReLU activation, BatchNorm, Dropout(0.2)
                ↓
Hidden Layer 2: 32 neurons, ReLU activation, BatchNorm, Dropout(0.2)
                ↓
Hidden Layer 3: 16 neurons, ReLU activation
                ↓
Output Layer:   1 neuron (no activation — regression output)
```

Total trainable parameters: ~3,000+. The decreasing layer sizes (64 → 32 → 16) create a "funnel" that forces the network to learn compact, essential representations.

---

### Q. Why ReLU activation?

ReLU (Rectified Linear Unit): `f(x) = max(0, x)`

- Simple and computationally efficient
- Avoids the **vanishing gradient problem** that sigmoid/tanh suffer from — gradients don't shrink to near zero in deep networks
- Introduces non-linearity — allowing the network to learn non-linear relationships
- Sparse activation — neurons that fire 0 effectively "turn off", creating efficient representations
- Works well in practice for regression tasks

We do NOT use ReLU on the output layer for regression — we use **no activation (linear)** so the output can be any real number including negative values.

---

### Q. What is the vanishing gradient problem?

In deep networks, backpropagation multiplies gradients through many layers. Sigmoid and tanh activations have maximum gradient ~0.25 and ~1.0 respectively. When multiplied many times through deep layers, gradients shrink exponentially — the early layers receive nearly zero gradient and stop learning. ReLU has gradient exactly 1 for positive inputs, so gradients don't shrink through ReLU layers.

---

### Q. Why is the output layer linear (no activation)?

For regression, we need the output to be any real number — house prices range from ~$5k to ~$50k. If we used sigmoid, output would be constrained to [0,1]. If we used ReLU, output would be constrained to non-negative numbers but would still introduce unnecessary non-linearity at the output. Linear output `f(x) = x` allows the network to freely output any value needed.

---

### Q. What is BatchNormalization and why did you use it?

BatchNormalization normalizes the activations of each layer to have zero mean and unit variance within each mini-batch during training. Benefits:
- Stabilizes training — reduces internal covariate shift (distribution of layer inputs keeps changing during training)
- Allows higher learning rates
- Acts as a mild regularizer — reduces overfitting
- Makes training faster and more reliable

In practice: after each Dense layer, BatchNorm smooths out the learning landscape.

---

### Q. What is Dropout and why?

Dropout randomly "drops" (sets to zero) a fraction of neurons during each training step. We used `Dropout(0.2)` — 20% of neurons are randomly disabled each forward pass.

Purpose: **Regularization** — prevents overfitting. When neurons are randomly dropped, the network cannot rely on any single neuron. It's forced to learn redundant representations. At test time, all neurons are active but their outputs are scaled by (1 - dropout_rate).

---

### Q. What optimizer did you use and why Adam?

We used **Adam (Adaptive Moment Estimation)** with learning_rate=0.001.

Adam is an adaptive optimizer that:
- Maintains per-parameter learning rates (adapts based on gradient history)
- Combines the benefits of RMSprop (scales learning rate by recent gradient magnitudes) and Momentum (uses exponential moving average of gradients)
- Works well with default hyperparameters — lr=0.001 is the standard starting point
- Converges faster than plain SGD on most problems

---

### Q. What is the loss function and why MSE?

We used `loss='mse'` (Mean Squared Error). For regression problems, MSE is the standard loss function because:
- It penalizes large errors more heavily (squared term)
- It's differentiable everywhere — required for gradient descent
- Minimizing MSE is equivalent to maximum likelihood estimation assuming Gaussian noise
- It's the same metric we use for evaluation — so training directly optimizes what we measure

---

### Q. What is Early Stopping?

```python
EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
```

Early stopping monitors validation loss during training. If val_loss doesn't improve for 20 consecutive epochs (patience=20), training stops automatically. `restore_best_weights=True` ensures the model reverts to the weights from the epoch with the best validation loss — not the final epoch.

Purpose: Prevents **overfitting** — without early stopping, the model might memorize training data as training loss keeps decreasing but validation loss starts rising.

---

### Q. What were your DNN results?

| Metric | Linear Regression | DNN |
|--------|------------------|-----|
| MSE | 24.291 | ~11-12 |
| MAE | 3.189 | ~2.2-2.5 |
| R² | 0.669 | ~0.835-0.842 |

DNN reduces MSE by ~50% and improves R² from 66.9% to 83.5%+ because it can model the **non-linear relationships** between features and price that linear regression misses.

---

### Q. Why does DNN outperform Linear Regression?

Linear Regression assumes a **linear relationship** between features and price. In reality, the relationship is non-linear — for example, the effect of adding one room (RM) on price is not constant; it depends on other factors like location and tax rate.

DNN with multiple hidden layers and ReLU activations can approximate **any continuous function** (Universal Approximation Theorem). It learns complex interactions between features — e.g., high RM + low LSTAT + low CRIM together = very high price, which linear regression can't capture with just additive weights.

---

### Q. What is the Universal Approximation Theorem?

It states that a feedforward neural network with at least one hidden layer and a non-linear activation function can approximate any continuous function to any desired accuracy, given enough neurons. This is why DNNs are powerful — they are universal function approximators. However, this doesn't mean training always finds the optimal weights — it just means the representational capacity is there.

---

### Q. What is Backpropagation?

Backpropagation is the algorithm used to train neural networks. Steps:
1. **Forward pass** — input flows through layers, prediction is made
2. **Loss computation** — MSE between prediction and actual price
3. **Backward pass** — compute gradient of loss with respect to every weight using the chain rule
4. **Weight update** — Adam optimizer adjusts weights in the direction that reduces loss

This repeats for each batch of data, for many epochs until convergence.

---

### Q. What is a batch and an epoch?

- **Batch (batch_size=16)** — training data is divided into small batches of 16 samples. Weights are updated after each batch. Smaller batches = noisier but more frequent updates; larger batches = smoother gradients but fewer updates per epoch.
- **Epoch** — one complete pass through the entire training dataset. We set max 200 epochs; early stopping stopped training earlier (around 120-130 epochs typically).
- **Mini-batch gradient descent** — our approach (batch_size=16) is a compromise between batch gradient descent (full dataset) and stochastic gradient descent (1 sample at a time).

---

### Q. What is overfitting and how did you prevent it?

**Overfitting** — model memorizes training data, performs well on training but poorly on unseen test data. Signs: training loss decreases but validation loss increases.

We prevented it with:
1. **Dropout (0.2)** — randomly disables 20% of neurons during training
2. **BatchNormalization** — mild regularization effect
3. **Early Stopping** — stops training when validation performance starts degrading
4. **Small dataset management** — 80-20 split ensures proper evaluation

---

## SECTION 5 — Tricky Examiner Questions

---

### Q. Why did we use validation_split=0.2 inside model.fit?

The training set (404 samples) is further split into 80% actual training (323 samples) and 20% validation (81 samples) during fitting. The validation set is used by EarlyStopping to monitor generalization during training — it's completely separate from the test set (102 samples). So we have three splits: train (323) → fit model, validation (81) → monitor training, test (102) → final evaluation.

---

### Q. Can Linear Regression be considered a neural network?

Yes! A single-layer perceptron with no activation (linear activation) and MSE loss is mathematically equivalent to linear regression. It's a degenerate neural network — input directly connected to output with no hidden layers. This is why the assignment title says "Linear Regression **by** Deep Neural Network" — we first demonstrate simple linear regression, then show the DNN can do much better.

---

### Q. What would happen if we didn't scale the features?

Training would be unstable or very slow. Features like TAX (~400 average) vs CHAS (0 or 1) have vastly different magnitudes. Gradient descent would update weights very unevenly — large-magnitude features would have disproportionately small gradients while small-magnitude features would have large gradients. The network might oscillate, diverge, or converge extremely slowly. StandardScaler puts all features on equal footing.

---

### Q. Why use ReLU and not Sigmoid for hidden layers?

| Property | Sigmoid | ReLU |
|----------|---------|------|
| Output range | (0, 1) | [0, ∞) |
| Vanishing gradient | Yes — saturates | No — gradient is 1 for x>0 |
| Computation | Expensive (exp) | Cheap (max) |
| Dead neurons | No | Yes (dying ReLU, x<0 always) |
| Deep networks | Poor | Excellent |

For deep networks, ReLU is the default choice. Sigmoid is still used in output layers for binary classification.

---

### Q. What is the difference between loss and metrics in model.compile?

```python
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
```

- **loss** — the quantity minimized during training by the optimizer. Gradients are computed with respect to this.
- **metrics** — additional quantities monitored and reported during training but NOT used for optimization. We track MAE to see interpretable error in dollar terms alongside the MSE loss.

---

### Q. Why does model.predict return shape (n, 1) instead of (n,)?

Keras Dense(1) output layer produces a 2D array of shape (batch_size, 1). For regression comparison with y_test (1D shape), we call `.flatten()` to convert (102, 1) → (102,). Without flatten, metric calculations like r2_score would either error or give wrong results.

---

### Q. If you had to improve the model further, what would you do?

1. **Feature engineering** — add interaction terms like RM×LSTAT
2. **More data** — 506 samples is small for a DNN; augmentation or similar datasets
3. **Hyperparameter tuning** — grid search on learning rate, hidden units, dropout rate
4. **Deeper architecture** — add more layers with skip connections (ResNet-style)
5. **Ensemble** — combine DNN with gradient boosting (XGBoost) predictions
6. **K-fold cross validation** — more robust evaluation than single train-test split
7. **Feature selection** — remove highly correlated features (INDUS, NOX, AGE have high intercorrelation)

---

## SECTION 6 — Quick Revision Flashcards

| Question | One-line Answer |
|----------|----------------|
| Dataset size? | 506 samples, 13 features, 1 target (PRICE) |
| Problem type? | Supervised regression |
| Target variable? | PRICE — median home value in $1000s |
| Best feature (positive)? | RM (rooms, r = +0.70) |
| Best feature (negative)? | LSTAT (lower status %, r = -0.74) |
| Null values? | Zero — clean dataset |
| Train-test split? | 80-20 (404 train, 102 test) |
| Why StandardScaler? | Equal feature scale → stable gradient descent |
| DNN architecture? | 13 → 64 → 32 → 16 → 1 |
| Activation used? | ReLU for hidden, Linear for output |
| Why ReLU? | No vanishing gradient, non-linearity, fast |
| Why linear output? | Regression needs unconstrained real-number output |
| Loss function? | MSE (Mean Squared Error) |
| Optimizer? | Adam, lr=0.001 |
| What is Dropout? | Random neuron disabling — prevents overfitting |
| What is BatchNorm? | Normalizes layer activations — stabilizes training |
| What is Early Stopping? | Stops when val_loss stops improving |
| Patience=20? | Wait 20 epochs before stopping |
| restore_best_weights? | Reverts to best epoch's weights, not last epoch |
| LR R² score? | ~0.669 (66.9% variance explained) |
| DNN R² score? | ~0.835-0.842 (83.5%+ variance explained) |
| LR MSE? | ~24.291 |
| DNN MSE? | ~11-12 (~50% better) |
| Why DNN beats LR? | Captures non-linear relationships |
| Universal Approx Theorem? | DNN can approximate any continuous function |
| What is backpropagation? | Algorithm to compute gradients and update weights |
| What is an epoch? | One full pass through training data |
| What is batch_size=16? | 16 samples per gradient update (mini-batch GD) |
| validation_split=0.2? | 20% of training set used to monitor training |
| Why .flatten() on predictions? | Keras returns (n,1) — need (n,) for metrics |
| Boston vs California? | Boston = question requirement; California = modern alternative |

---

*Key message: Linear Regression is interpretable but assumes linearity.*
*DNN trades interpretability for power — learns complex non-linear patterns.*
*Know your results: LR → R²=0.669, DNN → R²=0.84. DNN wins by learning what LR can't.*