# Predicting Credit Card Default: Statistical Models vs a Neural Network

A lender needs to know which clients are likely to miss their next payment. This project
predicts whether a credit card client in Taiwan defaulted on their October 2005 payment, using
six months of billing and repayment history for 30,000 clients [1], and asks whether a neural
network does better than classical statistical models on the same data.

## Notebooks

| Part | Notebook | What it does |
|---|---|---|
| 1 | [CreditCard_ML_ModelComparison.ipynb](CreditCard_ML_ModelComparison.ipynb) | Explores the data and trains three statistical models: a random forest, an RBF support vector machine and Gaussian Naive Bayes |
| 2 | [NeuralNetwork_CreditCardDefaults.ipynb](NeuralNetwork_CreditCardDefaults.ipynb) | Trains a neural network in PyTorch on the same features and split, and compares it with the Part 1 models on the same test clients |

Both notebooks are saved with their outputs and explain each step next to the code, so they can
be read on GitHub without running anything.

## Data

The Default of Credit Card Clients dataset from the UCI Machine Learning Repository [1], loaded
straight through UCI's Python API (`ucimlrepo`), so there is no data file to download. Each row
is one client: credit limit, education, age, and for each month from April to September 2005
the repayment status, bill amount and amount paid.

- **22.1% of clients defaulted**, so always predicting "no default" is already 77.9% accurate.
  Every result is read against that floor, with recall and ROC-AUC reported next to accuracy.
- **Some codes are not in the data dictionary** (for example `EDUCATION` 0, 5 and 6). They are
  merged into "others", and the assumptions are noted in Part 1.

## Approach

- **Excluded features.** `SEX` and `MARRIAGE` are protected characteristics in credit decisions
  (for example under the US Equal Credit Opportunity Act [2]), so they are left out of every
  model. 21 features remain.
- **One split for both parts.** A stratified 80/20 train/test split with seed 20: 24,000
  training and 6,000 test clients. Part 1 saves its fitted models and the test rows to
  `models/`. Part 2 checks that it reproduces exactly the same test rows and loads those models,
  so all four models are scored on the same clients.
- **Part 1: statistical models.** Random forest, RBF SVM and Gaussian Naive Bayes in
  scikit-learn [3], each tuned with 5-fold cross-validation on ROC-AUC.
- **Part 2: neural network.** A feed-forward network in PyTorch [4]: fully connected layers with
  ReLU activations and dropout. 12 settings of network shape, dropout and learning rate are
  tried on a validation set held out from the training data, with early stopping. The best (three
  hidden layers of 256, 128 and 64 units) is retrained on the full training set.
- **Evaluation.** Accuracy, precision, recall, F1 and ROC-AUC on the test set, McNemar's test [5]
  to check whether two models really differ, and a look at specific clients the models get right
  and wrong.

## Results

Test set (6,000 clients, 0.5 threshold):

| Model | Accuracy | Recall (defaulters caught) | ROC-AUC |
|---|---|---|---|
| Always "no default" | 0.779 | 0.000 | 0.500 |
| Random forest | **0.821** | 0.356 | **0.771** |
| SVM (RBF) | 0.817 | 0.320 | 0.702 |
| Naive Bayes | 0.754 | **0.523** | 0.736 |
| Neural network | 0.820 | 0.371 | 0.770 |

- **The neural network ties the random forest.** 0.820 against 0.821 accuracy, and McNemar's
  test finds no difference (p = 0.87). The two make the same prediction for 97.5% of test
  clients, and where they disagree both are close to the 0.5 threshold.
- **The best models are only about 4 points above the "always no" floor.** Most of the signal
  sits in the latest repayment status.
- **Every model misses the same hard cases.** 552 of the 1,327 test defaulters (42%) are missed
  by all four models. Most of them were up to date on their latest payments, so nothing in their
  repayment history warns of default.
- **Naive Bayes catches the most defaulters** (recall 0.523), but flags so many clients that it
  falls below the accuracy floor.

**Verdict.** The random forest remains the better choice for this problem: the same accuracy as
the network with no preprocessing, fewer settings to tune, and predictions that are easier to
explain to a lender. This matches the wider finding that tree ensembles still match or beat deep
networks on medium-sized tables of features [6].

## Limitations

- **All models use the 0.5 threshold.** It keeps the comparison fair but leaves recall low. A
  lender would choose the threshold from what a missed defaulter costs compared with an
  unnecessary review.
- **The features carry limited signal.** The models agree on the easy cases and all miss the
  same hard ones, so better results likely need more information about each client, not a more
  complex model.

## Run it yourself

```bash
pip install ucimlrepo scikit-learn==1.6.1 torch pandas numpy matplotlib statsmodels joblib
```

Part 2 loads the saved Part 1 models from `models/`. They were saved with scikit-learn 1.6.1 and
may not load in other versions; rerunning Part 1 recreates them. Part 2 runs in about a minute
on a laptop CPU.

```
CreditCard_ML_ModelComparison.ipynb      Part 1: statistical models
NeuralNetwork_CreditCardDefaults.ipynb   Part 2: neural network and comparison
models/                                  saved models and the test-set row numbers
```

## References

1. Yeh, I.-C., & Lien, C.-H. (2009). *Default of credit card clients* [Dataset]. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
2. Equal Credit Opportunity Act, 15 U.S.C. § 1691. https://www.law.cornell.edu/uscode/text/15/1691
3. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830. https://www.jmlr.org/papers/v12/pedregosa11a.html
4. Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems*, 32. https://arxiv.org/abs/1912.01703
5. Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. *Neural Computation*, 10(7), 1895-1923. https://doi.org/10.1162/089976698300017197
6. Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? *Advances in Neural Information Processing Systems*, 35, 507-520. https://arxiv.org/abs/2207.08815
