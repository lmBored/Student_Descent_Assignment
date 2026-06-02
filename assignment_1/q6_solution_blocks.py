"""
Paste-ready blocks for question 6 of HW1_26_setup_p3.ipynb.

This file keeps the same informal notebook style:
- simple top-level code
- short helper functions
- markdown answers as plain strings

Choices that are not fully fixed by task.md are stated explicitly in the
markdown block `answer_q6_settings` below.
"""


# Cell 24
# load data

import pandas as pd

df = pd.read_csv("METABRIC_RNA_Mutation.csv", low_memory=False)
df_D = pd.concat([df["age_at_diagnosis"], df.iloc[:, 31:520]], axis=1)
D = df_D.to_numpy(dtype=float)
y = df["overall_survival_months"].to_numpy(dtype=float)


# Cell 25
import numpy as np

n, p = D.shape
p


# Cell 26
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def fit_ridge_model(X, y, lambdaa):
    penalty = np.eye(X.shape[1])
    penalty[0, 0] = 0
    A = X.T @ X + 2 * X.shape[0] * lambdaa * penalty
    b = X.T @ y
    return np.linalg.solve(A, b)


kf = KFold(n_splits=5, shuffle=True, random_state=69)
lambdas = np.logspace(-10, np.log10(8), 20)
lambdas_desc = lambdas[::-1]

ridge_train_sums = np.zeros(len(lambdas))
ridge_test_sums = np.zeros(len(lambdas))
ridge_feature_sums = np.zeros(len(lambdas))
lasso_train_sums = np.zeros(len(lambdas))
lasso_test_sums = np.zeros(len(lambdas))
lasso_feature_sums = np.zeros(len(lambdas))

for train_idx, test_idx in kf.split(D):
    D_train = D[train_idx]
    D_test = D[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    scaler = StandardScaler()
    D_train_scaled = scaler.fit_transform(D_train)
    D_test_scaled = scaler.transform(D_test)

    X_train = np.column_stack((np.ones(len(train_idx)), D_train_scaled))
    X_test = np.column_stack((np.ones(len(test_idx)), D_test_scaled))

    for i, lambdaa in enumerate(lambdas):
        beta_ridge = fit_ridge_model(X_train, y_train, lambdaa)

        # predict ridge model
        y_pred_ridge_train = X_train @ beta_ridge
        y_pred_ridge_test = X_test @ beta_ridge

        ridge_train_sums[i] += mean_squared_error(y_train, y_pred_ridge_train)
        ridge_test_sums[i] += mean_squared_error(y_test, y_pred_ridge_test)
        ridge_feature_sums[i] += np.sum(np.abs(beta_ridge[1:]) > 1e-16)

    lasso = Lasso(alpha=lambdas_desc[0], fit_intercept=True, warm_start=True, max_iter=50000)
    fold_lasso_train = np.zeros(len(lambdas_desc))
    fold_lasso_test = np.zeros(len(lambdas_desc))
    fold_lasso_features = np.zeros(len(lambdas_desc))

    for j, lambdaa in enumerate(lambdas_desc):
        lasso.alpha = lambdaa
        lasso.fit(D_train_scaled, y_train)

        y_pred_lasso_train = lasso.predict(D_train_scaled)
        y_pred_lasso_test = lasso.predict(D_test_scaled)

        fold_lasso_train[j] = mean_squared_error(y_train, y_pred_lasso_train)
        fold_lasso_test[j] = mean_squared_error(y_test, y_pred_lasso_test)
        fold_lasso_features[j] = np.sum(np.abs(lasso.coef_) > 1e-16)

    lasso_train_sums += fold_lasso_train[::-1]
    lasso_test_sums += fold_lasso_test[::-1]
    lasso_feature_sums += fold_lasso_features[::-1]

all_df = pd.DataFrame(
    {
        "lambda": lambdas,
        "ridge_train_mse": ridge_train_sums / 5,
        "ridge_test_mse": ridge_test_sums / 5,
        "ridge_features": ridge_feature_sums / 5,
        "lasso_train_mse": lasso_train_sums / 5,
        "lasso_test_mse": lasso_test_sums / 5,
        "lasso_features": lasso_feature_sums / 5,
    }
)


# Cell 27
all_df.head()


# Markdown for question 6 settings / assumptions
answer_q6_settings = """
Appendix/note:
+ For 5-fold cross-validation, we use `random_state=69`.
+ The question requires $$\\lambda \\in [10^{-10}, 8]$$, but it does not fix the exact grid. We use 20 log-spaced values in that interval.
+ The question does not say whether to standardize the predictors, but we standardize all predictors separately inside each training fold and apply the same scaler to the corresponding test fold. The purpose is to avoid leakage and make the regularization strength comparable across features.
+ For Ridge, we use the closed-form solution for the objective: $$\\frac{1}{2n}\\|y-X\\beta\\|^2 + \\lambda\\|\\beta_{\\neq \\text{bias}}\\|_2^2$$, so the fitted coefficients satisfy $$\\beta = (X^T X + 2n\\lambda L)^{-1} X^T y$$
+ We use `max_iter=50000` for Lasso to avoid convergence problems at small regularization values.
""".strip()


# Cell 28
# 6a plot

import matplotlib.pyplot as plt

best_test_lambda_ridge = all_df.loc[all_df["ridge_test_mse"].idxmin(), "lambda"]
best_test_lambda_lasso = all_df.loc[all_df["lasso_test_mse"].idxmin(), "lambda"]

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

axes[0].plot(all_df["lambda"], all_df["ridge_test_mse"], label="Ridge Test MSE", color="blue", lw=2)
axes[0].plot(all_df["lambda"], all_df["ridge_train_mse"], "--", label="Ridge Train MSE", color="blue", alpha=0.6)
axes[0].axvline(best_test_lambda_ridge, color="blue", linestyle=":", label=rf"Best test $\lambda$ = {best_test_lambda_ridge:.3g}")
axes[0].set_xscale("log")
axes[0].set_title(r"Ridge Regression: MSE vs $\lambda$")
axes[0].set_xlabel(r"Penalization weight ($\lambda$)")
axes[0].set_ylabel("MSE")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(all_df["lambda"], all_df["lasso_test_mse"], label="Lasso Test MSE", color="red", lw=2)
axes[1].plot(all_df["lambda"], all_df["lasso_train_mse"], "--", label="Lasso Train MSE", color="red", alpha=0.6)
axes[1].axvline(best_test_lambda_lasso, color="red", linestyle=":", label=rf"Best test $\lambda$ = {best_test_lambda_lasso:.3g}")
axes[1].set_xscale("log")
axes[1].set_title(r"Lasso Regression: MSE vs $\lambda$")
axes[1].set_xlabel(r"Penalization weight ($\lambda$)")
axes[1].set_ylabel("MSE")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# Markdown for 6a
best_ridge_row = all_df.loc[all_df["ridge_test_mse"].idxmin()]
best_lasso_row = all_df.loc[all_df["lasso_test_mse"].idxmin()]

answer_6a = f"""
For both Ridge and Lasso, the average test MSE is largest for very small values of $$\\lambda$$ and then decreases as the penalty becomes stronger. This means the left side of the plot is the high-variance part, e.g. the models are too flexible, they fit the training folds extremely well, and they generalize poorly. In particular, for tiny $$\\lambda$$ the train MSE is about {all_df.iloc[0]['ridge_train_mse']:.1f} while the test MSE is about {all_df.iloc[0]['ridge_test_mse']:.1f}, so the train-test gap is very large.

The sweet spot is around $$\\lambda \\approx {best_test_lambda_ridge:.3g}$$ for Ridge and $$\\lambda \\approx {best_test_lambda_lasso:.3g}$$ for Lasso. Around that range, the test MSE is minimized while the train MSE is still moderate. For Ridge, the minimum average test MSE is about {best_ridge_row['ridge_test_mse']:.1f}; for Lasso it is about {best_lasso_row['lasso_test_mse']:.1f}, which is slightly better.

For very large $$\\lambda$$, both models have high-bias. The coefficients are shrunk too strongly, the models become too simple, and both train and test MSE go up again. This is the right side of the plot. That is exactly the usual bias-variance tradeoff from theory: moving from left to right first reduces variance and improves test performance, but after the optimum the extra regularization increases bias too much.

The irreducible target noise $$\\sigma^2$$ is clearly not close to zero, because even at the best $$\\lambda$$ the test MSE stays far above zero. We cannot estimate $$\\sigma$$ exactly from this plot alone, but the nonzero floor of the test curve shows that part of the prediction error cannot be removed by changing the regularization strength.
""".strip()


# Cell 29
# 6b plot

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

axes[0].plot(all_df["lambda"], all_df["ridge_features"], label="Ridge", color="blue")
axes[0].plot(all_df["lambda"], all_df["lasso_features"], label="Lasso", color="red")
axes[0].set_xscale("log")
axes[0].set_title(r"Selected features vs $\lambda$")
axes[0].set_xlabel(r"Penalization weight ($\lambda$)")
axes[0].set_ylabel(r"Number of selected features ($|\beta_s| > 10^{-16}$)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(all_df["lambda"], (p - all_df["ridge_features"]) / p, label="Ridge", color="blue")
axes[1].plot(all_df["lambda"], (p - all_df["lasso_features"]) / p, label="Lasso", color="red")
axes[1].set_xscale("log")
axes[1].set_title(r"Fraction of non-selected features vs $\lambda$")
axes[1].set_xlabel(r"Penalization weight ($\lambda$)")
axes[1].set_ylabel(rf"Fraction of non-selected features ($\frac{{|\beta_s| \leq 10^{{-16}}}}{{{p}}}$)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# Markdown for 6b
answer_6b = f"""
The Ridge sparsity plot is almost completely flat. All {p} predictors stay selected for the full range of $$\\lambda$$ in this experiment, though that is what we expect from $$L_2$$ regularization. Ridge shrinks coefficients continuously, but it usually does not set them exactly to zero.

Lasso behaves very differently. For small $$\\lambda$$, it also keeps almost all predictors, but as $$\\lambda$$ increases it starts setting many coefficients exactly to zero. At the best test value $$\\lambda \\approx {best_test_lambda_lasso:.3g}$$, Lasso keeps on average about {best_lasso_row['lasso_features']:.1f} of the {p} predictors, and for the largest $$\\lambda$$ it keeps only about {all_df.iloc[-1]['lasso_features']:.1f}. So the model becomes much sparser as the penalty grows.

This matches the lecture that ridge is a shrinkage method, while Lasso is both a shrinkage and feature-selection method. The sparsity plot also helps explain the fit from part 6a. When $$\\lambda$$ is too small, both models are too flexible and variance is high. When $$\\lambda$$ is too large, Lasso removes too many predictors and Ridge shrinks everything too much, so bias becomes too high. The best region is in between, where enough information is kept to predict survival reasonably well, but the model is still regularized enough to avoid severe overfitting.
""".strip()


# Markdown for 6c
answer_6c = """
Plot (a): this is not possible if the horizontal axis is really the regularization weight $$\\lambda$$. As $$\\lambda$$ increases, the train MSE should not decrease so strongly, because stronger regularization restricts the model more. The test MSE decreasing at first is explainable, since regularization can reduce variance, but the train curve going down at the same time contradicts the usual behavior of Ridge/Lasso with increasing penalty. So the plot is either mislabeled, uses a different horizontal quantity, or is not a valid train/test MSE curve for increasing regularization.

Plot (b): this is only possible in an idealized or nearly idealized situation. The fact that train and test MSE are almost identical for all $$\\lambda$$ means variance is extremely small and the train-test split behaves almost the same on both sets. That could happen approximately if the dataset is very large, the model is very stable, and the target noise is small. The upward trend means increasing $$\\lambda$$ mainly adds bias, so the model is moving into underfitting. On a finite real dataset, exact equality of the two curves for all $$\\lambda$$ would be very unusual.

Plot (c): I interpret this one using its own legend, because the legend in the figure conflicts with the question. Under the plot legend, the blue curve is train MSE and the orange dashed curve is test MSE. This is the normal for very small $$\\lambda$$, the model overfits, so train MSE is low but test MSE is high. As $$\\lambda$$ increases, variance drops, the test MSE decreases, and the train MSE increases. After that the curves flatten, which indicates we are near the bias-variance tradeoff region where extra regularization gives very limited improvement. The remaining gap and the positive error floor show nonzero irreducible noise in the target.
""".strip()
