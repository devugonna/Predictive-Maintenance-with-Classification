'''
    eda.py
Uses basic matplotlib to plot the results to visually show both database
and model viability results to be put in the figures folder.
Machine Failure has 5 different types and are set in the FLAGS
array. All 7 figures and a feature summary .csv file
are added to the project's figures folder, after evaluation is run.
'''
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, ConfusionMatrixDisplay

from config import OUTPUTS_DIR, STATE

FIG_DIR = OUTPUTS_DIR / "figures"
MEASUREMENTS = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]',
                'Torque [Nm]', 'Tool wear [min]', 'Temp diff [K]']
FLAGS = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=150, bbox_inches='tight')
    plt.close(fig)


def basic_stats(df, train, y_train, y_test):
    by_type = train.groupby('Type')['Machine failure'].mean().rename(index={0.0: 'L', 1.0: 'M', 2.0: 'H'})
    print("Rows:", len(df), " Missing:", df.isna().sum().sum(), " Duplicates:", df.duplicated().sum())
    print("Positive rate (train):", round(y_train.mean(), 4))
    print("Failures in train:", y_train.sum(), " Failures in test:", y_test.sum())
    print("\nFailure rate by Type:\n", by_type.round(4))
    print("\nFailure flags set per row:\n", df[FLAGS].sum(axis=1).value_counts())
    print("\nFailures per mode:\n", df[FLAGS].sum())
    summary = train[MEASUREMENTS + ['Power [W]', 'Strain']].describe().T
    summary.to_csv(OUTPUTS_DIR / "feature_summary.csv")
    print("\nFeature summary:\n", summary.round(2))


def fig_type(train):
    rate = train.groupby('Type')['Machine failure'].mean() * 100
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(['L', 'M', 'H'], rate.values)
    ax.axhline(train['Machine failure'].mean() * 100, color='red', linestyle='--', label='overall')
    ax.set_xlabel('Product type')
    ax.set_ylabel('Failure rate (%)')
    ax.set_title('Failure rate by product type')
    ax.legend()
    save(fig, "01_failure_by_type.png")


def fig_distributions(train):
    fail = train['Machine failure'] == 1
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, col in zip(axes.ravel(), MEASUREMENTS):
        bins = np.linspace(train[col].min(), train[col].max(), 31)
        ax.hist(train.loc[~fail, col], bins=bins, density=True, alpha=0.5, label='healthy')
        ax.hist(train.loc[fail, col], bins=bins, density=True, alpha=0.5, label='failure')
        ax.set_title(col)
    axes[0, 0].legend()
    save(fig, "02_distributions.png")


def fig_correlation(train):
    cols = MEASUREMENTS + ['Power [W]', 'Strain']
    corr = train[cols].corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha='right')
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha='center', va='center', fontsize=8)
    fig.colorbar(im)
    ax.set_title('Correlation between measurements')
    save(fig, "03_correlation.png")


def fig_wear(train):
    wear_bin = pd.cut(train['Tool wear [min]'], bins=range(0, 275, 25), include_lowest=True)
    rate = train.groupby(wear_bin, observed=True)['Machine failure'].mean() * 100
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(rate)), rate.values)
    ax.set_xticks(range(len(rate)))
    ax.set_xticklabels([str(b) for b in rate.index], rotation=45, ha='right')
    ax.set_xlabel('Tool wear bin [min]')
    ax.set_ylabel('Failure rate (%)')
    ax.set_title('Failure rate by tool wear')
    save(fig, "04_wear_vs_failure.png")


def fig_pr_curve(tuner, X_test, y_test):
    p = tuner.predict_proba(X_test)[:, 1]
    pred = tuner.predict(X_test)
    prec, rec, _ = precision_recall_curve(y_test, p)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, label='tuned model')
    ax.scatter([recall_score(y_test, pred)], [precision_score(y_test, pred)], color='red', zorder=3,
               label=f"threshold {tuner.best_threshold_:.2f}")
    ax.axhline(y_test.mean(), color='grey', linestyle='--', label='no skill')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-recall curve (test set)')
    ax.legend()
    save(fig, "05_pr_curve.png")


def fig_confusion(tuner, X_test, y_test):
    pred = tuner.predict(X_test)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax, colorbar=False)
    ax.set_title('Confusion matrix (test set)')
    save(fig, "06_confusion_matrix.png")


def fig_importance(eng_permutation, non_eng_permutation):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, series, title in zip(axes, [eng_permutation, non_eng_permutation], ['Engineered', 'Raw']):
        s = series.sort_values()
        ax.barh(s.index, s.values)
        ax.set_xlabel('Drop in PR-AUC when shuffled')
        ax.set_title(title)
    save(fig, "07_permutation_importance.png")


def run():
    df = STATE['df_eng']
    X_train, X_test = STATE['X_train'], STATE['X_test']
    y_train, y_test = STATE['y_train'], STATE['y_test']
    train = df.loc[X_train.index]
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    basic_stats(df, train, y_train, y_test)
    fig_type(train)
    fig_distributions(train)
    fig_correlation(train)
    fig_wear(train)
    fig_pr_curve(STATE['best_tuning'], X_test, y_test)
    fig_confusion(STATE['best_tuning'], X_test, y_test)
    fig_importance(STATE['eng_permutation'], STATE['non_eng_permutation'])

    STATE.update(train_df=train)


if __name__ == "__main__":
    import data_load, model_selection, training, evaluate
    data_load.run()
    model_selection.run()
    training.run()
    evaluate.run()
    run()
