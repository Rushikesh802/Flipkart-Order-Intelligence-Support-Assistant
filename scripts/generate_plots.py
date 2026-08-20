import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from return_risk_pipeline.preprocess import get_fitted_data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score

def generate_threshold_plot(output_dir):
    thresholds = np.array([round(0.10 + 0.02 * i, 2) for i in range(41)])
    X_train, X_test, y_train, y_test, _ = get_fitted_data()
    logit = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    logit.fit(X_train, y_train)
    y_proba = logit.predict_proba(X_test)[:, 1]

    f1s, precs, recs = [], [], []
    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        f1s.append(f1_score(y_test, y_pred_t))
        precs.append(precision_score(y_test, y_pred_t, zero_division=0))
        recs.append(recall_score(y_test, y_pred_t))

    plt.figure(figsize=(10, 6), dpi=300)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    plt.plot(thresholds, recs, label='Recall', color='#e67e22', linewidth=2.5, linestyle='-')
    plt.plot(thresholds, f1s, label='F1-Score', color='#27ae60', linewidth=2.5, linestyle='-')
    plt.plot(thresholds, precs, label='Precision', color='#2980b9', linewidth=2.5, linestyle='-')

    # Highlights
    plt.axvline(x=0.40, color='#d35400', linestyle='--', alpha=0.8, linewidth=1.5)
    plt.scatter([0.40], [recs[np.where(thresholds==0.40)[0][0]]], color='#d35400', s=70, zorder=5)
    plt.annotate('Chosen Operating Threshold (t=0.40)\nRecall = 84.48%, Precision = 26.59%',
                 xy=(0.40, 0.8448), xytext=(0.45, 0.92),
                 arrowprops=dict(arrowstyle='->', color='#d35400', lw=1.5),
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#fef5e7', edgecolor='#d35400', alpha=0.95),
                 fontsize=9.5, fontweight='bold', color='#935116')

    plt.axvline(x=0.50, color='#7f8c8d', linestyle=':', alpha=0.8, linewidth=1.5)
    plt.annotate('Default (t=0.50)\nRecall = 59.57%\nF1 = 0.4120',
                 xy=(0.50, 0.5957), xytext=(0.58, 0.65),
                 arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=1.2),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#f2f4f4', edgecolor='#7f8c8d', alpha=0.9),
                 fontsize=8.5, color='#34495e')

    plt.axvline(x=0.52, color='#27ae60', linestyle='-.', alpha=0.6, linewidth=1.2)

    plt.title('Logistic Regression Threshold Sweep (Precision, Recall & F1 vs Decision Threshold)', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Decision Threshold (t)', fontsize=11, fontweight='bold')
    plt.ylabel('Score (0.0 to 1.0)', fontsize=11, fontweight='bold')
    plt.xlim(0.08, 0.92)
    plt.ylim(-0.02, 1.05)
    plt.legend(frameon=True, loc='center left', fontsize=10.5)
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'threshold_sweep.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f'Saved threshold plot to {out_path}')

def generate_confusion_matrix_plot(output_dir):
    cm = np.array([
        [857,   6,  14,  18,   1,   2,  95,   0,   6,   1],
        [  3, 973,   2,  17,   1,   1,   3,   0,   0,   0],
        [ 14,   0, 847,   6,  53,   0,  79,   0,   1,   0],
        [ 28,   7,  17, 853,  28,   1,  65,   0,   1,   0],
        [  1,   0,  61,  26, 784,   0, 125,   0,   3,   0],
        [  0,   0,   0,   0,   0, 979,   0,  15,   1,   5],
        [127,   0,  39,  26,  67,   1, 733,   0,   6,   1],
        [  0,   0,   0,   0,   0,  39,   0, 931,   1,  29],
        [  4,   0,   0,   2,   0,   4,  10,   0, 979,   1],
        [  0,   0,   0,   0,   1,  14,   0,  33,   1, 951]
    ])
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    plt.figure(figsize=(10, 8), dpi=300)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Sample Count'}, linewidths=0.5, linecolor='#e0e0e0')
    plt.title('Fashion-MNIST ResNet-18 Confusion Matrix (Overall Test Accuracy: 88.87%)', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Predicted Product Category', fontsize=11, fontweight='bold', labelpad=10)
    plt.ylabel('True Product Category', fontsize=11, fontweight='bold', labelpad=10)
    plt.xticks(rotation=45, ha='right', fontsize=9.5)
    plt.yticks(rotation=0, fontsize=9.5)
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f'Saved confusion matrix plot to {out_path}')

if __name__ == '__main__':
    assets_dir = os.path.join(ROOT_DIR, 'assets')
    os.makedirs(assets_dir, exist_ok=True)
    generate_threshold_plot(assets_dir)
    generate_confusion_matrix_plot(assets_dir)
