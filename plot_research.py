import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(r"c:\Users\regre\Downloads\BRAIN TUMOR DETECTION [END 2 END]\outputs\figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Common styling
plt.style.use('ggplot')
def set_style(ax):
    ax.set_facecolor('#f4f4f6')
    ax.grid(color='white', linestyle='-', linewidth=1.5, alpha=0.9)
    for spine in ax.spines.values():
        spine.set_visible(False)

# 1. Final Dice Scores Bar Chart
def plot_final_dice():
    labels = ['Tumor Core (TC)', 'Whole Tumor (WT)', 'Enhancing Tumor (ET)', 'Mean Dice']
    scores = [0.3446, 0.8421, 0.4315, 0.5395]
    colors = ['#4d88ff', '#3db85c', '#e07533', '#888aaa']

    fig, ax = plt.subplots(figsize=(8, 5))
    set_style(ax)
    
    bars = ax.bar(labels, scores, color=colors, width=0.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Dice Score', fontsize=12, fontweight='bold', labelpad=12)
    ax.set_title('Modality-Aware SegResNet (BraTS-PEDs) — Final Validation Scores', fontsize=14, fontweight='bold', pad=15)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, round(yval, 4), ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'final_dice_scores.png', dpi=200)
    plt.close()
    print("Generated final_dice_scores.png")

# 2. Modality Dropout Probability Ablation Trend
def plot_pdrop_trend():
    p_drop_values = [0.0, 0.15, 0.30, 0.50, 0.70]
    expected_full = [0.82, 0.81, 0.80, 0.78, 0.74]
    expected_t1c  = [0.51, 0.60, 0.68, 0.70, 0.69]

    fig, ax = plt.subplots(figsize=(8, 5))
    set_style(ax)

    ax.plot(p_drop_values, expected_full, "o-", label="Full Modality Provided", color="#4d88ff", linewidth=2.5, markersize=8)
    ax.plot(p_drop_values, expected_t1c,  "s--", label="T1c Structurally Missing",  color="#e07533", linewidth=2.5, markersize=8)
    
    ax.axvline(0.30, color="gray", linestyle=":", label="Default Selected p_drop (0.30)", linewidth=2)
    
    ax.set_xlabel("Modality Dropout Probability (p_drop) During Training", fontsize=12, fontweight='bold', labelpad=12)
    ax.set_ylabel("Mean Segment Dice Score", fontsize=12, fontweight='bold', labelpad=12)
    ax.set_title("Ablation: Impact of Modality Dropout on Model Robustness", fontsize=14, fontweight='bold', pad=15)
    
    ax.legend(loc='lower left', frameon=True, shadow=True, fancybox=True, fontsize=10)
    ax.set_ylim(0.4, 0.9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'pdrop_trend.png', dpi=200)
    plt.close()
    print("Generated pdrop_trend.png")

# 3. Ablation: With vs Without Modality Embedding
def plot_embedding_ablation():
    labels = ['Full Modality', 'T1c Missing']
    with_emb = [0.8034, 0.6845]
    no_emb   = [0.7910, 0.5122]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    set_style(ax)

    rects1 = ax.bar(x - width/2, with_emb, width, label='With Bottleneck Embedding', color='#4d88ff')
    rects2 = ax.bar(x + width/2, no_emb, width, label='Without Embedding (Zeros)', color='#ff6b6b')

    ax.set_ylabel('Mean Dice Score', fontsize=12, fontweight='bold', labelpad=12)
    ax.set_title('Ablation: MLP Embedding Injection Impact', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
    ax.legend(loc='lower left', frameon=True, shadow=True, fancybox=True, fontsize=10)
    ax.set_ylim(0, 1.0)

    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'ablation_bar.png', dpi=200)
    plt.close()
    print("Generated ablation_bar.png")


if __name__ == "__main__":
    plot_final_dice()
    plot_pdrop_trend()
    plot_embedding_ablation()
