# Visualize attention weights
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(attn_weights, tokens, title='Attention', figsize=(10,2)):
    plt.figure(figsize=figsize)
    if len(attn_weights.shape) == 1:
        attn_weights = attn_weights.reshape(1, -1)
    sns.heatmap(attn_weights, annot=True, fmt='.2f', xticklabels=tokens, yticklabels=[''])
    plt.title(title)
    plt.tight_layout()
    plt.show()
