import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_architecture():
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Define Colors
    color_input = '#e1f5fe'
    color_encoder = '#bbdefb'
    color_attention = '#90caf9'
    color_fusion = '#64b5f6'
    color_output = '#42a5f5'
    color_adversarial = '#ffccbc'

    # 1. Input Layer
    ax.add_patch(patches.FancyBboxPatch((0.5, 8.5), 2, 1, boxstyle="round,pad=0.1", color=color_input, ec='black'))
    ax.text(1.5, 9, 'Multilingual Text\n(English/Hindi/etc)', ha='center', va='center', fontsize=10, weight='bold')

    ax.add_patch(patches.FancyBboxPatch((0.5, 7), 2, 1, boxstyle="round,pad=0.1", color=color_input, ec='black'))
    ax.text(1.5, 7.5, 'Source Metadata\n(Embedding)', ha='center', va='center', fontsize=10, weight='bold')

    ax.add_patch(patches.FancyBboxPatch((0.5, 5.5), 2, 1, boxstyle="round,pad=0.1", color=color_input, ec='black', ls='--'))
    ax.text(1.5, 6, 'Image/Reel Branch\n(ResNet18)', ha='center', va='center', fontsize=10, weight='bold')

    # 2. Backbone
    ax.add_patch(patches.FancyBboxPatch((3.5, 8.5), 2.5, 1, boxstyle="round,pad=0.1", color=color_encoder, ec='black'))
    ax.text(4.75, 9, 'XLM-Roberta\n+ LoRA Adapters', ha='center', va='center', fontsize=11, weight='bold')

    # 3. Features
    ax.add_patch(patches.FancyBboxPatch((3.5, 7), 2.5, 1, boxstyle="round,pad=0.1", color=color_attention, ec='black'))
    ax.text(4.75, 7.5, 'Self-Attention\n(Keyword Focuser)', ha='center', va='center', fontsize=10, weight='bold')

    # 4. Adversarial Branch
    ax.add_patch(patches.FancyBboxPatch((7, 8.5), 2, 1, boxstyle="round,pad=0.1", color=color_adversarial, ec='black'))
    ax.text(8, 9, 'GRL + Language\nDiscriminator', ha='center', va='center', fontsize=10, weight='bold')

    # 5. Fusion & Classifier
    ax.add_patch(patches.FancyBboxPatch((7, 6), 2.5, 1.5, boxstyle="round,pad=0.1", color=color_fusion, ec='black'))
    ax.text(8.25, 6.75, 'Feature Fusion\n& Deep Classifier\n(MLP)', ha='center', va='center', fontsize=11, weight='bold')

    # 6. Final Output
    ax.add_patch(patches.Circle((8.25, 4.5), 0.6, color=color_output, ec='black'))
    ax.text(8.25, 4.5, 'SOTA\nPREDICTION', ha='center', va='center', fontsize=10, weight='bold', color='white')

    # Arrows
    arrow_style = dict(arrowstyle='->', lw=2, color='black')
    ax.annotate('', xy=(3.5, 9), xytext=(2.5, 9), arrowprops=arrow_style)
    ax.annotate('', xy=(3.5, 7.5), xytext=(4.75, 8.5), arrowprops=arrow_style)
    ax.annotate('', xy=(7, 9), xytext=(6, 9), arrowprops=arrow_style)
    ax.annotate('', xy=(7.5, 7.5), xytext=(6, 7.5), arrowprops=arrow_style)
    ax.annotate('', xy=(8.25, 6), xytext=(8.25, 5.1), arrowprops=arrow_style)
    
    # Metadata and Image to Fusion
    ax.annotate('', xy=(7, 6.5), xytext=(2.5, 7.5), arrowprops=dict(arrowstyle='->', lw=1.5, color='gray', ls=':'))
    ax.annotate('', xy=(7, 6.2), xytext=(2.5, 6), arrowprops=dict(arrowstyle='->', lw=1.5, color='gray', ls=':'))

    plt.title('SOTA Multilingual Adversarial Fake News Detector - Architecture', fontsize=16, weight='bold', pad=20)
    plt.savefig('architecture_graph.png', dpi=300, bbox_inches='tight')
    print("Architecture graph saved as 'architecture_graph.png'!")

if __name__ == "__main__":
    draw_architecture()
