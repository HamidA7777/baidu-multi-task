import pandas as pd
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

from multi_task_fuse import MultiTaskFusionModel

def get_weights(model, model_weights_path):
    model_weights_path = os.path.expanduser(model_weights_path)
    model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))

    weights = None

    for name, param in model.named_parameters():
        if 'weight' in name:
            weights = param.data.flatten().tolist()

    return weights

if __name__ == '__main__':
    base_path_multi_task = '~/baidu-bert-utils/multi-task-one-logit-pos-debiasing'
    multi_task_fusion_model_path = os.path.join(base_path_multi_task, 'best_multi-task-fusion_model.pt')

    if not os.path.exists(multi_task_fusion_model_path):
        raise FileNotFoundError(
            "Model checkpoint not found. "
            "Please train the model first or download the checkpoint as described in README."
        )

    model = MultiTaskFusionModel()

    labels = ['Click', 'Skip', 'Dwell-time']
    x = np.arange(len(labels))
    bar_width = 0.4

    fusion_weights = get_weights(model, multi_task_fusion_model_path)

    print(fusion_weights)

    fig, ax = plt.subplots(figsize=(8,6))

    ax.bar(x, fusion_weights, bar_width, label='Multi-task model with position debiasing')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylabel('Fusion Weight')
    ax.set_title('Contribution of each signal to the final relevance score in the Fusion Model')

    plt.tight_layout()
    plt.show()
