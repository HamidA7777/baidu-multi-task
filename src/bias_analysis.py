import torch
import matplotlib.pyplot as plt
import os

from model import CrossEncoder
from transformers import BertConfig

VOCAB_SIZE = 22000
HIDDEN_SIZE = 768
NUM_HIDDEN_LAYERS = 12
NUM_ATTENTION_HEADS = 12

config = BertConfig(
    vocab_size=VOCAB_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_hidden_layers=NUM_HIDDEN_LAYERS,
    num_attention_heads=NUM_ATTENTION_HEADS,
)

model_path = os.path.join(os.path.expanduser('~/baidu-bert-utils/multi-task-separate-logits-pos-debiasing'), 'multi-task-model-with-debiasing-separate-logits.pt')

if not os.path.exists(model_path):
    raise FileNotFoundError(
        "Model checkpoint not found. "
        "Please train the model first"
    )

model = CrossEncoder(config=config)
model.load_state_dict(torch.load(model_path, map_location='cpu')['model'])
model.eval()

positions = torch.arange(1, 11)
with torch.no_grad():
    bias = model.bias_tower(positions)

click_bias  = bias[:, 0].cpu().numpy()
skip_bias   = bias[:, 1].cpu().numpy()
dwell_bias  = bias[:, 2].cpu().numpy()

positions = positions.cpu().numpy()

# print(click_bias)
# print(skip_bias)
# print(dwell_bias)

plt.plot(positions, click_bias, label="Click Bias")
plt.plot(positions, skip_bias, label="Skip Bias")
plt.plot(positions, dwell_bias, label="Dwell-time Bias")

plt.xlabel("Position")
plt.ylabel("Position Bias (per-head)")
plt.title("Learned Position Bias from Bias Tower")
plt.legend()
plt.grid(True)
plt.show()
