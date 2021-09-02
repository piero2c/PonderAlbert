# PonderALBERT

Hi! This is an experimental project that tries to combine variable-depth Pretrained Transformers with a halt mechanism.
For this project, I choose to use ALBERT ([Lan et. al](https://arxiv.org/abs/1909.11942)), which has variable depth (due to weight-sharing), and the Halting mechanism proposed by the recent paper "PonderNet: Learning to Ponder" ([Banino et. al](https://arxiv.org/abs/2107.05407)).

For a detailed description of the halting mechanism, I suggest reading the [PonderNet paper ](https://arxiv.org/abs/2107.05407) or watching the amazing [video explanation](https://www.youtube.com/watch?v=nQDZmf2Yb9k) created by Yannic Kilcher.

## Usage

**Model loading**

```python
from transformers import AlbertConfig, AlbertTokenizer
from ponder_albert.models import PonderAlbertClassifier

# A blank classifier with an Albert encoder can be initialized directly using an AlbertConfig object
# and a trained tokenizer
config = AlbertConfig(num_hidden_layers=12)
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
model = PonderAlbertClassifier(config, tokenizer, target_halt_probability=0.2)

# Alternatively, you can initialize the classifier using pretrained weights from the HF model database.
# Since ALBERT has a variable-depth encoder, you can still set the number of layers used as you want
model = PonderAlbertClassifier.from_pretrained('albert-base-v2', num_hidden_layers=43,
                                               target_halt_probability=0.2)

```

**Model training**

```python
from ponder_albert.losses import PonderClassificationLoss

# Sample dataset
texts = ['The cat sat on the mat']
labels = [0]

# Initializes the PonderNet criterion for text-classification
optimizer = torch.optim.Adam(model.parameters())
criterion = PonderClassificationLoss(kl_penalty_factor=1e-2)
model.train()

# Single parameter update
prediction = model(texts)
loss = criterion(prediction, labels)['total_loss']
loss.backward()
optimizer.step()
```

**Inference with halting mechanism**

During inference, the halting mechanism can stop the model halfway, but since the halting
mechanism is stochastic, the results can still vary.

```python
model.eval()

# Let's try it once
model(['My cool new text'])

# {'logits': ...,
#  'halt_probabilities': ...,
#  'model_halt_dist': GeneralizedGeometricDist(),
#  'target_halt_dist': GeneralizedGeometricDist(),
#  'passes': 5
#  }

# And again!
model(['My cool new text'])

# {'logits': ...,
#  'halt_probabilities': ...,
#  'model_halt_dist': GeneralizedGeometricDist(),
#  'target_halt_dist': GeneralizedGeometricDist(),
#  'passes': 8
#  }
```

