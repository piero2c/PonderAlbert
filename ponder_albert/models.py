from typing import Union, List, Tuple
import torch
from torch import nn

from transformers import (
    PreTrainedTokenizer, PreTrainedTokenizerFast,
    AutoConfig, AutoModel, AutoTokenizer
)
from transformers.models.albert.modeling_albert import (
    AlbertEmbeddings, AlbertConfig, AlbertLayer
)

from .distributions import GeneralizedGeometricDist


class PonderAlbertClassifier(nn.Module):
    def __init__(self, config,
                 tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
                 target_halt_probability: float, **kwargs):
        super().__init__()

        self.config = config

        # Albert encoder
        self.embeddings = AlbertEmbeddings(config)
        self.embedding_hidden_mapping_in = nn.Linear(config.embedding_size,
                                                     config.hidden_size)
        self.albert_layer = AlbertLayer(config)

        # Adds a classification head using the last hidden_state
        # of the first token (CLS)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 128),
            nn.GELU(),
            nn.Linear(128, config.num_labels)
        )

        # Classifier that predicts weather the model should stop
        # using the hidden state of the second (CLS) token.
        self.ponder = nn.Sequential(
            nn.Linear(config.hidden_size, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # Since we add an extra [CLS] token, the tokenizer is also stored
        # in this object
        assert isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast))
        self.tokenizer = tokenizer

        # Halt probability of target distribution
        self.target_halt_probability = target_halt_probability

    @classmethod
    def from_pretrained(cls, model_name, target_halt_probability: float,
                        **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, **kwargs)

        obj = cls(model.config, tokenizer, target_halt_probability)

        # Compares and loads state dict
        model_layer_groups = model.encoder.albert_layer_groups
        if len(model_layer_groups) != 1:
            raise ValueError(
                'This module only supports loading from a single layer_group'
            )

        model_layers = model_layer_groups[0].albert_layers
        if len(model_layers) != 1:
            raise ValueError

        # Adds embedding layers
        model_state_dict = {
            k: param for k, param in model.state_dict().items()
            if 'embeddings' in k
        }

        # Adds albert layers params
        model_state_dict.update({
            f'albert_layer.{k}': param
            for k, param in model_layers[0].state_dict().items()
        })

        # Adds embedding_hidden_mapping_in params
        model_state_dict.update({
            k: param for k, param in model.encoder.state_dict().items()
            if 'embedding_hidden_mapping_in.' in k
        })

        # Adds classifier and ponder from obj
        for param_name, param in obj.state_dict().items():
            if any(kw in param_name for kw in ['classifier', 'ponder']):
                assert param_name not in model_state_dict
                model_state_dict[param_name] = param

        # Checks if there's nothing left
        not_found = obj.state_dict().keys() - model_state_dict.keys()
        if len(not_found) != 0:
            raise ValueError(f'The keys {not_found} failed to match w/ {model_name}.')

        # Loads the state dict
        obj.load_state_dict(model_state_dict)

        return obj

    def forward(self, texts: Union[List[str], Tuple[List[str], List[str]]]):
        # Adds an extra CLS token for the halting model
        if isinstance(texts, list):
            texts = [self.tokenizer.cls_token + text for text in texts]
            tokenized_texts = self.tokenizer(texts,
                                             padding=True, truncation=True,
                                             return_tensors='pt')
        elif isinstance(texts, tuple):
            texts = [[self.tokenizer.cls_token + text for text in d] for d in texts]
            tokenized_texts = self.tokenizer(texts[0], texts[1],
                                             padding=True, truncation=True,
                                             return_tensors='pt')
        else:
            raise ValueError

        attention_mask, input_ids = tokenized_texts['attention_mask'], tokenized_texts['input_ids']
        token_type_ids = tokenized_texts['token_type_ids']

        nb_texts = attention_mask.shape[0]

        # Resizes attention mask for AlbertLayer forward pass
        attention_mask = attention_mask.reshape(nb_texts, 1, 1, -1)
        attention_mask = (1 - attention_mask) * (-10000.0)

        embeddings = self.embeddings(input_ids, token_type_ids)
        state = self.embedding_hidden_mapping_in(embeddings)

        # Stores hidden_states, halt_probabilities and classifier predictions
        # for each layer
        hidden_states, halt_probabilities, predictions = [], [], []
        alive_layers = torch.rand(nb_texts) < 1  # Only used in eval mode

        for i in range(self.config.num_hidden_layers):
            # Encoder step
            state = self.albert_layer(state, attention_mask)[0]
            hidden_states.append(state)

            # Runs the classifier head using the first [CLS] token
            classifier_prediction = self.classifier(state[:, 0])
            predictions.append(classifier_prediction)

            # Runs the ponder model using the second [CLS] token
            halt_probability = self.ponder(state[:, 1])
            halt_probabilities.append(halt_probability)

            if not self.training:
                alive_layers *= (torch.rand(nb_texts) > halt_probability.squeeze())

                if not alive_layers.any():
                    break

        # Halt distribution
        halt_probabilities = torch.cat(halt_probabilities, axis=1)
        model_halt_dist = GeneralizedGeometricDist(halt_probabilities)

        # KL Div
        batch_size, max_steps = halt_probabilities.shape
        target_halt_dist = GeneralizedGeometricDist(self.target_halt_probability,
                                                    batch_size=batch_size,
                                                    max_steps=max_steps)

        return {
            'logits': torch.stack(predictions),
            'halt_probabilities': halt_probabilities,
            'model_halt_dist': model_halt_dist,
            'target_halt_dist': target_halt_dist,
            'passes': i
        }
