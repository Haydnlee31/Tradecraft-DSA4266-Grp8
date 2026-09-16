"""MLP classifier architectures for the centralized-heavy and centralized-light lanes.

Both lanes solve the same 8-class task on the same pooled, standardized feature set
(see dataset.py) and share this one configurable MLP class -- only the layer widths
differ. This keeps the Heavy/Light comparison in Section 05 of the plan an apples-
to-apples "same architecture family, different capacity" comparison rather than two
unrelated designs.

Design notes (tied to Lecture 1 "Introduction to Deep Learning" and Lecture 2 "Deep
Learning Improvements"):
- Nonlinear activation (ReLU) between every hidden layer -- Lecture 1's "Nonlinear
  Activation Function: Importance" slide: strip the nonlinearity and a stack of
  Dense layers collapses to a single linear function no matter how deep it is, so
  depth only buys anything once ReLU sits between the layers. ReLU specifically
  (over sigmoid/tanh) because Lecture 1 also flags sigmoid/tanh's vanishing-gradient
  problem at the tails, which inhibits gradient propagation in a 4-layer-deep Heavy
  network; ReLU (roughly) sidesteps that.
- Output layer emits raw logits, no explicit Softmax module. Lecture 2's "Multi
  Class Classification" slides: output dimension = number of classes (8 here) and
  must be probability-normalized via Softmax so the outputs sum to 1 (the
  multi-class generalization of the binary sigmoid case) -- but `nn.CrossEntropyLoss`
  applies log-softmax internally in a numerically stable, fused way, so a separate
  `nn.Softmax` before it would double-apply the normalization (a common
  silently-wrong-but-still-training bug). Softmax is applied only at inference, when
  converting logits to class probabilities. (Lecture 2 also notes binary cross-
  entropy is a special case of this same categorical cross-entropy loss.)
- Weight initialization (Lecture 2's "Weight Initialization" slide) is set
  explicitly rather than left at PyTorch's default: He/Kaiming init on every hidden
  layer (the lecture's rule for ReLU-activated layers -- matches
  `tf.keras.initializers.VarianceScaling`), Xavier/Glorot init on the final output
  layer (the lecture's rule for a layer feeding a sigmoid/softmax, not ReLU --
  matches `tf.keras.initializers.GlorotUniform`). Both exist so that output
  variance roughly matches input variance layer-to-layer at the start of training,
  per the lecture's stated objective; PyTorch's own default init is a Kaiming
  variant tuned for a different assumption and isn't calibrated for either case here.
  All weights are randomized (never zero-initialized) since identical weights would
  all receive identical gradients, per the same slide.
- BatchNorm + Dropout after every hidden layer: the 46 input features span wildly
  different scales even after standardization (e.g. `Header_Length` has heavy-
  tailed outliers, flag counts are near-binary) -- BatchNorm stabilizes activations
  layer-to-layer, Dropout is Lecture 2's "Dropout Regularisation" (drop random
  neurons per training step so the network can't over-rely on one feature) and is
  the main defense against the Heavy lane overfitting the rare classes (Brute Force
  is ~1.5k rows vs. DDoS's ~206k). Both are active only in training mode
  (`model.train()`) and turned off at eval time (`model.eval()`), exactly as the
  lecture specifies for Dropout -- see train.py's `Trainer`, which switches modes.
- Heavy vs. Light is this project's concrete instance of Lecture 2's "Bias-Variance
  trade-off" slide: Heavy (4 hidden layers, 512-256-128-64, ~199k params) sits
  toward the low-bias/high-variance end -- enough capacity to fit the majority
  classes tightly, but (per the lecture's stated symptom) at real risk of
  overfitting the thin classes without the Dropout/early-stopping countermeasures
  above. Light (64-32, ~5.5k params) sits toward the high-bias end deliberately, so
  Section 05's "what accuracy is lost purely by shrinking the model" comparison
  reflects a genuine bias-variance trade rather than an undertrained strawman.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class MLPConfig:
    name: str
    hidden_dims: list[int]
    dropout: float = 0.3


HEAVY_CONFIG = MLPConfig(name="centralized_heavy", hidden_dims=[512, 256, 128, 64], dropout=0.3)
LIGHT_CONFIG = MLPConfig(name="centralized_light", hidden_dims=[64, 32], dropout=0.2)


class MLPClassifier(nn.Module):
    def __init__(self, in_features: int, num_classes: int, config: MLPConfig):
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []
        prev = in_features
        for h in config.hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(config.dropout),
            ]
            prev = h
        output_layer = nn.Linear(prev, num_classes)
        layers.append(output_layer)
        self.net = nn.Sequential(*layers)
        self._init_weights(output_layer)

    def _init_weights(self, output_layer: nn.Linear) -> None:
        """He init on ReLU-fed hidden layers, Xavier init on the final (softmax-fed)
        layer -- see the "Weight initialization" design note above."""
        for module in self.net:
            if isinstance(module, nn.Linear) and module is not output_layer:
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)
        nn.init.xavier_normal_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # raw logits; see module docstring

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
