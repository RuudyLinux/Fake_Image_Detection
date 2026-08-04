"""
Grad-CAM for the trained EfficientNet-B4 fake-image classifier.

Hooks the last conv block (backbone.features[-1]) to get activations + their
gradients w.r.t. the predicted class logit, then combines them into a
localization heatmap: which regions pushed the model toward its decision.
"""

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """Returns a [0, 1]-normalized CAM at the feature map's resolution."""
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        score = logits[0, class_idx]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = cam.squeeze().detach().cpu().numpy()

        cam -= cam.min()
        peak = cam.max()
        if peak > 1e-8:
            cam /= peak
        return cam


def build_gradcam(model) -> GradCAM:
    target_layer = model.backbone.features[-1]
    return GradCAM(model, target_layer)
