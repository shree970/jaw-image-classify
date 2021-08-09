"""
GradCAM code is taken from
https://github.com/kazuto1011/grad-cam-pytorch/blob/fd10ff7fc85ae064938531235a5dd3889ca46fed/grad_cam.py
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from utils import helper


class GradCAM:
    """Class for extracting activations and
    registering gradients from targetted intermediate layers
    target_layers = list of convolution layer index as shown in summary
    """

    def __init__(self, model, candidate_layers=None):
        def save_fmaps(key):
            def forward_hook(module, input, output):
                self.fmap_pool[key] = output.detach()

            return forward_hook

        def save_grads(key):
            def backward_hook(module, grad_in, grad_out):
                self.grad_pool[key] = grad_out[0].detach()

            return backward_hook

        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = []
        self.fmap_pool = {}
        self.grad_pool = {}
        self.candidate_layers = candidate_layers

        # If any candidates are not specified, the hook is registered to all the layers.
        for name, module in self.model.named_modules():
            if self.candidate_layers is None or name in self.candidate_layers:
                self.handlers.append(module.register_forward_hook(save_fmaps(name)))
                self.handlers.append(module.register_backward_hook(save_grads(name)))

    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.nll).to(self.device)
        one_hot.scatter_(1, ids, 1.0)
        return one_hot

    def forward(self, image):
        """
        Forward function.
        """
        self.image_shape = image.shape[2:]
        self.nll = self.model(image)
        return self.nll.sort(dim=1, descending=True)  # ordered results

    def backward(self, ids):
        """
        Class-specific backpropagation
        """
        one_hot = self._encode_one_hot(ids)
        self.model.zero_grad()
        self.nll.backward(gradient=one_hot, retain_graph=True)

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def generate(self, target_layer):
        """
        Generate GradCAM for a target layer.
        """
        fmaps = self._find(self.fmap_pool, target_layer)
        grads = self._find(self.grad_pool, target_layer)
        weights = F.adaptive_avg_pool2d(grads, 1)

        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        gcam = F.relu(gcam)
        gcam = F.interpolate(gcam, self.image_shape, mode="bilinear", align_corners=False)

        B, C, H, W = gcam.shape
        gcam = gcam.view(B, -1)
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        gcam /= gcam.max(dim=1, keepdim=True)[0]
        gcam = gcam.view(B, C, H, W)

        return gcam


def generate_gradcam(misclassified_images, model, target_layers, device):
    """
    Generate GradCAM for misclassified images.
    """
    images = []
    labels = []
    for img, _, correct in misclassified_images:
        images.append(img)
        labels.append(correct)

    model.eval()

    images = torch.stack(images).to(device)

    gcam = GradCAM(model, target_layers)

    probs, ids = gcam.forward(images)

    ids_ = torch.LongTensor(labels).view(len(images), -1).to(device)

    gcam.backward(ids=ids_)
    layers = []

    for target_layer in target_layers:
        print(f"Generating Grad-CAM for {target_layer}")
        layers.append(gcam.generate(target_layer=target_layer))

    gcam.remove_hook()
    return layers, probs, ids


def plot_gradcam(gcam_layers, target_layers, class_names, image_size, predicted, misclassified_images):
    """
    Plotting GradCAM output for misclassified images.
    """
    images = []
    labels = []
    for i, (img, _, correct) in enumerate(misclassified_images):
        images.append(img)
        labels.append(correct)

    c = len(images) + 1
    r = len(target_layers) + 2
    fig = plt.figure(figsize=(30, 14))
    fig.subplots_adjust(hspace=0.01, wspace=0.01)
    ax = plt.subplot(r, c, 1)
    ax.text(0.3, -0.5, "INPUT", fontsize=14)
    plt.axis("off")

    for i, target_layer in enumerate(target_layers):
        ax = plt.subplot(r, c, c * (i + 1) + 1)
        ax.text(0.3, -0.5, target_layer, fontsize=14)
        plt.axis("off")

        for j, image in enumerate(images):
            img = np.uint8(255 * helper.unnormalize(image.view(image_size)))
            if i == 0:
                ax = plt.subplot(r, c, j + 2)
                ax.text(
                    0, 0.2, f"actual: {class_names[labels[j]]} \npredicted: {class_names[predicted[j][0]]}", fontsize=12
                )
                plt.axis("off")
                plt.subplot(r, c, c + j + 2)
                plt.imshow(img)
                plt.axis("off")

            heatmap = 1 - gcam_layers[i][j].cpu().numpy()[0]
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            superimposed_img = cv2.resize(cv2.addWeighted(img, 0.5, heatmap, 0.5, 0), (128, 128))
            plt.subplot(r, c, (i + 2) * c + j + 2)
            plt.imshow(superimposed_img, interpolation="bilinear")

            plt.axis("off")
    plt.show()
