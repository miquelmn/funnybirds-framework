""" Implementation of the Multi-Gradient Carlini-Wagner (Multi-Gradient CW) attribution method.

Multi-Gradient CW is an attribution method that generates adversarial examples for each class and
uses the perturbations to compute feature importance. The method optimizes a loss function that
combines the distance of the perturbation, the confidence of misclassification, and the complexity
of the perturbation.

Author: Miquel Miró-Nicolau, 2026 (UIB)
"""
from typing import Callable, Optional, Any, Union
import logging
import cv2
import numpy as np

import torch
from torch import nn, Tensor, optim

from tqdm.auto import tqdm

from .nmf import NMF

from captum.log import log_usage
from captum.attr._utils.attribution import GradientAttribution
from captum._utils.typing import TargetType, TensorOrTupleOfTensorsGeneric
from torch.autograd import Variable
from torchvision.transforms.functional import to_pil_image

EPS = 1e-12


def SubRegionDivision(image, region_size=30):
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()

    if len(image.shape) == 4:
        image = image[0] # we remove the batch
        image = np.moveaxis(image, 0, -1)

    element_sets_V = []

    slic = cv2.ximgproc.createSuperpixelSLIC(image, region_size=region_size, ruler = 20.0)
    slic.iterate(20)     # The number of iterations, the larger the better the effect

    label_slic = slic.getLabels()        # Get superpixel label
    number_slic = slic.getNumberOfSuperpixels()  # Get the number of superpixels

    for i in range(number_slic):
        img_copp = (label_slic == i)[:,:, np.newaxis].astype(int)
        element_sets_V.append(img_copp)
    return np.dstack(element_sets_V)


def _var2numpy(var):
    """Make Variable to numpy array. No transposition will be made.

    Args:
        var: Variable instance on whatever device

    Returns:

    """
    return var.data.cpu().numpy()


def entropy(expl):
    batch_size = expl.shape[0]
    expl = abs(expl).reshape(batch_size, -1)

    # Calculate entropy (using a small epsilon for stability)
    expl = expl / (torch.abs(expl).sum() + EPS)

    complexity = -(expl * torch.log(expl + EPS)).sum()

    return complexity



class MultiGradientCW(GradientAttribution):
    def __init__(
        self,
        forward_func: Callable,
        nb_classes=None,
        n_iter: int = 200,
        lr: float = 1e-2,
        absolute: bool = True,
        device=None,
        verbose: int = 0,
        weights_loss=None,
        super_pixel = None
    ):
        super().__init__(forward_func)

        if nb_classes is None:
            nb_classes = MultiGradientCW.get_model_num_classes(forward_func)

        if nb_classes is None:
            raise ValueError(
                "Number of classes could not be automatically determined. Please "
                "provide 'nb_classes' argument."
            )

        self.__nb_classes = nb_classes
        self.__n_iter = n_iter
        self.__absolute = absolute
        self.__lr = lr

        if weights_loss is None:
            weights_loss = (0, 1, 1) #L2, CLS, L1

        self.__weights_loss = weights_loss

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.__device = device

        if verbose > 2:
            logging.basicConfig(level=logging.DEBUG)

        self.__verbose = verbose
        if super_pixel is None or super_pixel == 'nmf':
            self.__super_pixel = self.nmf
        else:
            self.__super_pixel = SubRegionDivision

    def get_features(self, image):
        with torch.no_grad():
            features = self.forward_func.model.features(image)
            flat_features = features.permute(0, 2, 3, 1).contiguous().view((-1, features.size(1))) # NxCxHxW -> (N*H*W)xC

            return features, flat_features

    def nmf(self, image, k=10):
        with torch.no_grad():
            features, flat_features = self.get_features(image)
            W, _ = NMF(flat_features, k, device=self.__device, max_iter=50)

        clusters = W.cpu().view(features.size(0), features.size(2), features.size(3), k).permute(0,3,1,2) # (N*H*W)xK -> NxKxHxW
        clusters = torch.nn.functional.interpolate(clusters, size=(image.shape[2:]), mode='bilinear', align_corners=False) ## 14x14 -> 224x224
        clusters = torch.swapaxes(clusters[0], 0, -1)

        summation = torch.sum(clusters, dim=-1)
        summation = summation / summation.max()

        diff = 1 - summation

        maximum = torch.amax(clusters, dim=(0, 1)).reshape((1, 1, -1))
        clusters = clusters / maximum


        clusters = torch.dstack([clusters, diff])

        return clusters



    @staticmethod
    def get_model_num_classes(model):
        """Automatically detect the number of output classes from a PyTorch model.

        Args:
            model: PyTorch model (nn.Module)

        Returns:
            Integer representing the number of output classes.
        """
        # 1. Check for Hugging Face Transformers config
        if hasattr(model, "config") and hasattr(model.config, "num_labels"):
            return model.config.num_labels

        # 2. List of common layer names for the final output
        # 'fc' -> ResNet
        # 'classifier' -> VGG, AlexNet, DenseNet, EfficientNet
        # 'heads' -> ViT (Vision Transformer)
        # 'head' -> Some custom implementations
        common_attributes = ["fc", "classifier", "heads", "head", "linear"]

        for attr in common_attributes:
            if hasattr(model, attr):
                layer = getattr(model, attr)

                # Case A: The attribute is a direct Linear layer
                if isinstance(layer, nn.Linear):
                    return layer.out_features

                # Case B: The attribute is a Sequential container (e.g., VGG)
                # We assume the actual classification layer is the last one in the block
                if isinstance(layer, nn.Sequential):
                    # Iterate backwards through the sequential block to find the last Linear layer
                    for sub_layer in reversed(layer):
                        if isinstance(sub_layer, nn.Linear):
                            return sub_layer.out_features

        return None

    @staticmethod
    def __min_max(image):
        minimum = image.min()
        maximum = image.max()

        return (image - minimum) / (maximum - minimum)

    def __single_target(self, data, target, multi_channel = True):
        regions = self.__super_pixel(data)
        regions = torch.Tensor(regions).to(self.__device)
        if multi_channel:
            regions = regions[None, :, :, :].repeat((3,1,1,1)) # Per canal

        target_oh = torch.zeros(
            target.size() + (self.__nb_classes,), device=self.__device
        )  # Resulting shape: (B, C)
        target_oh[0, target] = 1.0
        target_oh.requires_grad_()

        noise_shape = (3, 1, 1, regions.shape[-1]) if multi_channel else (1, 1, regions.shape[-1])

        real_noise = torch.zeros(noise_shape, requires_grad=True, device=self.__device)
        optimizer = optim.Adam([real_noise], lr=self.__lr)

        for step in range(self.__n_iter):
            noise = torch.sum(real_noise * regions.detach(), dim=-1)

            adversarial = data + noise

            loss, adv_output = self.__cw_loss(adversarial, real_noise, target_oh)

            logging.debug(f"Step {step} - Loss function {loss}")

            if (
                torch.argmax(adv_output).detach().cpu() == target
            ):  # If we find the AE we stop
                break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return noise.detach()

    def __cw_loss(
        self, adversarial: Union[Tensor, Any], noise: Tensor, target_oh_var: Tensor
    ):
        adv_logits_output = self.forward_func(adversarial)
        dist = torch.linalg.vector_norm(noise)  # L2 Norm
        sparse = torch.linalg.vector_norm(noise, ord=1) # L1 norm

        target_activ_var = torch.sum(
            target_oh_var * adv_logits_output, 1
        )  # Logit del target
        inf = 1e4  # sadly pytorch does not work with np.inf; 1e4 is also used in Carlini's code

        assert (adv_logits_output.max(1)[0] >= -inf).all(), "assumption failed"

        maxother_activ_var = torch.max(
            ((1 - target_oh_var) * adv_logits_output - target_oh_var * inf), 1
        )[0]

        # Compute $f(x')$, where $x'$ is the adversarial example in image space.
        confidence = 0
        f_var = torch.clamp(maxother_activ_var - target_activ_var + confidence, min=0.0)

        w1, w2, w3 = self.__weights_loss
        loss = torch.sum(w1 * dist + w2 * f_var + w3 * sparse)
        if self.__verbose > 5:
            logging.debug(f"Loss function {loss}: "
                          f"\n\tDIST {dist} "
                          f"\n\tCHANGE {f_var} "
                          f"\n\tSPARSE {sparse}")

        # the total loss of current batch, should be of dimension [1]
        return loss, adv_logits_output  # type: Variable

    @log_usage(part_of_slo=True)
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType = None,
        additional_forward_args: Optional[object] = None,
    ):
        output = torch.zeros_like(inputs)
        output = output.repeat(self.__nb_classes, 1, 1, 1)

        max_target = torch.argmax(self.forward_func(inputs))

        if target is None:
            target = max_target

        if target != max_target:
            # Cas diferent, primer AE i explicam l'AE
            noise = self.__single_target(inputs, torch.Tensor([target]).int())
            inputs = inputs + noise

        aux = 0
        for i in tqdm(
            range(self.__nb_classes), disable=(self.__verbose < 1), desc="Contrastive"
        ):
            if i != target:
                i = torch.Tensor([i]).int()
                output[i - aux] = self.__single_target(inputs, i)
            else:
                aux = 1

        return _var2numpy(torch.mean(output, dim=0))
