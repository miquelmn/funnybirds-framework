from typing import Callable, Optional

import torch
from torch import nn

from captum.log import log_usage
from captum.attr._utils.attribution import GradientAttribution
from captum._utils.typing import TargetType, TensorOrTupleOfTensorsGeneric


class MultiGradientExplainer(GradientAttribution):

    def __init__(
            self,
            forward_func: Callable,
            loss_function = None,
            nb_classes = None,
            n_iter: int = 0,
            n_step: int = 0.1,
            absolute: bool = True
    ):
        super().__init__(forward_func)

        if nb_classes is None:
            nb_classes = MultiGradientExplainer.get_model_num_classes(forward_func)

        if nb_classes is None:
            raise ValueError("Number of classes could not be automatically determined. Please "
                             "provide 'nb_classes' argument.")

        self.__nb_classes = nb_classes
        self.__n_iter = n_iter
        self.__n_step = n_step
        self.__absolute = absolute


        if loss_function is None:
            loss_function = nn.CrossEntropyLoss()
        
        self.loss_function = loss_function

    @staticmethod
    def get_model_num_classes(model):
        """ Automatically detect the number of output classes from a PyTorch model.

        Args:
            model: PyTorch model (nn.Module)

        Returns:
            Integer representing the number of output classes.
        """
        # 1. Check for Hugging Face Transformers config
        if hasattr(model, 'config') and hasattr(model.config, 'num_labels'):
            return model.config.num_labels

        # 2. List of common layer names for the final output
        # 'fc' -> ResNet
        # 'classifier' -> VGG, AlexNet, DenseNet, EfficientNet
        # 'heads' -> ViT (Vision Transformer)
        # 'head' -> Some custom implementations
        common_attributes = ['fc', 'classifier', 'heads', 'head', 'linear']

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

        return (image - minimum) / (maximum-minimum)

    @log_usage(part_of_slo=True)
    def attribute(self,
                  inputs: TensorOrTupleOfTensorsGeneric,
                  target: TargetType = None,
                  additional_forward_args: Optional[object] = None):

        if target is None:
            target = torch.argmax(self.forward_func(inputs))
        directions = torch.arange(self.__nb_classes - 1, dtype=torch.int64)

        if target != self.__nb_classes - 1:
            directions[target] = self.__nb_classes - 1

        if len(inputs.shape) != 4:
            raise ValueError(f"Expected input tensor of 4 dimensions, found {inputs.shape}")

        mult_inputs = inputs.repeat(self.__nb_classes - 1, 1, 1, 1).requires_grad_(True)
        mult_output = self.forward_func(mult_inputs)

        directions = directions.to(torch.cuda.current_device())

        loss = self.loss_function(mult_output, directions)
        grad = torch.autograd.grad(loss, mult_inputs)[0]


        for _ in range(self.__n_iter):
            mult_inputs = mult_inputs + (grad * self.__n_step)
            mult_output = self.forward_func(mult_inputs)

            loss = self.loss_function(mult_output, directions)
            grad = torch.autograd.grad(loss, mult_inputs)[0]

        if self.__absolute:
            grad = torch.abs(grad)


        mins = torch.amin(grad, dim=(1, 2, 3), keepdim=True)
        maxs = torch.amax(grad, dim=(1, 2, 3), keepdim=True)
        grad = (grad - mins) / (maxs - mins)
        
        res = torch.mean(grad, dim=0)

        return res.detach()