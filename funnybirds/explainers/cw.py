from typing import Callable, Optional, Any, Union
import logging

import torch
from torch import nn, Tensor, optim

from tqdm.auto import tqdm

from captum.log import log_usage
from captum.attr._utils.attribution import GradientAttribution
from captum._utils.typing import TargetType, TensorOrTupleOfTensorsGeneric
from torch.autograd import Variable

EPS = 1e-12

def _var2numpy(var):
    """ Make Variable to numpy array. No transposition will be made.

    Args:
        var: Variable instance on whatever device

    Returns:

    """
    return var.data.cpu().numpy()

def complexity_bhat(expl):
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
        n_iter: int = 10000,
        lr: float = 1e-3,
        absolute: bool = True,
        device = None,
        verbose: int = 0,
        weights_loss = None,
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
            weights_loss = (1, 1, 1)

        self.__weights_loss = weights_loss

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.__device = device
        
        if verbose > 5:
            logging.basicConfig(level=logging.DEBUG)

        self.__verbose = verbose
        
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

    def __single_target(self, data, target):
        target_oh = torch.zeros(target.size() + (self.__nb_classes,), device=self.__device) # Resulting shape: (B, C)
        target_oh[0, target] = 1.0
        target_oh.requires_grad_()

        noise = torch.zeros(data.shape, requires_grad=True, device=self.__device)
        optimizer = optim.Adam([noise], lr=self.__lr)

        for step in range(self.__n_iter):
            adversarial = data + noise
            loss, adv_output = self.__cw_loss(adversarial, noise, target_oh)

            logging.debug(f"Step {step}: Cls0({adv_output[0, 0]}) - Cls{target}({adv_output[0, target]})")
            if torch.argmax(adv_output).detach().cpu() == target: # If we find the AE we stop
                break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # noise_t = torch.Tensor(noise).detach()
        # first_q = torch.quantile(abs(noise_t), torch.Tensor([0.5]))
        # noise_t[abs(noise_t) < first_q] = 0
        # adversarial = data + noise_t

        # res = self.forward_func(adversarial)
        # print(f"{res[0, 0:2]}")

        return noise.detach()


    def __cw_loss(self, adversarial: Union[Tensor, Any], noise: Tensor, target_oh_var: Tensor):
        adv_logits_output = self.forward_func(adversarial)
        dist = torch.linalg.vector_norm(noise)  # L2 Norm

        target_activ_var = torch.sum(target_oh_var * adv_logits_output, 1)  # Logit del target
        inf = 1e4  # sadly pytorch does not work with np.inf;
        # 1e4 is also used in Carlini's code
        # In Carlini's code, `maxother_activ_var` is called `other`.
        # It should be a Variable of tensor of dimension [B], such that the
        # `maxother_activ_var[i]` is the maximum final activation of all classes
        # other than class $t$, where $t$ is the attack target or the image
        # label.
        #
        # The assertion here ensures (sufficiently yet not necessarily) the
        # assumption behind the trick to get `maxother_activ_var` holds, that
        # $\max_{i \ne t}{o_i} \ge -\text{_inf}$, where $t$ is the target and
        # $o_i$ the $i$th element along axis=1 of `adv_logits_output`.
        #

        assert (adv_logits_output.max(1)[0] >= -inf).all(), 'assumption failed'

        maxother_activ_var = torch.max(((1 - target_oh_var) * adv_logits_output
                                        - target_oh_var * inf), 1)[0]

        # Compute $f(x')$, where $x'$ is the adversarial example in image space.
        confidence = 0
        f_var = torch.clamp(maxother_activ_var - target_activ_var + confidence, min=0.0)

        f_complexity = complexity_bhat(noise)

        w1, w2, w3 = self.__weights_loss
        loss = torch.sum(w1 * dist + w2 * f_var + w3 * f_complexity)

        # the total loss of current batch, should be of dimension [1]
        return loss, adv_logits_output   # type: Variable

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
        for i in tqdm(range(self.__nb_classes), disable=(self.__verbose < 1), desc="Contrastive"):
            if i != target:
                i = torch.Tensor([i]).int()
                output[i - aux] = self.__single_target(inputs, i)
            else:
                aux = 1

        return _var2numpy(torch.mean(output, dim=0))
