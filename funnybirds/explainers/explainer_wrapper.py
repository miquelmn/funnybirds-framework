import torch
import torch.nn as nn
from torchvision import transforms

import numpy as np
from PIL import Image

from funnybirds.models.ViT.ViT_explanation_generator import Baselines, LRP


def get_p_thresholds():
    return np.linspace(0.01, 0.50, num=80)


class AbstractExplainer:
    def __init__(self, explainer, baseline=None):
        """
        An abstract wrapper for explanations.

        Args:
            explainer: XAI method.
        """
        self.explainer = explainer
        self.explainer_name = type(self.explainer).__name__
        self.baseline = baseline
        print(self.explainer_name)

    def explain(self, input, target=None, *args, **kwargs):
        explainer = self.explainer
        if hasattr(explainer, 'attribute'):
            res = explainer.attribute(input, target, *args, **kwargs)
        elif callable(explainer):
            res = explainer(input, target, *args, **kwargs)
        elif hasattr(explainer, 'explain'):
            res = self.explainer.explain(input, target, *args, **kwargs)
        else:
            raise ValueError(f"Explainer not known.")

        return res

    def get_important_parts(self, image, part_map, target, colors_to_part, thresholds,
                            with_bg=False):
        """ Outputs parts of the bird that are important according to the explanation.

        This must be reimplemented for different explanation types. Output is of the
        form: ['beak', 'wing', 'tail']

        Args:
            image: the input image
            part_map: the corresponding segmentation map where one color denotes one part
            target: the target class
            colors_to_part: a list that maps colors to parts
            thresholds: the different thresholds to use to estimate which parts are important
            with_bg: include the background parts in the computation

        Returns:
                a list of important parts for each threshold, e.g. [['beak', 'wing'],
                ['beak', 'wing', 'tail'], ...]
        """
        assert image.shape[0] == 1  # B = 1
        attribution = self.explain(image, target=target)

        part_importances = self.get_part_importance(image, part_map, target, colors_to_part,
                                                    with_bg=with_bg)

        important_parts_for_thresholds = []
        for threshold in thresholds:
            important_parts = []
            for key in part_importances.keys():
                if part_importances[key] > (attribution.sum() * threshold):
                    important_parts.append(key)
            important_parts_for_thresholds.append(important_parts)
        return important_parts_for_thresholds

    def get_part_importance(self, image, part_map, target, colors_to_part, with_bg=False):
        """ Outputs part importances for each part.

        Args:
            image: the input image
            part map: the corresponding segmentation map where one color denotes one part
            target: the target class
            colors_to_part: a list that maps colors to parts
            with_bg: include the background parts in the computation


        """
        assert image.shape[0] == 1  # B = 1
        attribution = self.explain(image, target=target)

        part_importances = {}

        dilation1 = nn.MaxPool2d(5, stride=1, padding=2)
        for part_color in colors_to_part.keys():
            torch_color = torch.zeros(1, 3, 1, 1).to(image.device)
            torch_color[0, 0, 0, 0] = part_color[0]
            torch_color[0, 1, 0, 0] = part_color[1]
            torch_color[0, 2, 0, 0] = part_color[2]
            color_available = torch.all(part_map == torch_color, dim=1, keepdim=True).float()

            color_available_dilated = dilation1(color_available)
            attribution_in_part = attribution * color_available_dilated
            attribution_in_part = attribution_in_part.sum()

            part_string = colors_to_part[part_color]
            part_string = ''.join((x for x in part_string if x.isalpha()))
            if part_string in part_importances.keys():
                part_importances[part_string] += attribution_in_part.item()
            else:
                part_importances[part_string] = attribution_in_part.item()

        if with_bg:
            for i in range(50):
                torch_color = torch.zeros(1, 3, 1, 1).to(image.device)
                torch_color[0, 0, 0, 0] = 204
                torch_color[0, 1, 0, 0] = 204
                torch_color[0, 2, 0, 0] = 204 + i
                color_available = torch.all(part_map == torch_color, dim=1, keepdim=True).float()
                color_available_dilated = dilation1(color_available)

                attribution_in_part = attribution * color_available_dilated
                attribution_in_part = attribution_in_part.sum()

                bg_string = 'bg_' + str(i).zfill(3)
                part_importances[bg_string] = attribution_in_part.item()

        return part_importances

    def get_p_thresholds(self):
        return np.linspace(0.01, 0.50, num=80)



class ViTGradCamExplainer(AbstractExplainer):
    def __init__(self, model, explainer):
        super().__init__(explainer)
        self.model = model
        self.explainer = Baselines(self.model.model)

    def explain(self, input, target=None, *args, **kwargs):
        B, C, H, W = input.shape
        assert B == 1
        input_ = torch.nn.functional.interpolate(input, (224, 224))
        attribution = self.explainer.generate_cam_attn(input_, index=target).reshape(1, 1, 14, 14)
        m = transforms.Resize((H, W), interpolation=Image.NEAREST)
        attribution = m(attribution)
        print(attribution.shape)
        return attribution


class ViTRolloutExplainer(AbstractExplainer):
    def __init__(self, model):
        self.model = model
        super().__init__(Baselines(self.model.model))

    def explain(self, input, target=None, *args, **kwargs):
        B, C, H, W = input.shape
        assert B == 1
        input_ = torch.nn.functional.interpolate(input, (224, 224))
        attribution = self.explainer.generate_rollout(input_, start_layer=1).reshape(1, 1, 14, 14)
        m = transforms.Resize((H, W), interpolation=Image.NEAREST)
        attribution = m(attribution)
        return attribution


class ViTCheferLRPExplainer(AbstractExplainer):
    def __init__(self, model):
        self.model = model
        super().__init__(LRP(self.model.model))

    def explain(self, input, target=None, *args, **kwargs):
        B, C, H, W = input.shape
        assert B == 1
        input_ = torch.nn.functional.interpolate(input, (224, 224))
        attribution = self.explainer.generate_LRP(input_, index=target, start_layer=1,
                                                  method="transformer_attribution").reshape(1, 1,
                                                                                            14, 14)
        m = transforms.Resize((H, W), interpolation=Image.NEAREST)
        attribution = m(attribution)
        return attribution
