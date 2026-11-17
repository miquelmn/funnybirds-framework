import numpy as np

import torch
from torch import nn
import torch.optim as optim

from sklearn import preprocessing
from art.estimators.classification import PyTorchClassifier
from art.attacks import evasion


class AdversarialExplainer:
    def __init__(
            self,
            model,
            nb_classes = None,
            input_shape = None,
            methods = None,
            difference_fn = None,
            scaler = None,
    ) -> None:
        if input_shape is None:
            input_shape = (1, 1, 1)

        if nb_classes is None and isinstance(model, nn.Module):
                nb_classes = AdversarialExplainer.get_model_num_classes(model)

        if nb_classes is None:
            raise ValueError("Number of classes could not be automatically determined. Please "
                             "provide 'nb_classes' argument.")

        if isinstance(model, nn.Module):
            model = AdversarialExplainer.__make_art_classifier(
                model,
                input_shape=input_shape,
                nb_classes=nb_classes
            )
        self.model = model
        self.__nb_classes = nb_classes
        
        if difference_fn is None:
            difference_fn = lambda x, y: abs(x-y)
        self.difference_fn = difference_fn

        if scaler is None:
            scaler = preprocessing.MinMaxScaler()
        self.scaler = scaler

        if methods is None:
            methods = [
                evasion.FastGradientMethod(model, norm=2, targeted=True, minimal=True, batch_size=nb_classes-1),
                evasion.ProjectedGradientDescent(model, norm=2, targeted=True, batch_size=nb_classes-1, verbose=False),
                # evasion.CarliniLInfMethod(model, targeted=True, verbose=True, max_iter=3, learning_rate=0.05, batch_size=nb_classes-1),
                # evasion.CarliniL0Method(model, targeted=True, verbose=True, max_iter=3, learning_rate=0.05, batch_size=nb_classes-1)
            ]

        self.methods = methods
        

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
    def __make_art_classifier(model, input_shape, nb_classes, *args, **kwargs):
        loss = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        with torch.cuda.device(1):
            model = PyTorchClassifier(
                model=model,
                optimizer=optimizer,
                loss=loss,
                input_shape=input_shape,
                nb_classes=nb_classes,
                *args,
                **kwargs
            )

        return model

    @staticmethod
    def __to_zero_one(image):
        minimum = image.min()
        maximum = image.max()

        return (image - minimum) / (maximum-minimum)
        

    def __call__(
            self,
            image,
            target = None,
            *args,
            **kwargs
    ):
        """
        Generate adversarial examples and compute differences across multiple attack methods.

        This method generates adversarial examples using the attack methods specified during
        initialization and computes per-pixel differences between original and adversarial images.

        Args:
            input_imgs (np.ndarray): Input images with shape (n_images, height, width, channels).
            ground_truth (np.ndarray): Ground truth labels for each image. Shape: (n_images,).
            target
        """
        if isinstance(image, torch.Tensor):
            if image.is_cuda:
                image = image.detach().cpu()
            image = image.numpy()

        explanation = np.zeros_like(image)
        with torch.cuda.device(1):
            predictions = self.model.predict(image)
            predictions = np.argmax(predictions, axis=1)

            image_input = np.repeat(image, self.__nb_classes-1, axis=0)
            target = np.arange(self.__nb_classes-1)

            if predictions != (self.__nb_classes-1):
                target[predictions] = self.__nb_classes-1
            
            explanation_single = np.zeros_like(image_input)
            for method in self.methods:
                adversarial = method.generate(x=image_input, y=target)

                #predictions_a = self.model.predict(adversarial)
                #predictions_a = np.argmax(predictions_a, axis=1)
    
                #diff_out = ((predictions - predictions_a) != 0).astype(np.float32)
                difference = self.difference_fn(adversarial, image_input)# * diff_out[:, None, None, None]
    
                explanation_single = explanation_single + (difference / (self.__nb_classes - 1))
                explanation = explanation + (explanation_single / len(self.methods))
    
            explanation = np.mean(np.sum(explanation, axis=0), axis=0) # Importance is equal to the mean of the channel importance
            explanation = AdversarialExplainer.__to_zero_one(explanation)
        
        return explanation
