import torch
from torch import nn


class Col:
    def __init__(
            self,
            explainer,
            nb_classes,
            absolute = True,
            softmax = True,
            device = None,
            max_batch = 10,
            eps = 1e-12,
            aggregate = True,
            contrastive = False,
            normalization = "XAI" # XAI => Normalize each explanation | "IMG" => Normalize the whole image 
    ):
        model = explainer.forward_func
      
        if not Col.__is_softmax(model) and softmax:
            if model.__class__.__name__ == "StandardModel":
                model.model = nn.Sequential(model.model, nn.Softmax(dim=1))
            else:
                model = nn.Sequential(model, nn.Softmax(dim=1))

        explainer.forward_func = model

        self._absolute = absolute
        self._explainer = explainer
        self._nb_classes = nb_classes
        self._eps = eps

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._device = device
        self._max_batch =  max(1, int(max_batch))
        self._all_classes = torch.arange(nb_classes, dtype=torch.int64, device=device)
        self._aggregate = aggregate
        self._contrastive = contrastive
        self._normalization = normalization

    @property
    def model(self):
        return self._explainer.forward_func

    @staticmethod
    def __is_softmax(model):
        if model.__class__.__name__ == "StandardModel":
            return Col.__is_softmax(model.model)
        if isinstance(model, nn.Sequential):
            return len(model) > 0 and isinstance(model[-1], nn.Softmax)
        if isinstance(model, nn.Module):
            children = list(model.children())
            return len(children) > 0 and isinstance(children[-1], nn.Softmax)
        
        return False

    def __explain_tensor(self, inputs, targets):
        explanations = self._explainer.attribute(inputs, target=targets)

        if self._absolute:
            explanations.abs_()

        if self._normalization == "XAI":
            mins = torch.amin(explanations, dim=(1, 2, 3), keepdim=True)
            maxs = torch.amax(explanations, dim=(1, 2, 3), keepdim=True)

            explanations.sub_(mins) # / (maxs - mins)
            explanations.div_((maxs - mins).clamp_(min=self._eps))

        if self._aggregate:
            return explanations.sum(dim=0, keepdim=True).detach()
        else:
            return explanations.detach()

    def __attribute_single(
            self,
            inputs,
            target,
    ):
        """ Compute the COL explanation for a given input and target class.

        Args:
            inputs:
            target:

        Returns:
            aggregate=True : (1, C, H, W), mean over the explained directions.
            aggregate=False: (n_dirs, C, H, W), one normalized map per direction
                             (row order given by `directions(target)`).
        """

        if len(inputs.shape) != 4:
            raise ValueError(
                f"Expected input tensor of 4 dimensions, found {inputs.shape}"
            )
        
        all_classes = self._all_classes.to(inputs.device)

        if not 0 <= int(target) < self._nb_classes:
            raise ValueError(
                f"Target class {target} is out of range for {self._nb_classes} classes"
            )

        if self._contrastive:
            directions = all_classes[:]
        else:
            directions = torch.cat((all_classes[:target], all_classes[target + 1:])) # elimina el target

        if self._aggregate:
            explanation = torch.zeros_like(inputs)
        else:
            explanation = torch.zeros(
                (self._nb_classes,) + inputs.shape[1:],
                device=inputs.device,
                dtype=inputs.dtype
            )

        for i, dirs in enumerate(torch.split(directions, self._max_batch, dim=0)):
            chunk = inputs.detach().expand(len(dirs), -1, -1, -1).clone()
            chunk_xai = self.__explain_tensor(chunk, dirs)

            if self._aggregate:
                explanation += chunk_xai
            else:
                start = i * self._max_batch
                explanation[start:start + len(dirs)] = chunk_xai

        if self._aggregate:
            explanation /= len(directions)

        if self._normalization == "IMG":
            explanation.div_((explanation.max()).clamp_(min=self._eps))
        
        return explanation.detach()

    def attribute(
            self,
            input,
            target,
    ):
        """ Compute the COL explanation for a given input and target class.

        Args:
            inputs:
            target:

        Returns:
            aggregate=True : (N, C, H, W)
            aggregate=False: (N, n_dirs, C, H, W)
        """
        # One host transfer for all targets instead of one sync per sample.
        if isinstance(target, torch.Tensor):
            targets = target.reshape(-1).tolist()
        elif isinstance(target, int):
            targets = [target] * input.shape[0]
        else:
            targets = [int(t) for t in target]
 
        if len(targets) != input.shape[0]:
            raise ValueError(
                f"Got {input.shape[0]} inputs but {len(targets)} targets"
            )
 
        attr = [
            self.__attribute_single(input[i:i + 1], targets[i])
            for i in range(input.shape[0])
        ]

        if self._aggregate:
            return torch.cat(attr, dim=0)
        else:
            return torch.stack(attr, dim=0)


    def get_name(self):
        return self._explainer.get_name() + "_col"