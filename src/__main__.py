import random

import numpy as np
import torch

from evaluation_protocols import (controlled_synthetic_data_check_protocol,
                                  single_deletion_protocol, \
    preservation_check_protocol, deletion_check_protocol, target_sensitivity_protocol, \
    distractibility_protocol)

from src.explainers import explainer_wrapper


random.seed(0)
torch.manual_seed(0)

def __explainer_2_fb(explainer):
    return explainer_wrapper.AbstractExplainer(explainer)

def __main__(model, explainer, data, device = None):
    """ Main function to evaluate explainability of a model using Funnybirds framework.

    Funnybirds framework evaluates explainability of a model using a combination of several
    protocols, including:
        - Controlled Synthetic Data Check Protocol
        - Target Sensitivity Protocol
        - Single Deletion Protocol
        - Preservation Check Protocol
        - Deletion Check Protocol
        - Distractibility Protocol

    The authors then aggregate the scores from these protocols to compute a final explainability
    score (mx) for the model.

    Args:
        model: PyTorch model to be evaluated.
        explainer: Explainer object that provides explanations for the model's predictions.
        data: String path to the folder containing the dataset
        device: PyTorch device to run the evaluation on (e.g., 'cpu' or 'cuda').

    Returns:
        mx: The final explainability score computed as the mean of various protocol scores.
    """
    if device is None:
        device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")

    model = model.to(device)
    model.eval()

    csdc = controlled_synthetic_data_check_protocol(model, explainer)
    ts = target_sensitivity_protocol(model, explainer, data=data, device=device)
    ts = round(ts, 5)
    sd = single_deletion_protocol(model, explainer, data=data, device=device)
    sd = round(sd, 5)

    pc = preservation_check_protocol(model, explainer, data=data, device=device)
    dc = deletion_check_protocol(model, explainer, data=data, device=device)
    distractibility = distractibility_protocol(model, explainer, data=data, device=device)

    max_score = 0
    best_threshold = -1
    for threshold in csdc.keys():
        max_score_tmp = csdc[threshold] / 3. + pc[threshold] / 3. + dc[threshold] / 3. + \
                        distractibility[threshold]
        if max_score_tmp > max_score:
            max_score = max_score_tmp
            best_threshold = threshold

    compl = np.mean([
        round(csdc[best_threshold], 5),
        round(pc[best_threshold], 5),
        round(dc[best_threshold], 5),
        round(distractibility[best_threshold], 5)
    ])
    mx = np.mean([compl, sd, ts])

    return mx
