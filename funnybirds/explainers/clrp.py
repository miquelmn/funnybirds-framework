"""
This is implementation of Contrastive Layerwise Relevance Propagation (CLRP) in the paper:

Gu, Jindong, Yinchong Yang, and Volker Tresp."Understanding Individual Decisions of CNNs via Contrastive Backpropagation."
Asian Conference on Computer Vision. Springer, Cham, 2018.
"""

from funnybirds.explainers import clrp_lib


from PIL import Image
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable

# register forward and backward hook functions in Conv, MaxPooling, Linear layers of the loaded model
def register_hooks(model=None):
    first_detected = False
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            layer.register_forward_hook(clrp_lib.conv_forward_hook)
            if not first_detected:
                layer.register_backward_hook(clrp_lib.FirstConv_backward_hook)
                first_detected = True
            else:
                layer.register_backward_hook(clrp_lib.conv_forward_hook)
        elif isinstance(layer, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)):
            layer.register_forward_hook(clrp_lib.maxpool_forward_hook)
            layer.register_backward_hook(clrp_lib.maxpool_backward_hook)
        elif isinstance(layer, nn.Linear):
            layer.register_forward_hook(clrp_lib.linear_forward_hook)
            layer.register_backward_hook(clrp_lib.linear_backward_hook)

    return model

# to create a saliency map of the target signal or the contrastive signal
def clrp(out, target_class, contrastive_signal = False, CLRP = 'type1'):
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = "cpu"

    # initialize the relevance, i.e., the logit score
    nega_flag_ = False
    if contrastive_signal == False:
        R = torch.zeros(out.shape).to(device)
        R[0, target_class] = out[0, target_class].detach()
    else:
        if CLRP == 'type1': 
            R = out.detach()
            R[0, target_class] = 0
        elif CLRP == 'type2':
            R = torch.zeros(out.shape).to(device)
            R[0, target_class] = out[0, target_class].detach()
            nega_flag_ = True

    # (nega_flag_= False) for CLRP1 and (nega_flag_= True) for CLRP2
    clrp_lib.initialise_rel(torch.abs(R), Nega_flag_ = nega_flag_)

    # propagate the relevance score back, which is computed in registered hook functions
    out.backward(out, retain_graph=True)
    return clrp_lib.return_rel().cpu().data.numpy()


# visualize the saliency maps (LRP and CLRP)
def visualize(sm_lrp, sm_contrast, name):
    sm_lrp_ = (sm_lrp[0]/sm_lrp.max()).transpose((1, 2, 0))
    sm_lrp_ = np.maximum(0, sm_lrp_)
    Image.fromarray((sm_lrp_*255).astype(np.uint8), mode='RGB').save('SMs/'+ name + '_LRP.png')

    sm_lrp_ = sm_lrp/sm_lrp.sum()
    sm_contrast = sm_contrast/sm_contrast.sum()
    sm_clrp = sm_lrp_ - sm_contrast

    sm_clrp_ = (sm_clrp[0]/sm_clrp.max()).transpose((1, 2, 0))
    sm_clrp_ = np.maximum(0, sm_clrp_)
    Image.fromarray((sm_clrp_*255).astype(np.uint8), mode='RGB').save('SMs/'+ name + '_CLRP.png')

    print(name, 'Explaining Classification finished!')

def attribute(input, target, model):
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = "cpu"

    model = register_hooks(model=model)
    model = model.eval()
    model = model.to(device)


    input_var = Variable(input, requires_grad=True)
    out = model(input_var)

    return clrp(out, target, contrastive_signal = True, CLRP = 'type1')
