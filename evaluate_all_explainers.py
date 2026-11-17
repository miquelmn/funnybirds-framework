"""Sequentially evaluate several XAI methods on FunnyBirds and store the results in a TSV file.

Example:
    python evaluate_all_explainers.py --data ./data/FunnyBirds --model vgg16 \
        --checkpoint_name ./checkpoints/vgg16_final_1_checkpoint_best.pth.tar \
        --output results/vgg16.tsv --gpu 0

By default every explainer compatible with the chosen model is run and every protocol is
computed. Edit the EXPLAINERS dictionary to choose which explainers run (comment out entries to
skip them) and use --protocols to restrict the protocols. One row per explainer is appended to
the TSV as soon as that explainer finishes, so partial results survive a crash or a timeout.
"""
import argparse
import csv
import datetime
import gc
import os
import random
import traceback

import torch
from captum.attr import InputXGradient, IntegratedGradients, NoiseTunnel, Saliency

from funnybirds.evaluation_protocols import (
    accuracy_protocol,
    background_independence_protocol,
    controlled_synthetic_data_check_protocol,
    deletion_check_protocol,
    distractibility_protocol,
    preservation_check_protocol,
    single_deletion_protocol,
    target_sensitivity_protocol,
)
from funnybirds.explainers.explainer_wrapper import (
    AAAExplainer,
    AbstractExplainer,
    CaptumAttributionExplainer,
    CLRPExplainer,
    ColExplainer,
    ColXExplainer,
    GCWExplainer,
    MultiGradientExplainer,
    ViTCheferLRPExplainer,
    ViTRolloutExplainer,
)
from funnybirds.models.model_wrapper import StandardModel, ViTModel
from funnybirds.models.resnet import resnet50
from funnybirds.models.vgg import vgg16
from funnybirds.models.ViT.ViT_LRP import vit_base_patch16_224 as vit_LRP
from funnybirds.models.ViT.ViT_new import vit_base_patch16_224


def integrated_gradients_explainer(model):
    device = next(model.parameters()).device
    baseline = torch.zeros((1, 3, 256, 256)).to(device)
    return CaptumAttributionExplainer(IntegratedGradients(model), baseline=baseline)


def gradient_explainer(model):
    return CaptumAttributionExplainer(Saliency(model))


def smoothgrad_explainer(model):
    return CaptumAttributionExplainer(NoiseTunnel(Saliency(model)))


def input_x_gradient_explainer(model):
    return AbstractExplainer(InputXGradient(model))


# Explainers to evaluate: name -> callable(model) returning the explainer.
# Comment out entries to skip them.
EXPLAINERS = {
    'IntegratedGradients': integrated_gradients_explainer,
    'Gradient': gradient_explainer,
    'SG': smoothgrad_explainer,
    'InputXGradient': input_x_gradient_explainer,
    'AAA': AAAExplainer,
    'MGE': MultiGradientExplainer,
    'CLRP': CLRPExplainer,
    'GCW': GCWExplainer,
    'COL': ColExplainer,
    'COLX': ColXExplainer,
    'Rollout': ViTRolloutExplainer,
    'CheferLRP': ViTCheferLRPExplainer,
}
# Explainers in EXPLAINERS that only work with ViT models (skipped for other architectures).
VIT_ONLY_EXPLAINERS = {'Rollout', 'CheferLRP'}

ALL_PROTOCOLS = ['accuracy', 'controlled_synthetic_data_check', 'single_deletion',
                 'preservation_check', 'deletion_check', 'target_sensitivity',
                 'distractibility', 'background_independence']

TSV_COLUMNS = ['timestamp', 'model', 'checkpoint', 'explainer', 'seed',
               'accuracy', 'csdc', 'pc', 'dc', 'distractibility', 'background_independence',
               'sd', 'ts', 'best_threshold', 'runtime_s', 'status', 'error']

parser = argparse.ArgumentParser(description='DataFB - Sequential explanation evaluation')
parser.add_argument('--data', metavar='DIR', required=True, help='path to dataset')
parser.add_argument('--model', required=True, choices=['resnet50', 'vgg16', 'vit_b_16'],
                    help='model architecture')
parser.add_argument('--checkpoint_name', type=str, default=None,
                    help='checkpoint name (including dir)')
parser.add_argument('--protocols', nargs='+', default=ALL_PROTOCOLS, choices=ALL_PROTOCOLS,
                    help='protocols to compute (default: all)')
parser.add_argument('--output', type=str, default='results/results.tsv',
                    help='TSV file where results are appended')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
parser.add_argument('--seed', default=0, type=int, help='seed')
parser.add_argument('--batch_size', default=32, type=int,
                    help='batch size for protocols that do not require custom BS such as accuracy')
parser.add_argument('--nr_itrs', default=2501, type=int,
                    help='number of iterations for protocols that use it')
parser.add_argument('--skip_done', default=False, action='store_true',
                    help='skip explainers that already have a successful row for this model/checkpoint '
                         'in the output TSV')


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def build_model(model_name, explainer_name, checkpoint_name, device):
    if model_name == 'resnet50':
        model = StandardModel(resnet50(num_classes=50))
    elif model_name == 'vgg16':
        model = StandardModel(vgg16(num_classes=50))
    elif model_name == 'vit_b_16':
        if explainer_name == 'CheferLRP':
            model = ViTModel(vit_LRP(num_classes=50))
        else:
            model = ViTModel(vit_base_patch16_224(num_classes=50))
    else:
        raise ValueError(f'Model not implemented: {model_name}')

    if checkpoint_name:
        state = torch.load(checkpoint_name, map_location=torch.device('cpu'))
        model.load_state_dict(state['state_dict'])
    model = model.to(device)
    model.eval()
    return model


def select_best_threshold(threshold_metrics):
    """Pick the threshold maximising csdc/3 + pc/3 + dc/3 + distractibility (as in the original
    script). Metrics that were not computed are simply left out of the sum."""
    weights = {'csdc': 1 / 3., 'pc': 1 / 3., 'dc': 1 / 3., 'distractibility': 1.}
    available = {k: v for k, v in threshold_metrics.items() if v is not None}
    if not available:
        return None
    thresholds = next(iter(available.values())).keys()
    best_threshold, max_score = None, float('-inf')
    for threshold in thresholds:
        score = sum(weights[k] * v[threshold] for k, v in available.items())
        if score > max_score:
            max_score, best_threshold = score, threshold
    return best_threshold


def fmt(value):
    if value is None:
        return ''
    if isinstance(value, float):
        return f'{value:.5f}'
    return value


def append_row(path, row):
    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=TSV_COLUMNS, delimiter='\t')
        if new_file:
            writer.writeheader()
        writer.writerow({k: fmt(row.get(k)) for k in TSV_COLUMNS})


def already_done(path, model, checkpoint):
    if not os.path.exists(path):
        return set()
    with open(path, newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        return {r['explainer'] for r in reader
                if r.get('status') == 'ok' and r.get('model') == model
                and r.get('checkpoint') == (checkpoint or '')}


def evaluate_explainer(explainer_name, args, device, model_only_cache):
    protocols = set(args.protocols)
    set_seed(args.seed)

    model = build_model(args.model, explainer_name, args.checkpoint_name, device)
    explainer = EXPLAINERS[explainer_name](model)
    kw = {'args': args, 'device': device}

    res = {}

    # Model-only metrics do not depend on the explainer: compute them once and reuse.
    # (CheferLRP on ViT uses a different model implementation, so it gets its own cache entry.)
    cache_key = 'lrp' if (args.model == 'vit_b_16' and explainer_name == 'CheferLRP') else 'std'
    cache = model_only_cache.setdefault(cache_key, {})
    if 'accuracy' in protocols:
        if 'accuracy' not in cache:
            print('Computing accuracy...')
            cache['accuracy'] = round(accuracy_protocol(model, args), 5)
        res['accuracy'] = cache['accuracy']
    if 'background_independence' in protocols:
        if 'background_independence' not in cache:
            print('Computing background independence...')
            cache['background_independence'] = round(
                background_independence_protocol(model, **kw), 5)
        res['background_independence'] = cache['background_independence']

    threshold_metrics = {'csdc': None, 'pc': None, 'dc': None, 'distractibility': None}
    if 'controlled_synthetic_data_check' in protocols:
        print('Computing controlled synthetic data check...')
        threshold_metrics['csdc'] = controlled_synthetic_data_check_protocol(model, explainer, **kw)
    if 'target_sensitivity' in protocols:
        print('Computing target sensitivity...')
        res['ts'] = round(target_sensitivity_protocol(model, explainer, **kw), 5)
    if 'single_deletion' in protocols:
        print('Computing single deletion...')
        res['sd'] = round(single_deletion_protocol(model, explainer, **kw), 5)
    if 'preservation_check' in protocols:
        print('Computing preservation check...')
        threshold_metrics['pc'] = preservation_check_protocol(model, explainer, **kw)
    if 'deletion_check' in protocols:
        print('Computing deletion check...')
        threshold_metrics['dc'] = deletion_check_protocol(model, explainer, **kw)
    if 'distractibility' in protocols:
        print('Computing distractibility...')
        threshold_metrics['distractibility'] = distractibility_protocol(model, explainer, **kw)

    best_threshold = select_best_threshold(threshold_metrics)
    if best_threshold is not None:
        res['best_threshold'] = float(best_threshold)
        for name, values in threshold_metrics.items():
            if values is not None:
                res[name] = round(float(values[best_threshold]), 5)

    del explainer, model
    return res


def main():
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    explainers = list(EXPLAINERS)
    if args.model != 'vit_b_16':
        skipped = [e for e in explainers if e in VIT_ONLY_EXPLAINERS]
        if skipped:
            print(f'Skipping ViT-only explainers for {args.model}: {skipped}')
        explainers = [e for e in explainers if e not in VIT_ONLY_EXPLAINERS]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    if args.skip_done:
        done = already_done(args.output, args.model, args.checkpoint_name)
        if done:
            print(f'Already evaluated, skipping: {sorted(done)}')
        explainers = [e for e in explainers if e not in done]

    print(f'Evaluating {len(explainers)} explainers on {args.model}: {explainers}')
    print(f'Results will be appended to {args.output}')

    model_only_cache = {}
    for i, explainer_name in enumerate(explainers, 1):
        print(f'\n===== [{i}/{len(explainers)}] {explainer_name} =====')
        start = datetime.datetime.now()
        row = {
            'timestamp': start.isoformat(timespec='seconds'),
            'model': args.model,
            'checkpoint': args.checkpoint_name or '',
            'explainer': explainer_name,
            'seed': args.seed,
        }
        try:
            row.update(evaluate_explainer(explainer_name, args, device, model_only_cache))
            row['status'] = 'ok'
        except Exception as e:  # keep going with the next explainer
            traceback.print_exc()
            row['status'] = 'error'
            row['error'] = f'{type(e).__name__}: {e}'.replace('\t', ' ').replace('\n', ' ')
        row['runtime_s'] = round((datetime.datetime.now() - start).total_seconds(), 1)

        append_row(args.output, row)
        print('Result:', {k: row.get(k) for k in TSV_COLUMNS if row.get(k) not in (None, '')})

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print('\nAll explainers evaluated.')


if __name__ == '__main__':
    main()
