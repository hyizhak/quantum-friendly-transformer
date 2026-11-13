"""Measure branch signal and validation-only ablations of the FN1 models."""

import importlib.util
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F


ROOT = Path('/data/hyizhan/qft-fn1-search-20261113')
spec = importlib.util.spec_from_file_location('frozen_fn1', ROOT / 'unit_fn_search.py')
search = importlib.util.module_from_spec(spec)
spec.loader.exec_module(search)
search.base.configure()
plan, original, study = search.load_plan(ROOT)
validation = torch.load(original / 'data/genomic/validation.pt', weights_only=True)
sample = {k: v[torch.linspace(0, len(v) - 1, 32).long()].cuda() for k, v in validation.items()}


@torch.inference_mode()
def measure(model):
    network = getattr(model, 'network', model)
    rows, hooks = [], []
    for index, layer in enumerate(network.bert.encoder.layer):
        row = {'layer': index}
        rows.append(row)

        def attention_hook(module, args, row=row):
            h, cu = args[:2]
            length = int(cu[1] - cu[0])
            h = h[:length]
            if hasattr(module, 'Wq'):
                q, k = module.Wq(h), module.Wk(h)
                scale = math.sqrt(module.attention_head_size)
            else:
                q, k, _ = module.Wqkv(h).chunk(3, dim=-1)
                scale = module.attention_head_size ** -0.5
            q = q.reshape(length, module.num_attention_heads, -1).transpose(0, 1)
            k = k.reshape(length, module.num_attention_heads, -1).transpose(0, 1)
            scores = (q @ k.transpose(-1, -2)) * scale
            bias = args[5][0, :, :length, :length]
            centered = scores - scores.mean(-1, keepdim=True)
            centered_bias = bias - bias.mean(-1, keepdim=True)
            row['qk_key_centered_rms'] = centered.square().mean().sqrt().item()
            row['alibi_key_centered_rms'] = centered_bias.square().mean().sqrt().item()
            row['attention_change_from_alibi_only_l1'] = (
                (scores + bias).softmax(-1) - bias.softmax(-1)).abs().sum(-1).mean().item()
            row['attention_input_row_norm'] = h.norm(dim=-1).mean().item()

        def projection_hook(module, args, output, row=row, kind='attention'):
            signal = F.linear(args[0], module.weight, None)
            signal_norm = signal.norm(dim=-1).mean().item()
            bias_norm = module.bias.norm().item() if module.bias is not None else 0.
            row[kind + '_linear_signal_norm'] = signal_norm
            row[kind + '_bias_norm'] = bias_norm
            row[kind + '_bias_to_linear_signal'] = bias_norm / max(signal_norm, 1e-30)

        def mlp_hook(module, args, output, row=row):
            incoming = args[0]
            row['ffn_input_row_norm'] = incoming.norm(dim=-1).mean().item()
            row['ffn_output_row_norm'] = output.norm(dim=-1).mean().item()
            row['ffn_input_output_cosine'] = F.cosine_similarity(incoming, output, dim=-1).mean().item()
            row['ffn_residual_scale'] = float(getattr(module, 'residual_scale', 1.))

        hooks.append(layer.attention.self.register_forward_pre_hook(attention_hook))
        hooks.append(layer.attention.output.dense.register_forward_hook(projection_hook))
        hooks.append(layer.mlp.wo.register_forward_hook(
            lambda module, args, output, row=row: projection_hook(module, args, output, row, 'ffn')))
        hooks.append(layer.mlp.register_forward_hook(mlp_hook))
    model.eval()
    search.base.forward(model, sample, 'multilayer')
    for hook in hooks:
        hook.remove()
    return {'layers': rows, 'layer_means': {
        key: sum(row[key] for row in rows) / len(rows) for key in rows[0] if key != 'layer'}}


def student(trial):
    model = search.build(plan, original, study, trial).cuda()
    if (ROOT / trial['name'] / 'best.pt').exists():
        state = torch.load(ROOT / trial['name'] / 'best.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(state['model'], strict=True)
    return model


def evaluate(model):
    return search.base.evaluate(model, validation, search.JOB, 256)[:2]


report = {'sample': '32 evenly spaced validation rows; no test access', 'models': {}}
teacher_job = {**search.JOB, 'variant': 'vanilla'}
teacher, _ = search.base.build_model(study, original, teacher_job)
state = torch.load(plan['teacher_checkpoint'], map_location='cpu', weights_only=True)
teacher.load_state_dict(state['model'], strict=True)
del state
teacher.cuda()
report['models']['native_teacher'] = measure(teacher)
print('native_teacher', json.dumps(report['models']['native_teacher']['layer_means']), flush=True)
del teacher
torch.cuda.empty_cache()

initial_trial = {**plan['trials'][1], 'name': 'diagnostic_initial_teacher_to_fn1'}
model = student(initial_trial)
report['models']['teacher_converted_to_fn1'] = measure(model)
print('teacher_converted_to_fn1', json.dumps(report['models']['teacher_converted_to_fn1']['layer_means']), flush=True)
del model
torch.cuda.empty_cache()

model = student(plan['trials'][0])
report['models']['trained_fn1_recovery'] = measure(model)
print('trained_fn1_recovery', json.dumps(report['models']['trained_fn1_recovery']['layer_means']), flush=True)
baseline, original_predictions = evaluate(model)
report['ablations'] = {'baseline': baseline}
layers = model.network.bert.encoder.layer

for name in ['ffn_linear_signal_zero', 'qk_content_zero', 'attention_linear_signal_zero', 'ffn_branch_zero']:
    hooks, scales = [], []
    for layer in layers:
        if name == 'ffn_linear_signal_zero':
            hooks.append(layer.mlp.wo.register_forward_hook(
                lambda module, args, output: module.bias.expand_as(output)))
        elif name == 'qk_content_zero':
            hooks.append(layer.attention.self.Wq.register_forward_hook(
                lambda module, args, output: torch.zeros_like(output)))
        elif name == 'attention_linear_signal_zero':
            hooks.append(layer.attention.output.dense.register_forward_hook(
                lambda module, args, output: module.bias.expand_as(output)))
        else:
            scales.append(layer.mlp.residual_scale.detach().clone())
            with torch.no_grad():
                layer.mlp.residual_scale.zero_()
    metrics, predictions = evaluate(model)
    metrics['changed_predictions'] = int((predictions != original_predictions).sum())
    report['ablations'][name] = metrics
    print(name, json.dumps(metrics), flush=True)
    for hook in hooks:
        hook.remove()
    if scales:
        with torch.no_grad():
            for layer, value in zip(layers, scales):
                layer.mlp.residual_scale.copy_(value)

destination = Path(plan['repository']) / 'artifacts/rigorous_rerun/unit_fn_diagnostics.json'
destination.write_text(json.dumps(report, indent=2) + '\n')
print('saved', destination, flush=True)
