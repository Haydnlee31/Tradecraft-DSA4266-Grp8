"""Isolated optional Laya runtime. Offline inputs contain no target labels.

Run with outputs/laya-runtime/bin/python; main project dependencies stay
unchanged. Public pretrained weights must already be downloaded and pinned.
No API call or upload occurs in this program. Truncated inputs fail closed.
"""
import argparse
import importlib.metadata
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import torch
import laya
from src.eval.official_evidence import digest, require, read_json
from src.data.label_map import CLASSES


def read(path):
    return read_json(Path(path).read_text())


def save(path, value):
    import json
    # Exclusive creation protects completed predictions from accidental reruns.
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    require(not args.output.exists(), 'Preserve existing output')
    request = read(args.inputs)
    spec = request['plan']['laya']
    require(request['classes'] == CLASSES and request['test_opened'] is False
            and spec['device'] == 'mps' and spec['batch_size'] == 4
            and torch.backends.mps.is_available(), 'Wrong request or MPS unavailable')
    checkpoint = {str(p.relative_to(args.checkpoint)): digest(p)
                  for p in sorted(args.checkpoint.rglob('*'))
                  if p.is_file() and '.cache' not in p.parts}
    require(checkpoint == request['checkpoint_files'], 'Checkpoint bytes changed')
    torch.set_num_threads(2)
    torch.manual_seed(7)
    started = time.perf_counter()
    agent = laya.load(str(args.checkpoint.resolve()), device='mps', fast=False, compile=False, backend='eager')
    require(str(agent.dtype_for(4)) == 'torch.float32', 'Unexpected autocast precision')
    questions = request['questions']
    internal = {k: agent._to_internal(v) for k, v in questions.items()}
    report = dict(status='running', input_sha256=digest(args.inputs), checkpoint_files=checkpoint,
        source_sha256=digest(__file__), test_opened=False, fine_tuned=False,
        runtime={p: importlib.metadata.version(p) for p in ['laya', 'torch', 'transformers', 'numpy', 'safetensors']},
        device=str(agent.device), inference_dtype=str(agent.dtype_for(4)), arms={})
    # Inspect every token sequence BEFORE any model prediction. Upstream's
    # private preflight method is tied to the pinned source revision.
    for mode, states in request['states'].items():
        lengths = []
        for state in states:
            item = agent._encode_state(state, list(questions), internal,
                        max_len=spec['max_len'], head_max_len=spec['head_max_len'])[0]
            require(not item['state_stats']['truncated'] and len(item['markers']) == 8
                    and item['options']['options_distinct'] == 8, 'Input or options truncated')
            lengths.append(len(item['ids']))
        report['arms'][mode] = dict(rows=len(states), max_tokens=max(lengths), min_tokens=min(lengths))
    print('PASS: both serializers fit the context; all eight options retained', flush=True)
    for mode, states in request['states'].items():
        predictions, probabilities = [], []
        begin = time.perf_counter()
        for start in range(0, len(states), 4):
            batch = agent.predict_batch(states[start:start+4], questions, batch_size=4,
                        max_len=spec['max_len'], head_max_len=spec['head_max_len'])
            for result in batch:
                require(not result['usage']['truncated'], 'Inference truncated input')
                answer = result['answers']['attack']
                predictions.append(CLASSES.index(answer['choice']))
                probabilities.append([answer['probabilities'][c] for c in CLASSES])
            if start % 128 == 0:
                print(f'{mode}: {len(predictions)}/{len(states)}', flush=True)
        torch.mps.synchronize()
        report['arms'][mode].update(predictions=predictions, rounded_probabilities=probabilities,
                                  tokenization_and_inference_seconds=time.perf_counter()-begin)
    require(agent.cpu_fallback_count == 0, 'Unexpected CPU fallback')
    require(checkpoint == {str(p.relative_to(args.checkpoint)): digest(p)
                  for p in sorted(args.checkpoint.rglob('*'))
                  if p.is_file() and '.cache' not in p.parts}, 'Checkpoint mutated')
    report.update(status='complete', cpu_fallback_count=agent.cpu_fallback_count,
                  elapsed_seconds=time.perf_counter()-started)
    save(args.output, report)
    print('PASS: bounded offline Laya inference complete; no test or audit scoring', flush=True)


if __name__ == '__main__':
    main()
