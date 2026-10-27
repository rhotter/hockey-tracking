"""Retain original crop examples while adapting to sequence-verified hard cases."""
from pathlib import Path
import time
from distill import read, write, sha, classification, BASE, OUT, SOURCE


def train(commit):
    import cv2
    import numpy as np
    import torch
    from torch import nn
    from crop_verifier import SpatialBallVerifier, patch_inputs
    torch.set_num_threads(4); cv2.setNumThreads(1)
    started = time.monotonic()
    protocol = read('/work/distill-replay-protocol.json')
    teacher = read('/work/distill-labels.json')['rows']
    transfer = read('/work/distill-transfer-inputs.json')['rows']
    assert sha('/work/distill-labels.json') == protocol['teacher_labels_sha256']
    assert sha('/work/distill-transfer-inputs.json') == protocol['transfer_inputs_sha256']
    original_path = BASE/'ball-rebuild-20261026-v1/verifier-v4/training-samples.json'
    old = read(original_path)
    warm = BASE/'ball-rebuild-20261026-v1/verifier-v4/best.pt'
    assert sha(warm) == protocol['warmstart_sha256']
    cache = {}
    def crop(path, point):
        path = str(path)
        if path not in cache: cache[path] = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        return patch_inputs(cache[path], point).astype(np.float16)
    excluded = []
    def allowed(row, split):
        path = Path(row['image'])
        if SOURCE in path.parents and int(path.stem)-1 >= 2694:
            excluded.append({**row, 'split': split}); return False
        return True
    old = {split: [r for r in rows if allowed(r, split)] for split, rows in old.items()}
    train_rows = [{**r, 'source': 'original'} for r in old['train']]
    train_rows += [{'image': str(SOURCE/f'{r["frame"]+1:06d}.jpg'), 'point': r['pixel'],
                    'label': r['label'], 'source': 'teacher'} for r in teacher if r['split'] == 'train']
    print({'phase': 'crop_preparation', 'training': len(train_rows),
           'old_validation': len(old['validation']), 'excluded': len(excluded)}, flush=True)
    X = torch.from_numpy(np.stack([crop(r['image'], r['point']) for r in train_rows]))
    Y = torch.tensor([r['label'] for r in train_rows], dtype=torch.float32)
    validation = torch.from_numpy(np.stack([crop(r['image'], r['point']) for r in old['validation']]))
    teacher_x = []
    for row in teacher:
        path = SOURCE/f'{row["frame"]+1:06d}.jpg'; assert sha(path) == row['source_sha256']
        teacher_x.append(crop(path, row['pixel']))
    teacher_x = torch.from_numpy(np.stack(teacher_x))
    transfer_x, transfer_indices = [], []
    for i, row in enumerate(transfer):
        if row['pixel'] is None: continue
        folder = BASE/'new-games-20261025-v1/frames'/row['clip'] if row['clip'].endswith('0600') else OUT/'extra-frames'/row['clip']
        path = folder/f'{row["frame"]+1:06d}.jpg'; assert sha(path) == row['source_sha256']
        transfer_x.append(crop(path, row['pixel'])); transfer_indices.append(i)
    transfer_x = torch.from_numpy(np.stack(transfer_x)); cache.clear()
    pools = {(source, label): [i for i, r in enumerate(train_rows) if r['source'] == source and r['label'] == label]
             for source in ['original', 'teacher'] for label in [0, 1]}
    old_labels = [r['label'] for r in old['validation']]
    dev_indices = [i for i, r in enumerate(teacher) if r['split'] == 'development']
    test_indices = [i for i, r in enumerate(teacher) if r['split'] == 'test']
    dest = OUT/'distillation-replay'; dest.mkdir(exist_ok=True)
    results = {}
    def predict(model, data):
        model.eval(); scores = []
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            for batch in data.split(64): scores.extend(model(batch.cuda().float()).float().sigmoid().cpu().tolist())
        return scores
    def select(old_scores, scores):
        options = []
        for threshold in protocol['thresholds']:
            a = classification(old_scores, old_labels, threshold)
            b = classification([scores[i] for i in dev_indices], [teacher[i]['label'] for i in dev_indices], threshold)
            options.append({'threshold': threshold, 'mean_f1': (a['f1']+b['f1'])/2,
                            'fp': a['fp']+b['fp'], 'original': a, 'teacher': b})
        return max(options, key=lambda r: (r['mean_f1'], -r['fp'], r['threshold']))
    for variant in protocol['variants']:
        torch.manual_seed(protocol['seed']); rng = np.random.default_rng(protocol['seed'])
        model = SpatialBallVerifier().cuda()
        model.load_state_dict(torch.load(warm, map_location='cpu', weights_only=False)['state_dict'])
        if variant == 'head-replay':
            for p in model.encoder.parameters(): p.requires_grad_(False)
        groups = [{'params': model.head.parameters(), 'lr': protocol['head_learning_rate']}]
        if variant == 'encoder-replay': groups.append({'params': model.encoder.parameters(), 'lr': protocol['encoder_learning_rate']})
        optimizer = torch.optim.AdamW(groups, weight_decay=protocol['weight_decay'])
        initial = select(predict(model, validation), predict(model, teacher_x))
        best = {'epoch': 0, 'selection': initial, 'state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        history = []
        for epoch in range(1, protocol['epochs']+1):
            model.train()
            for layer in model.modules():
                if isinstance(layer, nn.modules.batchnorm._BatchNorm): layer.eval()
            losses = []
            for _ in range(protocol['steps_per_epoch']):
                ids = np.concatenate([rng.choice(pools[('original', 1)], 24), rng.choice(pools[('original', 0)], 24),
                                      rng.choice(pools[('teacher', 1)], 8), rng.choice(pools[('teacher', 0)], 8)])
                batch = X[ids].cuda().float(); labels = Y[ids].cuda()
                flipped = torch.rand(len(batch), device='cuda') < .5
                batch[flipped] = batch[flipped].flip(-1)
                batch = batch*torch.empty(len(batch), 1, 1, 1, device='cuda').uniform_(.9, 1.1)
                batch += torch.empty(len(batch), 1, 1, 1, device='cuda').uniform_(-.1, .1)
                optimizer.zero_grad()
                with torch.autocast('cuda', dtype=torch.bfloat16): loss = nn.functional.binary_cross_entropy_with_logits(model(batch), labels)
                if not torch.isfinite(loss): raise FloatingPointError('Non-finite loss')
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
                losses.append(float(loss.detach()))
            selection = select(predict(model, validation), predict(model, teacher_x))
            history.append({'epoch': epoch, 'loss': float(np.mean(losses)), **selection})
            rank = lambda r: (r['mean_f1'], -r['fp'])
            if rank(selection) > rank(best['selection']):
                best = {'epoch': epoch, 'selection': selection, 'state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            if epoch % 5 == 0: print({'variant': variant, 'epoch': epoch, 'selection': selection}, flush=True)
        model.load_state_dict(best['state_dict']); scores = predict(model, teacher_x)
        threshold = best['selection']['threshold']
        output_scores = predict(model, transfer_x)
        index_scores = dict(zip(transfer_indices, output_scores))
        weight_path = dest/f'{variant}.pt'; torch.save(best, weight_path)
        results[variant] = {'selected_epoch': best['epoch'], 'threshold': threshold,
            'development': best['selection'], 'test': classification([scores[i] for i in test_indices], [teacher[i]['label'] for i in test_indices], threshold),
            'weights_sha256': sha(weight_path), 'history': history,
            'rows': [{**r, 'score': s, 'accepted': s >= threshold} for r, s in zip(teacher, scores)],
            'transfer': [{**r, 'verifier_score': index_scores.get(i), 'accepted': index_scores.get(i, -1) >= threshold} for i, r in enumerate(transfer)],
            'original_validation': [{**r, 'score': s} for r, s in zip(old['validation'], predict(model, validation))]}
        print({'variant': variant, 'test': results[variant]['test'], 'epoch': best['epoch']}, flush=True)
        del model; torch.cuda.empty_cache(); commit()
    report = {'protocol': protocol, 'original_samples_sha256': sha(original_path),
              'training_counts': {f'{source}_{label}': len(ids) for (source, label), ids in pools.items()},
              'excluded_original_samples': excluded, 'variants': results, 'seconds': time.monotonic()-started,
              'limits': protocol['scope']}
    write(dest/'report.json', report); commit(); return report
