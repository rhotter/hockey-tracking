"""Identity-supervised contrastive adaptation using reviewed NBHL jersey evidence."""
import json
from pathlib import Path
from nbhl_reid_learning import sha, dump

DINO_MODEL = 'facebook/dinov2-base'
DINO_REVISION = 'f9e44c814b77203eaa57a6bdbbd535f21ede1415'


def load_model(base, variant, checkpoint=None, device='cuda'):
    import torch
    from nbhl_reid_learning import load_model as load_osnet
    from torch.nn import functional as F
    assert variant in ['osnet_supcon', 'osnet_parts', 'dino_head', 'dino_last',
                       'osnet_parts_late', 'osnet_parts_head', 'osnet_parts_regularized', 'osnet_parts_expanded',
                       'osnet_parts_hardteam', 'osnet_parts_temporal', 'osnet_parts_temporal_s73', 'osnet_parts_late_s73']
    if variant.startswith('osnet'):
        original, provenance = load_osnet(base, device=device)
        provenance = {k: v for k, v in provenance.items() if k in [
            'model', 'pretraining', 'code_url', 'code_revision', 'code_sha256',
            'weights_repository', 'weights_revision', 'weights_file', 'weights_sha256']}
        backbone = original.backbone
        dim = 512
    else:
        from transformers import AutoModel
        backbone = AutoModel.from_pretrained(DINO_MODEL, revision=DINO_REVISION, local_files_only=True)
        provenance = {'model': DINO_MODEL, 'revision': DINO_REVISION, 'pretraining': 'generic self-supervised images'}
        dim = 768
    class IdentityModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.projection = torch.nn.Linear(dim, dim, bias=False)
            with torch.no_grad():
                self.projection.weight.copy_(torch.eye(dim))
        def forward(self, x):
            if variant.startswith('osnet_parts'):
                maps = self.backbone.featuremaps(x)
                parts = F.adaptive_avg_pool2d(maps, (3, 1)).squeeze(-1).transpose(1, 2)
                z = self.backbone.fc(parts.reshape(-1, 512))
                z = F.normalize(self.projection(z), dim=1)
                return z.reshape(len(x), -1)/(3.**.5)
            if variant.startswith('osnet'):
                z = self.backbone(x)
            else:
                z = self.backbone(pixel_values=x).last_hidden_state[:, 0]
            return F.normalize(self.projection(z), dim=1)
    model = IdentityModel().to(device).eval()
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    return model, provenance


def preprocess(variant, augment=False):
    from torchvision import transforms as T
    size = (336, 168) if variant.startswith('dino') else (256, 128)
    operations = [T.Resize(size)]
    if augment:
        operations += [T.RandomApply([T.ColorJitter(.08, .08, .05, .01)], p=.5),
                       T.RandomAffine(degrees=2, translate=(.03, .03), scale=(.95, 1.05))]
    operations += [T.ToTensor(), T.Normalize([.485, .456, .406], [.229, .224, .225])]
    # No horizontal flip: it would mirror jersey digits. No aggressive erasing.
    return T.Compose(operations)


def assemble_dataset(base, out):
    from collections import defaultdict
    (out/'contrastive').mkdir(parents=True, exist_ok=True)
    old_path = base/'reid-finetune-v1/dataset.json'
    old = json.loads(old_path.read_text())
    images = dict(old['images'])
    groups = defaultdict(list)
    evidence_hashes = {}
    review_hashes = {}
    labels = []
    for start in [600, 1200, 1800]:
        suffix = '' if start == 600 else '-'+str(start)
        review_path = Path('/work')/('tracking-four-hour-identity-review'+suffix+'-v1.json')
        review = json.loads(review_path.read_text())
        assert review['status'] == 'sealed_for_training'
        folder = out/('identity-data' if start == 600 else 'identity-data-'+str(start))
        ep = folder/'evidence.json'
        evidence = json.loads(ep.read_text())
        assert review['evidence_sha256'] == sha(ep)
        evidence_hashes[str(start)] = sha(ep)
        review_hashes[str(start)] = sha(review_path)
        lookup = {r['id']: r for r in evidence['items']}
        for key, label in review['labels'].items():
            row = lookup[key]
            assert row['source_id'] in ['Z-tFHtxODsE', 'OLpK8enSjJs']
            identity = f'{row["source_id"]}/{label["team"]}/{label["number"]}'
            images[key] = {'path': row['image'], 'sha256': row['sha256'], 'source_id': row['source_id'],
                           'split': 'train', 'origin': 'assistant-reviewed-complete-number',
                           'source_time_s': row['source_time_s'], 'identity': identity}
            groups[identity].append(key)
            labels.append({'id': key, 'identity': identity, 'source_id': row['source_id'], 'time_s': row['source_time_s']})
    # Duplicate jersey identities simultaneously visible imply a labeling conflict.
    for identity, keys in groups.items():
        times = [images[k]['source_time_s'] for k in keys]
        assert len(set(times)) == len(times), ('co-occurring identity conflict', identity)
    groups = {k: v for k, v in groups.items() if len(v) >= 2}
    train_pairs = [r for r in old['pairs'] if r['split'] == 'train' and r['label_type'] == 'reviewed']
    val_pairs = [r for r in old['pairs'] if r['split'] == 'validation']
    assert {r['source_id'] for r in train_pairs} <= {'Z-tFHtxODsE', 'OLpK8enSjJs'}
    assert {r['source_id'] for r in val_pairs} == {'plDp2kPtk_g'}
    result = {'images': images, 'groups': groups, 'train_pairs': train_pairs, 'validation_pairs': val_pairs,
              'labels': labels, 'old_dataset_sha256': sha(old_path), 'evidence_sha256': evidence_hashes,
              'review_sha256': review_hashes, 'protocol_sha256': sha('/work/tracking-four-hour-contrastive-protocol-v1.json')}
    dump(out/'contrastive/dataset.json', result)
    return result


def assemble_dataset_v2(base, out):
    from collections import defaultdict
    data = json.loads((out/'contrastive/dataset.json').read_text())
    review_path = Path('/work/tracking-four-hour-identity-review-extra-v2.json')
    review = json.loads(review_path.read_text())
    assert review['status'] == 'sealed_for_training'
    ep = out/'identity-extra-training-v2/evidence.json'
    assert sha(ep) == review['evidence_sha256']
    evidence = json.loads(ep.read_text())
    lookup = {r['id']: r for r in evidence['items']}
    groups = defaultdict(list, data['groups'])
    for key, label in review['labels'].items():
        row = lookup[key]
        assert row['source_id'] in ['WWX74oJC9co', 'xbLtZLIiiyA']
        identity = f'{row["source_id"]}/{label["team"]}/{label["number"]}'
        data['images'][key] = {'path': row['image'], 'sha256': row['sha256'], 'source_id': row['source_id'],
                               'split': 'train', 'origin': 'assistant-reviewed-complete-number',
                               'source_time_s': row['source_time_s'], 'identity': identity}
        groups[identity].append(key)
        data['labels'].append({'id': key, 'identity': identity, 'source_id': row['source_id'], 'time_s': row['source_time_s']})
    for identity, keys in groups.items():
        times = [data['images'][key]['source_time_s'] for key in keys]
        assert len(times) == len(set(times)), identity
    data['groups'] = {k: v for k, v in groups.items() if len(v)>=2}
    # Endpoint-only long-pair equipment judgments are excluded from v2 training.
    data['train_pairs'] = [r for r in data['train_pairs'] if r['id'].startswith('pair_')]
    cp = Path('/work/tracking-four-hour-reference-corrections-v1.json')
    corrections = {r['pair_id']: r for r in json.loads(cp.read_text())['corrections']}
    validation = []
    for row in data['validation_pairs']:
        correction = corrections.get(row['id'])
        if correction:
            if correction['truth'] == 'uncertain':
                continue
            row['same'] = correction['truth'] == 'same'
        validation.append(row)
    data.update(validation_pairs=validation, parent_dataset_sha256=sha(out/'contrastive/dataset.json'),
                additional_review_sha256=sha(review_path), additional_evidence_sha256=sha(ep),
                corrections_sha256=sha(cp), protocol_sha256=sha('/work/tracking-four-hour-contrastive-protocol-v2.json'))
    dump(out/'contrastive/dataset-v2.json', data)
    return data


def train(base, out, variant, volume):
    import random, time, io
    import numpy as np
    import torch
    from torch.nn import functional as F
    from PIL import Image
    from nbhl_reid_learning import validation
    from concurrent.futures import ThreadPoolExecutor
    torch.set_num_threads(4)
    seed = 73 if variant.endswith('_s73') else 41
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    weak_temporal = variant in ['osnet_parts_temporal', 'osnet_parts_temporal_s73']
    hard_team = weak_temporal or variant == 'osnet_parts_hardteam'
    v2 = variant in ['osnet_parts_late', 'osnet_parts_head', 'osnet_parts_regularized', 'osnet_parts_expanded',
                     'osnet_parts_hardteam', 'osnet_parts_temporal', 'osnet_parts_temporal_s73', 'osnet_parts_late_s73']
    preservation_weight = 2. if v2 and variant != 'osnet_parts_expanded' else .25
    dataset_path = out/'contrastive'/('dataset-v3.json' if weak_temporal else 'dataset-v2.json' if v2 else 'dataset.json')
    data = json.loads(dataset_path.read_text())
    folder = out/'contrastive'/variant
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'completed.json').exists():
        return json.loads((folder/'completed.json').read_text())
    weak_groups = data.get('weak_groups', {}) if weak_temporal else {}
    keys = sorted(set(sum(data['groups'].values(), []) + sum(weak_groups.values(), [])) | {r[k] for r in data['train_pairs']+data['validation_pairs'] for k in ['a', 'b']})
    pictures = {}
    def read(key):
        row = data['images'][key]
        payload = Path(row['path']).read_bytes()
        import hashlib
        assert hashlib.sha256(payload).hexdigest() == row['sha256']
        return key, Image.open(io.BytesIO(payload)).convert('RGB')
    with ThreadPoolExecutor(max_workers=24) as pool:
        pictures.update(pool.map(read, keys))
    model, provenance = load_model(base, variant)
    plain, aug = preprocess(variant), preprocess(variant, True)
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith('projection') or (
            variant.startswith('osnet') and name.startswith('backbone.') and not name.startswith('backbone.classifier')) or (
            variant == 'dino_last' and name.startswith(('backbone.encoder.layer.10.', 'backbone.encoder.layer.11.', 'backbone.layernorm.')))
        if variant == 'osnet_parts_head':
            param.requires_grad = name.startswith('projection')
        elif variant in ['osnet_parts_late', 'osnet_parts_hardteam', 'osnet_parts_temporal', 'osnet_parts_temporal_s73', 'osnet_parts_late_s73']:
            param.requires_grad = name.startswith(('projection', 'backbone.conv4', 'backbone.conv5', 'backbone.fc'))
    params = [p for p in model.parameters() if p.requires_grad]
    learning_rate = 2e-4 if variant in ['dino_head', 'osnet_parts_head'] else 2e-6 if variant == 'osnet_parts_regularized' else 1e-5
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=1e-4)
    # Fixed generic clean-crop targets provide a compact preservation penalty.
    teacher_features = {}
    with torch.inference_mode():
        for first in range(0, len(keys), 32):
            group = keys[first:first+32]
            z = model(torch.stack([plain(pictures[k]) for k in group]).cuda()).float().cpu()
            teacher_features.update(zip(group, z))
    baseline = validation(model, pictures, data['validation_pairs'], plain)
    best = (baseline['tp'], baseline['average_precision'])
    best_step, best_val = 0, baseline
    torch.save(model.state_dict(), folder/'best.pt')
    source_groups = {}
    for identity in data['groups']:
        bucket = '/'.join(identity.split('/')[:2]) if hard_team else identity.split('/')[0]
        source_groups.setdefault(bucket, []).append(identity)
    source_groups = {k: v for k, v in source_groups.items() if len(v) >= 2}
    assert source_groups
    pools = {same: [r for r in data['train_pairs'] if r['same'] == same] for same in [False, True]}
    history = []
    started = time.monotonic()
    for step in range(1, 1801):
        source = random.choice(sorted(source_groups))
        identities = random.sample(source_groups[source], min(8, len(source_groups[source])))
        crop_keys, labels = [], []
        for label, identity in enumerate(identities):
            options = data['groups'][identity]
            # Prefer distinct temporal samples; sample with replacement only for tiny classes.
            use_weak = bool(weak_groups.get(identity))
            reviewed_count = 3 if use_weak else 4
            sampled = random.sample(options, min(reviewed_count, len(options)))
            sampled += random.choices(options, k=reviewed_count-len(sampled))
            if use_weak:
                sampled.append(random.choice(weak_groups[identity]))
            crop_keys.extend(sampled)
            labels.extend([label]*4)
        class_count = len(crop_keys)
        pairs = [random.choice(pools[i % 2 == 0]) for i in range(16)]
        crop_keys += [r['a'] for r in pairs] + [r['b'] for r in pairs]
        batch = torch.stack([aug(pictures[k]) for k in crop_keys]).cuda()
        z = model(batch)
        zi = z[:class_count]
        similarity = zi @ zi.T / .1
        eye = torch.eye(class_count, dtype=torch.bool, device='cuda')
        target = torch.tensor(labels, device='cuda')
        positive = (target[:, None] == target[None, :]) & ~eye
        logits = similarity.masked_fill(eye, -1e4)
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        contrastive = -(log_prob*positive).sum(1).div(positive.sum(1).clamp_min(1)).mean()
        a, b = z[class_count:].chunk(2)
        cosine = (a*b).sum(1)
        truth = torch.tensor([r['same'] for r in pairs], device='cuda')
        pair_loss = torch.where(truth, (1-cosine).square(), F.relu(cosine-.5).square()).mean()
        teacher = torch.stack([teacher_features[k] for k in crop_keys]).cuda()
        preservation = (1-(z*teacher).sum(1)).mean()
        loss = contrastive + pair_loss + preservation_weight*preservation
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.)
        optimizer.step()
        if step % 100 == 0:
            score = validation(model, pictures, data['validation_pairs'], plain)
            objective = (score['tp'], score['average_precision'])
            if objective > best:
                best, best_step, best_val = objective, step, score
                torch.save(model.state_dict(), folder/'best.pt')
            entry = {'step': step, 'loss': float(loss.detach()), 'contrastive': float(contrastive.detach()),
                     'pair_loss': float(pair_loss.detach()), 'preservation': float(preservation.detach()),
                     'validation': {k: v for k, v in score.items() if k != 'pairs'}, 'seconds': time.monotonic()-started}
            history.append(entry)
            dump(folder/'progress.json', {'variant': variant, 'best_step': best_step, 'history': history})
            volume.commit()
            print(json.dumps({'variant': variant, **entry}), flush=True)
    result = {'variant': variant, 'selected_step': best_step, 'training_steps': 1800,
              'reviewed_identity_crops': sum(map(len, data['groups'].values())), 'identity_classes': len(data['groups']),
              'reviewed_pair_count': len(data['train_pairs']), 'validation_pair_count': len(data['validation_pairs']),
              'baseline': baseline, 'selected': best_val, 'history': history, 'seconds': time.monotonic()-started,
              'weights_sha256': sha(folder/'best.pt'), 'dataset_sha256': sha(dataset_path), 'code_sha256': sha(__file__),
              'trainable_parameters': sum(p.numel() for p in params), 'base_model': provenance,
              'loss': f'Supervised contrastive temperature .1 + reviewed pair loss + {preservation_weight} generic clean-crop preservation.',
              'optimizer': {'lr': learning_rate, 'weight_decay': 1e-4, 'seed': seed},
              'sampling': 'same-team hard negatives' if hard_team else 'same-recording identities',
              'weak_temporal_crops': sum(map(len, weak_groups.values())),
              'weak_policy': 'At most one weak interior crop alongside three reviewed crops per identity in each contrastive batch.' if weak_temporal else None,
              'selection': f'Same-player recall at zero observed false merges, then average precision on {len(data["validation_pairs"])} Massachusetts pairs; step 0 eligible.',
              'scope': 'Assistant-reviewed NBHL training identities. Correlated crops and small identity set; no independent-human test claim.'}
    dump(folder/'completed.json', result)
    volume.commit()
    return result
