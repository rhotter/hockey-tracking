"""Paired, seed-excluded localization checks on reused NBHL references."""
from pathlib import Path
import json, sys, importlib.util
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('sequence_evaluate', ROOT/'experiments/ball-sequence-20261026/evaluate.py')
sequence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sequence)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, separators=(',', ':'))+'\n')


def predictions(data):
    """Nearest seed wins before examining visibility or reference labels."""
    result = {}
    for track in data['tracks']:
        for row in track['rows']:
            f = row['frame']
            distance = abs(f-track['seed_frame'])
            if f not in result or distance < result[f]['seed_distance']:
                result[f] = {**row, 'seed_frame': track['seed_frame'], 'seed_distance': distance}
    return result


def accepted(row, mode, protocol):
    if not (row['visible'] and row['inside_crop'] and row['inside_image']):
        return False
    if mode in ['shape', 'filtered'] and not row.get('shape_valid', True):
        return False
    return mode != 'filtered' or row['appearance_score'] >= protocol['appearance_threshold']


def summarize(rows):
    errors = [r['error_px'] for r in rows if r['error_px'] is not None]
    good = sum(e <= 25 for e in errors)
    bad = sum(e > 25 for e in errors)
    return {'reviewed_positions': len(rows), 'answered': len(errors),
            'within_12px': sum(e <= 12 for e in errors), 'within_25px': good,
            'outside_25px': bad, 'median_error_px': float(np.median(errors)) if errors else None,
            'selection_utility': good-3*bad, 'details': rows}


def score(data, protocol, labels=None):
    pred = predictions(data)
    seeds = [t['seed_frame'] for t in data['tracks']]
    labels = sequence.reference() if labels is None else labels
    labels = [r for r in labels if r['frame'] in pred and all(abs(r['frame']-s)>3 for s in seeds)]
    modes = ['raw', 'shape', 'filtered'] if data['model'].startswith('sam') else ['raw', 'filtered']
    out = {}
    for mode in modes:
        rows = []
        for label in labels:
            r = pred[label['frame']]
            valid = accepted(r, mode, protocol)
            error = float(np.linalg.norm(np.array(r['pixel'])-label['pixel'])) if valid else None
            rows.append({**label, 'prediction': r['pixel'] if valid else None, 'error_px': error,
                         'seed_frame': r['seed_frame'], 'seed_distance': r['seed_distance']})
        out[mode] = {'all': summarize(rows)}
        for split in ['development', 'validation']:
            out[mode][split] = summarize([r for r in rows if r['seed_frame'] in protocol[split+'_seeds']])
    return {'model': data['model'], 'seed_count': len(seeds), 'eligible_frames': len(pred),
            'proposals': {m: sum(accepted(r, m, protocol) for r in pred.values()) for m in modes}, 'scores': out}


def selection(pilots):
    # Only frozen filtered development scores can select the SAM configuration.
    def key(name):
        s = pilots[name]['scores']['filtered']['development']
        return (s['selection_utility'], s['within_12px'], -s['outside_25px'])
    return max(sorted(n for n in pilots if n.startswith('sam')), key=key)


def derive_pilot(folder):
    protocol = read(HERE/'protocol.json')
    items = {name: score(read(folder/'pilot'/(name+'.json')), protocol) for name in protocol['models']}
    items['tapnextpp512'] = score(read(folder/'tapnextpp512-pilot-verified.json'), protocol)
    labels = [[r['frame'] for r in value['scores']['raw']['all']['details']] for value in items.values()]
    assert all(frames == labels[0] for frames in labels), 'Pilot reference frames must match exactly'
    result = {'protocol': protocol, 'pilots': items, 'selected_variant': selection(items),
              'limits': 'Reused approximate assistant annotations on one previously examined five-minute NBHL recording. Validation is a seed-window split, not an independent game or human benchmark.'}
    return result


def pilot(folder):
    result = derive_pilot(folder)
    items = result['pilots']
    write(folder/'pilot-evaluation.json', result)
    for name, item in items.items():
        for mode, splits in item['scores'].items():
            print(name, mode, {split: {k:v for k,v in value.items() if k!='details'} for split,value in splits.items()})
    print('SELECTED', result['selected_variant'])
    return result


if __name__ == '__main__':
    pilot(Path(sys.argv[1]))
