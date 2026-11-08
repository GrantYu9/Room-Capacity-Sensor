"""Frozen-label synthetic challenge of the actual ImageProcessor inference path.

No training or threshold tuning. Count MAE measures the quantity the application
uses; exact counts and signed error expose under/overcounting. This is not mAP,
precision, recall, or an estimate of accuracy on real cameras.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import html
from importlib.metadata import distributions
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.process_data import ImageProcessor


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(rows):
    errors = np.array([row['predicted'] - row['expected'] for row in rows], dtype=float)
    absolute = np.abs(errors)
    # Resample scenes within one resolution, never treat the paired resolutions
    # as independent samples. This interval describes only this crafted set.
    rng = np.random.default_rng(42)
    means = absolute[rng.integers(0, len(rows), size=(20000, len(rows)))].mean(axis=1)
    return {'scenes': len(rows), 'exact_counts': int(np.sum(errors == 0)),
            'mae_people': float(absolute.mean()), 'signed_bias_people': float(errors.mean()),
            'maximum_absolute_error': float(absolute.max()),
            'mae_scene_bootstrap_95': np.percentile(means, [2.5, 97.5]).tolist(),
            'always_zero_mae_people': float(np.mean([row['expected'] for row in rows]))}


def write_reports(output, data, manifest):
    summary = data['summary']
    lines = ['# Synthetic person-counting challenge', '',
             'Seven generated scenes, independently visually counted before inference. '
             'One fixed model and confidence threshold; no training or tuning.', '',
             '| Scene | Visible people | Native prediction | 640×480 prediction |',
             '|---|---:|---:|---:|']
    pairs = []
    for scene in manifest['scenes']:
        rows = [next(row for row in data['results'] if row['scene'] == scene['id'] and row['mode'] == mode)
                for mode in ['native', 'camera-640x480']]
        pairs.append((scene, rows))
        lines.append(f"| {scene['id']} | {scene['people_count']} | {rows[0]['predicted']} | {rows[1]['predicted']} |")
    lines += ['', '## Count error', '']
    for mode, result in summary.items():
        low, high = result['mae_scene_bootstrap_95']
        lines.append(f"- **{mode}:** {result['exact_counts']}/{result['scenes']} exact counts; "
                     f"MAE {result['mae_people']:.3f} people (scene bootstrap interval {low:.3f}–{high:.3f}); "
                     f"signed bias {result['signed_bias_people']:+.3f} people. "
                     f"Always-zero baseline MAE: {result['always_zero_mae_people']:.3f}.")
    lines += ['', '## Production window aggregation sanity check', '',
              'Twenty identical still-image inputs per window, not real video and not independent samples:']
    for window in data['repeated_still_windows']:
        lines.append(f"- {window['scene']}: expected {window['expected']} occupants; "
                     f"mean count over {window['image_count']} identical frames = {window['mean_count']:.1f}.")
    lines += ['', '## Interpretation and measurement limits', '',
              '- All seven scenes are included. This deliberately small, synthetic challenge is not a representative sample of campus conditions. '
              'Bootstrap intervals describe resampling these scenes, not real-world performance.',
              '- Each resolution reuses the same seven scenes and labels. The 14 predictions are not 14 independent scenes. '
              'The 640×480 variant uses area downsampling to approximate capture resolution; real camera optics, gain, compression and motion remain untested.',
              '- Physical occupants count toward ground truth; printed people, video-call faces, coats and hats do not. '
              'A detector can mistake a depiction for an occupant even with a high confidence score.',
              '- Mean detection confidence is a detector score, not measured accuracy. Count matches can also hide a simultaneous missed person and false positive; '
              'inspect annotations rather than interpreting this as detection precision/recall.',
              '- Generated scenes cannot validate sound measurements. The analysis receives an explicit silent audio fixture solely to exercise the production API; '
              'it produces −120 dBFS and establishes no microphone calibration or dBA accuracy.',
              '- No training split is created: these are fixed-checkpoint challenge cases. Training data are not available for a complete contamination audit. '
              'Threshold and weights were fixed before running; any future tuning requires fresh validation scenes.',
              '- Production chain: camera frames → ImageProcessor → person detections → per-frame count → mean window count. '
              'This test replaces camera acquisition with labeled images. It tests no database uploads or live hardware.',
              '- Next validation step: labeled frames from the installed camera across rooms, crowding, lighting and times; '
              'keep scenes/sessions separate when evaluating any later tuning.', '',
              'Full configuration, hashes, exact dependency versions, detections, and timings: [results.json](results.json). '
              'Visual comparison: [report.html](report.html), [contact-sheet.png](contact-sheet.png).', '']
    (output / 'report.md').write_text('\n'.join(lines))

    cards = []
    score_label = lambda row: 'n/a' if row['mean_detection_confidence'] is None else f"{row['mean_detection_confidence']:.3f}"
    for scene, rows in pairs:
        pictures = ''.join(f'<figure><img loading="lazy" src="{row["annotated_image"]}" '
                           f'alt="{html.escape(scene["id"])} {row["mode"]} prediction boxes">'
                           f'<figcaption>{row["mode"]}: expected {row["expected"]}, detected {row["predicted"]}; '
                           f'mean detector score {score_label(row)}</figcaption></figure>'
                           for row in rows)
        cards.append(f'<section><h2>{html.escape(scene["id"])} · {scene["people_count"]} physical people</h2>'
                     f'<p>{html.escape(scene["annotation_notes"])}</p><div class="pair">{pictures}</div></section>')
    stats = ' · '.join(f'{mode}: {value["exact_counts"]}/7 exact, MAE {value["mae_people"]:.3f}' for mode, value in summary.items())
    (output / 'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>Synthetic count evaluation</title>'
        '<style>body{font:16px system-ui;max-width:1400px;margin:32px auto;padding:0 20px;color:#203c31;background:#f8f9f4}'
        'h1{font-size:30px}section{padding:24px 0;border-top:1px solid #cbd3c6}.pair{display:grid;grid-template-columns:1fr 1fr;gap:18px}'
        'figure{margin:0}img{display:block;width:100%;height:auto}figcaption{padding:8px 0;font-size:14px}'
        '@media(max-width:750px){.pair{grid-template-columns:1fr}}</style><h1>Synthetic person-counting challenge</h1>'
        f'<p><strong>{html.escape(stats)}</strong></p><p>Seven generated scenes, no tuning. Physical occupants only; '
        'posters and video-call faces are distractors. These results do not establish real-camera accuracy. '
        'Detector scores are not accuracy rates.</p>' + ''.join(cards) + '</html>')

    fig, axes = plt.subplots(4, 2, figsize=(14, 19))
    for ax, (scene, rows) in zip(axes.flat, pairs):
        row = rows[0]
        image = cv2.cvtColor(cv2.imread(str(output / row['annotated_image'])), cv2.COLOR_BGR2RGB)
        ax.imshow(image)
        ax.set_title(f"{scene['id']}: expected {row['expected']}, predicted {row['predicted']}", fontsize=12)
        ax.axis('off')
    axes.flat[-1].axis('off')
    axes.flat[-1].text(0.05, 0.7, 'Native-size inputs · confidence 0.25\n\n'
                      f"Exact counts: {summary['native']['exact_counts']}/7\n"
                      f"MAE: {summary['native']['mae_people']:.3f} people\n\n"
                      'Synthetic challenge only.\nNot a real-camera accuracy estimate.\n'
                      'Poster/screen images are not occupants.', fontsize=15, va='top')
    fig.tight_layout()
    fig.savefig(output / 'contact-sheet.png', dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'data/testing/synthetic-2026-11-07/labels.json')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    output = (args.output or manifest_path.parent / 'results' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')).resolve()
    output.mkdir(parents=True, exist_ok=False)
    # No fitting: all reviewed labels and image hashes are frozen before loading
    # weights. Resolution variants are the same group, not new independent data.
    for scene in manifest['scenes']:
        if sha256(manifest_path.parent / scene['image']) != scene['sha256']:
            raise ValueError(f"Image changed after annotation: {scene['id']}")
    (output / 'labels-frozen.json').write_text(json.dumps(manifest, indent=2) + '\n')
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(2)
    weights = ROOT / 'data/external/yolo26n.pt'
    processor = ImageProcessor(weights, confidence=manifest['confidence'])
    capture = []
    original_predict = processor._model.predict
    def record_predict(*args, **kwargs):
        result = original_predict(*args, **kwargs)
        capture[:] = result
        return result
    processor._model.predict = record_predict
    silence = np.zeros(8000, dtype=np.float32)
    processor.process_data(silence, np.zeros((480, 640, 3), dtype=np.uint8), 'warmup', sample_rate=8000)
    rows = []
    for mode in ['native', 'camera-640x480']:
        (output / 'annotations' / mode).mkdir(parents=True)
        for scene in manifest['scenes']:
            source = manifest_path.parent / scene['image']
            frame = cv2.imread(str(source))
            if frame is None:
                raise ValueError(f"Could not load {source}")
            image = source if mode == 'native' else cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
            start = time.perf_counter()
            analyzed = processor.process_data(silence, [image], scene['id'], sample_rate=8000)
            elapsed = time.perf_counter() - start
            prediction = int(analyzed['per_image_counts'][0])
            result = capture[0]
            annotated = Path('annotations') / mode / f"{scene['id']}.jpg"
            if not cv2.imwrite(str(output / annotated), result.plot()):
                raise OSError('Could not save prediction annotation')
            row = {'scene': scene['id'], 'mode': mode, 'expected': scene['people_count'],
                   'predicted': prediction, 'signed_error': prediction - scene['people_count'],
                   'mean_detection_confidence': analyzed['average_confidence'], 'seconds': elapsed,
                   'input_width': int(result.orig_shape[1]), 'input_height': int(result.orig_shape[0]),
                   'boxes_xyxy': result.boxes.xyxy.cpu().tolist(),
                   'box_confidences': result.boxes.conf.cpu().tolist(),
                   'annotated_image': str(annotated)}
            rows.append(row)
            print(json.dumps({key: row[key] for key in ['scene', 'mode', 'expected', 'predicted', 'mean_detection_confidence']}), flush=True)
    windows = []
    for scene in manifest['scenes']:
        if scene['id'] not in ['empty', 'crowded', 'distractors']:
            continue
        image = cv2.imread(str(manifest_path.parent / scene['image']))
        frame = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)
        analyzed = processor.process_data(silence, [frame] * 20, scene['id'], sample_rate=8000)
        window = {'scene': scene['id'], 'expected': scene['people_count'], 'image_count': analyzed['image_count'],
                  'mean_count': analyzed['count'], 'per_image_counts': analyzed['per_image_counts'],
                  'note': 'Identical still repeated to verify window aggregation; not a video or independent observations.'}
        windows.append(window)
        print(json.dumps({'repeated_still_window': scene['id'], 'mean_count': analyzed['count'], 'images': analyzed['image_count']}), flush=True)
    git = lambda *commands: subprocess.check_output(['git', '-C', str(ROOT), *commands], text=True).strip()
    predictor = processor._model.predictor
    data = {'created_at': datetime.now(timezone.utc).isoformat(), 'upstream_head': git('rev-parse', 'HEAD'),
            'working_tree_changes': git('status', '--short').splitlines(), 'model_sha256': sha256(weights),
            'labels_sha256': sha256(manifest_path), 'seed': 42, 'threads': 2, 'device': str(predictor.model.device),
            'settings': {key: getattr(predictor.args, key, None) for key in ['conf', 'iou', 'imgsz', 'max_det', 'end2end']},
            'backend_end2end': getattr(predictor.model, 'end2end', None),
            'source_sha256': {name: sha256(ROOT / name) for name in ['src/process_data.py', 'src/get_data.py', 'interface/main.py', 'testing/evaluate_synthetic.py']},
            'dependencies': {dist.metadata['Name']: dist.version for dist in distributions()},
            'results': rows, 'repeated_still_windows': windows,
            'summary': {mode: metrics([row for row in rows if row['mode'] == mode]) for mode in ['native', 'camera-640x480']}}
    (output / 'results.json').write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    write_reports(output, data, manifest)
    print(json.dumps({'report_directory': str(output), 'summary': data['summary']}, indent=2))


if __name__ == '__main__':
    main()
