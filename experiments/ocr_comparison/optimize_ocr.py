"""Optimize the OCR preprocessing+param config against ground truth.

Search (Optuna TPE) over src/ocr_pipeline transforms to MAXIMISE exact numeric
read accuracy. Ground truth comes from two sources, kept separate to expose
overfitting:
  * SYNTHETIC train  (labelkit.sample, seed A) -- the optimization objective;
  * SYNTHETIC val    (seed B)                  -- generalisation within synthetic;
  * REAL holdout     (data/holdout/truth.csv)  -- the hand-labelled transfer test.

A config that wins on synthetic but not on the real holdout is overfit to the
generator; we report all three so the choice is evidence-based, not assumed.
The winner (if it beats the production baseline on the real holdout) is saved to
data/best_ocr_cfg.json.

Run: .venv/bin/python experiments/ocr_comparison/optimize_ocr.py --trials 60 --train 240 --val 400
Requires local-only ground truth (data/holdout/truth.csv + the synthetic
labelkit generator); not redistributed. See experiments/README.md.
"""
import argparse, csv, json, os, random, sys
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import labelkit as lk
from ocr_pipeline import predict_value, parse_value, BASELINE
import optuna
from tqdm import tqdm

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")

# --- parallel evaluation across CPU cores (tesseract is CPU-only; GPU can't help
# it, but the per-crop evals are embarrassingly parallel). Each worker holds the
# datasets so only the small cfg dict is shipped per task. ---
_SETS = {}


def _init_worker(sets):
    os.environ["OMP_THREAD_LIMIT"] = "1"   # 1 thread/tesseract; we parallelise outside
    _SETS.update(sets)


def _eval_one(task):
    name, idx, cfg = task
    crop, truth = _SETS[name][idx]
    try:
        pred = predict_value(crop, cfg)
    except Exception:
        pred = None
    return int(pred is not None and abs(pred - truth) < 0.5)


def make_synth(n, seed):
    rng = random.Random(seed)
    data = []
    for i in range(n):
        axis = "y" if i % 2 == 0 else "x"
        crop, text = lk.sample(axis, rng)
        v = parse_value(text)
        if v is not None:
            data.append((crop, v))
    return data


def load_holdout():
    import cv2
    rows = list(csv.DictReader(open(os.path.join(ROOT, "data", "holdout", "truth.csv"))))
    data = []
    for r in rows:
        t = r["truth"].strip()
        if not t or t.upper() == "JUNK":
            continue
        v = parse_value(t)
        cf = os.path.join(ROOT, r["crop_file"])
        if v is not None and os.path.exists(cf):
            img = cv2.imread(cf)
            if img is not None:
                data.append((img, v))
    return data


def accuracy(name, n, cfg, pool):
    """Parallel exact-match accuracy of cfg over dataset `name` (first n items)."""
    if n == 0:
        return float("nan")
    res = pool.map(_eval_one, [(name, i, cfg) for i in range(n)], chunksize=8)
    return sum(res) / len(res)


def suggest(trial):
    th = trial.suggest_categorical("thresh", ["none", "otsu", "adaptive_mean", "adaptive_gauss", "sauvola"])
    return {
        "pad": trial.suggest_int("pad", 0, 20),
        "upscale": trial.suggest_float("upscale", 1.5, 6.0),
        "interp": trial.suggest_categorical("interp", ["cubic", "lanczos", "linear", "area"]),
        "gamma": trial.suggest_float("gamma", 0.5, 2.0),
        "clahe": trial.suggest_categorical("clahe", [False, True]),
        "unsharp": trial.suggest_float("unsharp", 0.0, 2.0),
        "thresh": th,
        "block": trial.suggest_int("block", 11, 51, step=2),
        "sauvola_k": trial.suggest_float("sauvola_k", 0.1, 0.4),
        "morph": trial.suggest_categorical("morph", ["none", "thicken", "thin", "open", "close"]),
        "morph_k": trial.suggest_int("morph_k", 2, 3),
        "psm": trial.suggest_categorical("psm", [6, 7, 8, 11, 13]),
        "oem": trial.suggest_categorical("oem", [1, 3]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--train", type=int, default=240)
    ap.add_argument("--val", type=int, default=400)
    ap.add_argument("--workers", type=int, default=8, help="parallel processes (M4 has 10 cores)")
    args = ap.parse_args()

    print("building datasets...", flush=True)
    train = make_synth(args.train, seed=1)
    val = make_synth(args.val, seed=2)
    holdout = load_holdout()
    nT, nV, nH = len(train), len(val), len(holdout)
    print(f"synth train={nT}  synth val={nV}  real holdout={nH}  | workers={args.workers}", flush=True)

    pool = Pool(args.workers, initializer=_init_worker,
                initargs=({"train": train, "val": val, "holdout": holdout},))

    base_val, base_hold = accuracy("val", nV, BASELINE, pool), accuracy("holdout", nH, BASELINE, pool)
    print(f"\nBASELINE (production)  synth-val={base_val*100:.1f}%  real-holdout={base_hold*100:.1f}%\n",
          flush=True)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=0))

    def objective(trial):
        return accuracy("train", nT, suggest(trial), pool)

    bar = tqdm(total=args.trials, desc="trials", file=sys.stdout, mininterval=0.5)
    def cb(study, trial):
        bar.update(1)
        bar.set_postfix(best_train=f"{study.best_value*100:.1f}%")
    study.optimize(objective, n_trials=args.trials, callbacks=[cb])
    bar.close()

    best = dict(BASELINE); best.update(study.best_params)
    best_train = study.best_value
    best_val, best_hold = accuracy("val", nV, best, pool), accuracy("holdout", nH, best, pool)
    pool.close(); pool.join()

    print("\n================ RESULTS ================")
    print(f"{'config':10s} {'synth-train':>12} {'synth-val':>11} {'real-holdout':>13}")
    print(f"{'baseline':10s} {'-':>12} {base_val*100:10.1f}% {base_hold*100:12.1f}%")
    print(f"{'best':10s} {best_train*100:11.1f}% {best_val*100:10.1f}% {best_hold*100:12.1f}%")
    print("\nbest params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    if best_hold >= base_hold:
        out = os.path.join(ROOT, "data", "best_ocr_cfg.json")
        json.dump(best, open(out, "w"), indent=2)
        print(f"\nwinner beats/ties baseline on real holdout -> saved {os.path.relpath(out, ROOT)}")
    else:
        print("\nbest config does NOT beat baseline on real holdout (overfit to synthetic) "
              "-- NOT saved. Baseline stands.")


if __name__ == "__main__":
    main()
