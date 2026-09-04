#!/bin/bash
# Consolidation Direction A :
#   Phase 1 — GroupNorm, graines 2 et 3 (8 entraînements)
#   Phase 2 — évaluation CIFAR-10-C des graines 2 et 3 (GN)
#   Phase 3 — référence BatchNorm propre pour HSV, 5 graines
#   Phase 4 — ré-évaluation results_v6 (ajoute HSV aux summaries)
# Reprenable : tout ce qui est déjà fait est sauté.
set -u
source ~/miniforge3/etc/profile.d/conda.sh
conda activate thesis
cd ~/decol-gen

echo "[$(date +%H:%M)] env=$CONDA_DEFAULT_ENV — démarrage"

# ── Phase 1 : GN graines 2 et 3 ─────────────────────────────
for s in 2 3; do
  for c in baseline_rgb exp_lab exp_hsv exp_grayscale; do
    d="results_v7/${c}_gn_seed${s}"
    [ -f "$d/summary.json" ] && { echo "[$(date +%H:%M)] SKIP $d"; continue; }
    echo "[$(date +%H:%M)] START $d"
    python -m scripts.train_model --config "configs/${c}_gn.yaml" \
        --seed "$s" --output_dir "$d"
  done
done

# ── Phase 2 : évals GN graines 2 et 3 ───────────────────────
for s in 2 3; do
  ok=1
  for c in baseline_rgb exp_lab exp_hsv exp_grayscale; do
    [ -f "results_v7/${c}_gn_seed${s}/best_model.pth" ] || ok=0
  done
  if [ $ok -eq 0 ]; then echo "[$(date +%H:%M)] eval seed $s ANNULÉE (checkpoints incomplets)"; continue; fi
  [ -f "results_v7/cifar10c_seed${s}/summary.json" ] && { echo "[$(date +%H:%M)] SKIP eval v7 seed $s"; continue; }
  echo "[$(date +%H:%M)] EVAL v7 seed $s"
  python -m scripts.eval_cifar10c --cifar10c_root data/CIFAR-10-C \
      --results_dir results_v7 --seed $s --norm gn --subdir_suffix _gn \
      --output_dir "results_v7/cifar10c_seed${s}"
done

# ── Phase 3 : HSV BatchNorm, constantes corrigées, 5 graines ─
for s in 0 1 2 3 42; do
  d="results_v6/exp_hsv_seed${s}"
  [ -f "$d/summary.json" ] && { echo "[$(date +%H:%M)] SKIP $d"; continue; }
  echo "[$(date +%H:%M)] START $d"
  python -m scripts.train_model --config configs/exp_hsv.yaml \
      --seed "$s" --output_dir "$d"
done

# ── Phase 4 : ré-évaluation v6 (RGB+LAB+HSV) ────────────────
for s in 0 1 2 3 42; do
  f="results_v6/cifar10c_seed${s}/summary.json"
  if [ -f "$f" ] && grep -q '"exp_hsv"' "$f"; then
    echo "[$(date +%H:%M)] SKIP eval v6 seed $s (HSV déjà inclus)"; continue
  fi
  [ -f "results_v6/exp_hsv_seed${s}/best_model.pth" ] || { echo "[$(date +%H:%M)] eval v6 seed $s ANNULÉE"; continue; }
  [ -d "results_v6/cifar10c_seed${s}" ] && cp -r "results_v6/cifar10c_seed${s}" "results_v6/cifar10c_seed${s}.bak" 2>/dev/null
  echo "[$(date +%H:%M)] EVAL v6 seed $s"
  python -m scripts.eval_cifar10c --cifar10c_root data/CIFAR-10-C \
      --results_dir results_v6 --seed $s \
      --output_dir "results_v6/cifar10c_seed${s}"
done

echo; echo "[$(date +%H:%M)] === CONSOLIDATION TERMINÉE ==="
python - << 'PY'
import json, glob, os
print("\n--- accuracies ---")
for pat in ["results_v7/*_gn_seed*/summary.json", "results_v6/exp_hsv_seed*/summary.json"]:
    for p in sorted(glob.glob(pat)):
        d = json.load(open(p))
        print(f"{os.path.basename(os.path.dirname(p)):<28}{d['best_val_acc']*100:>8.2f}%")
print("\n--- robustesse ---")
for root, seeds in [("results_v7", [0,1,2,3,42]), ("results_v6", [0,1,2,3,42])]:
    for s in seeds:
        f = f"{root}/cifar10c_seed{s}/summary.json"
        if not os.path.exists(f): continue
        d = json.load(open(f))
        for e, v in d.items():
            print(f"{root} seed{s:<3}{e:<18} mCA={v['mCA']:.4f}  lum={v['mCA_luminosity']:.4f}")
PY
