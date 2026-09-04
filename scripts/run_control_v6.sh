#!/bin/bash
# Runs de contrôle avec les constantes de normalisation corrigées.
# Reprend là où il s'est arrêté : un run déjà terminé est sauté.
set -u

cd ~/decol-gen

# Le run déjà fait porte un nom qu'eval_cifar10c ne sait pas retrouver
if [ -d results_v6/exp_lab_fixedstats_seed0 ] && [ ! -d results_v6/exp_lab_seed0 ]; then
    mv results_v6/exp_lab_fixedstats_seed0 results_v6/exp_lab_seed0
    echo "[$(date +%H:%M)] renommé -> results_v6/exp_lab_seed0"
fi

run () {   # $1 = config, $2 = nom du dossier, $3 = graine
    if [ -f "results_v6/$2/summary.json" ]; then
        echo "[$(date +%H:%M)] SKIP $2 (déjà fait)"
        return
    fi
    echo "[$(date +%H:%M)] START $2"
    python -m scripts.train_model --config "configs/$1" \
        --seed "$3" --output_dir "results_v6/$2"
    echo "[$(date +%H:%M)] DONE  $2"
}

for s in 0 1 42; do
    run exp_lab.yaml      "exp_lab_seed$s"      "$s"
    run baseline_rgb.yaml "baseline_rgb_seed$s" "$s"
done

echo
echo "[$(date +%H:%M)] === Entraînements terminés ==="
python - << 'PY'
import json, glob, os
print(f"{'run':<28}{'best_val_acc':>14}")
print("-" * 42)
for p in sorted(glob.glob("results_v6/*/summary.json")):
    d = json.load(open(p))
    print(f"{os.path.basename(os.path.dirname(p)):<28}{d['best_val_acc']*100:>13.2f}%")
PY
