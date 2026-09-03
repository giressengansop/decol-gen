#!/bin/bash
cd ~/decol-gen
echo "════ $(date '+%H:%M:%S') ════"
for d in results_v8 results_v9; do
  [ -d "$d" ] || continue
  tr=$(ls $d/exp_*/summary.json $d/baseline_*/summary.json 2>/dev/null | wc -l)
  ev=$(ls $d/cifar10c_seed*/corruption_summary.csv 2>/dev/null | wc -l)
  echo "$d :  $tr Trainings fertig   |   $ev Evaluationen fertig"
done
echo
echo "── läuft gerade ──"
ps -u $USER -o etime,cmd --no-headers | grep -E "[s]cripts\.(train_model|eval_cifar10c)" \
  | awk '{printf "  %-10s %s %s\n", $1, $4, $5}' | sort -u || echo "  (nichts)"
echo
echo "── GPU ──"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo
echo "── letzte Zeile im Log ──"
for f in results_v8_log.txt results_v9_log.txt; do
  [ -f "$f" ] && echo "  $f : $(grep -E 'Epoch|Experiment|Best val|corruption' $f | tail -1)"
done
