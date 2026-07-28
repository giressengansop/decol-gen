# Plan de recherche — suite du mémoire après le retour de l'encadreur

> Basé sur : (1) la synthèse du feedback oral de ton encadreur, (2) le budget de
> temps confirmé : **1 à 2 mois**, (3) point de départ choisi : **vérification
> multi-graines**.

---

## 0. Ce que dit vraiment le retour de l'encadreur (traduit en actions)

Derrière le ton informel, ton encadreur pointe 4 choses concrètes :

1. **Revue de littérature insuffisante** — il faut vérifier ce qui existe déjà sur
   l'effet des espaces colorimétriques (corrélés vs décorrélés) sur
   l'entraînement et la robustesse des CNN, avant d'aller plus loin.
2. **Le résultat "les espaces colorimétriques ne sont pas vraiment
   distinguables" doit être vérifié, pas juste constaté** — sur un seul run
   (seed=42), un seul modèle (ResNet-18), une seule normalisation. Il demande
   explicitement : est-ce que ça tient avec plusieurs graines aléatoires ?
   Plusieurs architectures ? Plusieurs normalisations ?
3. **Si le résultat tient**, il propose une piste positive : peut-être que
   l'espace décorrélé (LAB) n'exprime pas encore tout son potentiel, et qu'une
   normalisation différente (il a fourni de la doc sur Min-Max, Z-score,
   [-1,1], Local Contrast Normalization) pourrait changer la donne.
4. **Si ça échoue aussi**, il ouvre la porte à une piste plus ambitieuse :
   concevoir son propre espace colorimétrique, ou (ton idée à toi) une fonction
   de perte supplémentaire basée sur le canal de luminance.

Il a aussi fourni deux éléments concrets : une liste de modèles alternatifs
(**DenseNet-121, ViT-S/ViT-T, EfficientNet-B4**) et de la documentation sur les
techniques de normalisation d'image, avec des sources (Medium, Ultralytics docs,
ScienceDirect, PyTorch discuss, Albumentations, Towards Data Science, HAL, etc.).

**Diagnostic important à garder en tête** : nous avions déjà, sans le nommer
explicitement, ce résultat en main — au chapitre 3, LAB ne bat RGB que de 0.03
point (quasi ex-æquo), et sur les corruptions de luminosité, RGB fait même
légèrement mieux. Ton encadreur ne remet donc pas en cause tes résultats, il
demande de **prouver qu'ils sont robustes** avant de les interpréter comme une
conclusion scientifique.

---

## Phase 4 — Vérification statistique multi-graines (Semaines 1-2, priorité 1)

**Objectif** : savoir si le classement LAB ≈ RGB > HSV > Gris (et l'inversion
RGB > LAB sur les corruptions de luminosité) est un vrai phénomène ou juste le
bruit d'un seul entraînement.

**Méthode**
1. Ajouter un override `--seed` en ligne de commande dans
   [`scripts/train_model.py`](../scripts/train_model.py) (actuellement le seed
   ne vient que du YAML) pour ne pas dupliquer 20 fichiers de config.
2. Relancer chacune des 4 expériences (`baseline_rgb`, `exp_hsv`, `exp_lab`,
   `exp_grayscale`) avec **5 graines** (ex. 42, 0, 1, 2, 3) → 20 runs de 50
   époques. Regarde d'abord le temps réel d'un run dans tes logs pour savoir si
   5 graines est réaliste dans ton créneau GPU ; si c'est trop long, descends à
   3 graines minimum (c'est le strict minimum pour donner un écart-type
   interprétable).
3. Réévaluer chaque nouveau modèle sur CIFAR-10-C avec
   [`scripts/eval_cifar10c.py`](../scripts/eval_cifar10c.py) — **c'est le plus
   important** : c'est là que l'écart RGB vs LAB (0.822 vs 0.821) est le plus
   fin, donc le plus susceptible de s'inverser avec le bruit d'échantillonnage.
4. Calculer moyenne ± écart-type de `best_val_acc` et de `mCA_luminosity` par
   espace colorimétrique sur les graines, et faire un test simple (t-test
   apparié, ou juste regarder si les intervalles moyenne ± écart-type se
   chevauchent) pour conclure si une différence est significative.

**Critère de décision à la fin de cette phase**
- Si les écarts-types sont plus grands que les écarts entre espaces
  colorimétriques → conclusion confirmée : **le choix d'espace colorimétrique
  n'a pas d'effet significatif dans cette configuration**, avec preuve
  statistique à l'appui (résultat solide et publiable en l'état).
- Si un espace colorimétrique domine de façon stable sur toutes les graines →
  le résultat phase 2/3 initial était en fait robuste, et le finding nuancé sur
  la luminosité tient aussi.

**Fichiers concernés** : `scripts/train_model.py` (ajout CLI), pas de nouveau
fichier de config nécessaire, nouveaux dossiers `results_v4/<exp>_seed<N>/`.

---

## Phase 5 — Vérification de la normalisation (Semaines 3-4, priorité 2)

**Pourquoi celle-ci plutôt que les architectures** : elle réutilise ResNet-18
(zéro risque d'intégration), répond directement à la question concrète de ton
encadreur ("le potentiel de LAB est-il sous-exploité ?"), et si elle change le
résultat, c'est une contribution positive et actionnable pour le mémoire — pas
juste une case cochée. Le test d'architectures (DenseNet/ViT/EfficientNet) est
plus coûteux à intégrer et devient un point de **perspective/travaux futurs**
dans ce budget de 1-2 mois plutôt qu'une expérience menée à fond.

**Méthode**
1. Étendre [`src/color_dg/color_spaces/transforms.py`](../src/color_dg/color_spaces/transforms.py)
   pour proposer plusieurs schémas de normalisation, en gardant l'actuel
   (Z-score par canal via `CIFAR10_STATS`) comme référence :
   - Min-Max déjà implicite (les convertisseurs sortent en [0,1]) → tester
     **sans** normalisation supplémentaire.
   - Centrage [-1, 1] : `pixel * 2 - 1`.
   - Local Contrast Normalization (LCN) — normalisation locale par patch,
     plus complexe, à essayer seulement si le temps le permet.
2. Ne teste que sur **RGB et LAB** (les deux protagonistes du résultat le plus
   fin) pour limiter le nombre de runs — 3 normalisations × 2 espaces × 3
   graines (réutilise les graines de la phase 4) = 18 runs.
3. Compare surtout `mCA_luminosity` : est-ce que LAB dépasse RGB avec une
   normalisation différente ?

**Fichiers concernés** : `transforms.py` (nouvelles fonctions de
normalisation), `scripts/train_model.py` (nouveau champ config
`normalization:`), nouveaux YAML `configs/exp_lab_normXXX.yaml`.

---

## Phase 6 — Revue de littérature ciblée (en parallèle, dès maintenant)

Celle-ci ne demande aucun calcul GPU — à mener en tâche de fond pendant les
phases 4 et 5, pas après.

**Axes de recherche concrets**
- "decorrelated color space CNN generalization/robustness"
- "CIELAB vs RGB deep learning image classification"
- "input color space common corruptions robustness ImageNet-C / CIFAR-10-C"
- "shape bias vs texture bias CNN" (Geirhos et al., Stylized-ImageNet) — lien
  conceptuel fort : la question "le réseau apprend-il la forme ou la couleur"
  est exactement le sujet de ces travaux sur le biais de forme/texture.
- Un angle théorique à chercher spécifiquement, qui expliquerait *pourquoi* tes
  résultats montrent si peu de différence : **une couche convolutive entraînée
  + une BatchNorm peuvent apprendre elles-mêmes une transformation proche
  d'une décorrélation des canaux d'entrée** — donc un réseau entraîné from
  scratch (contrairement à un réseau pré-entraîné figé) pourrait "digérer" la
  différence RGB/LAB dès la première couche, rendant le choix moins critique.
  Cherche des travaux sur "input whitening neural network training" /
  "color space invariance deep learning first layer".

Je peux lancer cette recherche maintenant avec le skill de recherche approfondie
si tu veux — dis-le-moi et je le fais dans ce même échange.

---

## Phase 7 — Pistes ouvertes / perspectives (à mentionner dans le mémoire, pas à mener à fond dans ce budget de temps)

À citer explicitement comme "travaux futurs" dans ta discussion, pour montrer
que tu as bien entendu l'encadreur sans sur-promettre sur 1-2 mois :
- **Généralisation à d'autres architectures** (DenseNet-121, ViT-S/ViT-T,
  EfficientNet-B4) — mentionné par l'encadreur, mais trop coûteux à mener
  sérieusement (nouvelle intégration + graines multiples) dans ce créneau.
- **Fonction de perte auxiliaire sur le canal de luminance** (ton idée) —
  idée prometteuse mais c'est un nouveau design expérimental à part entière
  (définir la perte, la valider, la comparer) : à ne lancer que si les phases
  4 et 5 se terminent avec de la marge, sinon la garder comme proposition
  argumentée en conclusion.
- **Concevoir un espace colorimétrique sur mesure** — piste la plus ambitieuse
  suggérée par l'encadreur si tout le reste échoue ; à ne considérer que si le
  temps restant après les phases 4-5 le permet.

---

## Calendrier récapitulatif (sur 1 à 2 mois)

| Semaine | Action |
|---------|--------|
| 1       | Ajout `--seed` CLI, lancement des 20 runs phase 2 multi-graines (ou 12 si 3 graines) |
| 1-2     | Revue de littérature en parallèle (phase 6) |
| 2       | Réévaluation CIFAR-10-C sur tous les nouveaux checkpoints, analyse moyenne ± écart-type |
| 2       | **Décision** : le résultat "espaces non distinguables" est-il confirmé statistiquement ? |
| 3-4     | Extension `transforms.py` (normalisations), 18 runs RGB/LAB × 3 normalisations × 3 graines |
| 4       | Analyse : la normalisation change-t-elle le résultat pour LAB ? |
| 5 (marge)| Si le temps le permet : piste perte auxiliaire luminance, ou rédaction anticipée de la discussion/conclusion du mémoire |
| 5-6     | Rédaction : intégrer tous ces résultats dans le chapitre "Discussion", section "Robustness checks" + "Threats to validity" |

---

## Prochaine action immédiate

Dis-moi lequel tu veux que je fasse en premier :
1. Lancer la revue de littérature ciblée maintenant (phase 6, gratuite en
   temps GPU).
2. Implémenter l'override `--seed` dans `train_model.py` et générer les
   configs/commandes pour les 20 (ou 12) runs de la phase 4.
