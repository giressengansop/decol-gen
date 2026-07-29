# Discours — Zwischenpräsentation 10 min (29.07.2026)

> Script oral calé sur les 11 slides de
> `presentations/zwischenpraesentation_2026-07-29.pptx`, découpé selon le
> minutage cible d'une présentation intermédiaire de 10 minutes.

## 1 min — Intro & question de recherche (slides 1-2)

Bonjour, je vous présente l'avancement de mon mémoire sur l'effet du choix de
l'espace colorimétrique en entrée d'un réseau de neurones. Ma question de
recherche : est-ce qu'un changement d'espace colorimétrique — RGB, HSV ou LAB
— améliore la capacité d'un réseau à apprendre et à généraliser, en
particulier face à des perturbations réalistes de l'image ? Mon hypothèse :
les espaces décorrélés comme LAB, qui séparent la luminance de la couleur,
pourraient pousser le réseau à s'appuyer davantage sur la structure que sur
la couleur. Je travaille avec CIFAR-10 et sa version corrompue CIFAR-10-C,
sur un ResNet-18 entraîné from scratch, en comparant RGB, HSV, LAB et le
niveau de gris.

## 1 min — Où on en est / dernier meeting (slides 3-4)

Depuis notre échange du 14 juillet : les trois premières phases — pipeline
de conversion, entraînement de base, évaluation de robustesse — étaient déjà
terminées, mais sur un seul entraînement, une seule graine aléatoire. Votre
retour portait sur un point précis : mon résultat — les espaces
colorimétriques ne se distinguent presque pas, et RGB fait même légèrement
mieux que LAB sur les corruptions de luminosité — ne pouvait pas être
accepté comme conclusion scientifique sur la base d'un seul run. Vous m'avez
demandé de le vérifier avec plusieurs graines, et éventuellement avec une
normalisation différente pour voir si LAB n'exprimait pas encore tout son
potentiel. C'est exactement ce sur quoi j'ai travaillé : phase 4
(multi-graines), phase 5 (normalisation), et en parallèle phase 6 (revue de
littérature).

## 3 min — Ce qui a été fait + résultats Phase 4 (slides 5-6) — le cœur

Concrètement : pour la phase 4, j'ai ajouté la possibilité de changer la
graine aléatoire en ligne de commande, puis relancé les quatre espaces
colorimétriques sur cinq graines différentes — vingt entraînements complets
— les 95 jeux de corruptions de CIFAR-10-C (19 corruptions × 5 niveaux)

Résultats : sur la précision de validation, le classement RGB à peu près
égal à LAB, devant HSV, et le niveau de gris nettement en retrait, est
stable sur les cinq graines — LAB devance légèrement RGB, 92,91 % contre
92,67 %, mais les intervalles se chevauchent, donc pas significatif.

Le point le plus important : la robustesse à la luminosité, le
mCA_luminosity. Au chapitre 3, avec une seule graine, RGB semblait battre
LAB — 0,822 contre 0,821 — ce qui contredisait mon hypothèse de départ. Avec
cinq graines, regardées une par une : RGB gagne sur trois graines, LAB sur
deux. L'écart moyen entre les deux est cinq fois plus petit que sa propre
variabilité d'une graine à l'autre. Cette inversion n'était donc pas un
effet réel, c'était le bruit d'un seul entraînement. C'est exactement la
réponse que vous attendiez.

## 2 min — Résultats Phase 5, normalisation (slide 7)

Vous aviez ouvert une piste positive : peut-être que LAB n'exprime pas tout
son potentiel avec la normalisation z-score utilisée partout jusque-là. Pour
la phase 5, j'ai testé deux normalisations alternatives — min-max (pixels
entre 0 et 1) et un centrage entre -1 et 1 — sur RGB et LAB uniquement,
trois graines chacune, en me concentrant toujours sur la robustesse à la
luminosité.

Résultat : le "gagnant" change selon la normalisation — RGB l'emporte en
min-max, LAB dans les deux autres cas — mais dans les trois cas les
intervalles se chevauchent largement, et l'écart ne dépasse jamais 0,004,
plus petit que la variabilité propre à chaque configuration. Donc non :
changer de normalisation ne fait pas ressortir un avantage caché de LAB. Le
résultat "non distinguable" est confirmé une deuxième fois, de façon
indépendante — ce qui le rend plus solide, pas plus fragile.

## 1.5 min — Littérature + conclusion (slide 8)

En parallèle, revue de littérature ciblée sur cinq axes. Message principal :
mon résultat n'est pas une anomalie. L'étude la plus proche, ColorNet,
trouve elle aussi un avantage de LAB seul de seulement un à deux points sur
CIFAR-10 — les vrais gains ne viennent que de la fusion de plusieurs
espaces, pas d'un seul espace utilisé isolément. Je n'ai trouvé aucune étude
qui compare les espaces colorimétriques corruption par corruption sur
CIFAR-10-C — mon angle sur la luminosité semble donc assez original.

Pour expliquer la faiblesse de l'effet : les travaux de Geirhos montrent que
la robustesse dépend surtout du biais forme-texture du réseau, pas du
format d'entrée de la couleur. Et la BatchNorm, conçue dès l'origine comme
une approximation d'un blanchiment des données, pousse les réseaux à
développer spontanément des filtres de couleur proches de ce que fait LAB —
le réseau apprend lui-même à décorréler la couleur, quel que soit le format
d'entrée.

Conclusion, vérifiée sur deux niveaux indépendants — graines et
normalisation : le choix de l'espace colorimétrique n'a pas d'effet
significatif, ni sur la précision, ni sur la robustesse, même sur la
luminosité où j'attendais un effet. Seule la perte d'information du niveau
de gris a un effet net et systématique.

## 1 min — Limites + prochaines étapes (slides 9-10)

Limites : une seule architecture testée (ResNet-18), uniquement CIFAR-10 en
petite résolution, pas eu le temps de tester la normalisation locale par
contraste que vous aviez mentionnée — je les garde comme perspectives.

Pour la suite, avec un peu plus d'un mois avant le rendu, je me concentre
maintenant sur la rédaction : le chapitre discussion, avec ces vérifications
de robustesse et les menaces à la validité, dont j'ai déjà un premier
brouillon. Je proposerai les autres architectures et l'idée d'une perte
auxiliaire sur la luminance comme travaux futurs plutôt que comme nouvelles
expériences, vu le temps restant.

## Clôture (slide 11)

"Voilà pour cette itération — je suis ouvert à vos questions."
