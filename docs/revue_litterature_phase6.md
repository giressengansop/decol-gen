# Revue de littérature ciblée — espaces colorimétriques décorrélés et CNN (Phase 6)

> Demandée par l'encadreur suite à son retour oral : avant de traiter le
> résultat du chapitre 3 (LAB ne bat RGB que de 0,03 point d'exactitude
> globale, et RGB fait légèrement mieux que LAB sur les corruptions liées à la
> luminosité — mCA_luminosité : RGB 0,822 vs LAB 0,821) comme une conclusion
> scientifique, il faut vérifier ce que dit déjà la littérature sur l'effet
> d'un espace colorimétrique décorrélé (comme CIELAB) en entrée d'un CNN,
> comparé à un espace corrélé comme RGB.

## Question à laquelle répond cette revue

L'hypothèse testée dans le mémoire est la suivante : *décorréler la couleur
dès l'entrée du réseau (séparer explicitement luminance et chrominance,
comme le fait CIELAB) aide un CNN à apprendre des caractéristiques plus
robustes qu'avec un espace corrélé comme RGB, où intensité et couleur sont
mélangées dans les trois canaux.* Cette revue cherche, pour cinq axes définis
avec l'encadreur, si la littérature existante confirme, nuance ou contredit
cette hypothèse, et si elle offre une explication théorique au résultat
observé (quasi ex-æquo RGB/LAB, avec un léger avantage RGB sur les
corruptions de luminosité).

---

## Axe 1 — Espaces colorimétriques décorrélés, généralisation et robustesse des CNN

**Sources trouvées :**

1. **Gowda & Yuan, "ColorNet: Investigating the Importance of Color Spaces for
   Image Classification"** (arXiv:1902.00267, 2019).
   https://arxiv.org/abs/1902.00267 — Sur CIFAR-10, un CNN simple obtient sa
   meilleure exactitude en LAB (80,43 %) contre environ 78-79 % pour les
   autres espaces testés (RGB, HSV, YUV, YCbCr, HED, YIQ) : un écart réel mais
   faible. Le vrai gain vient de la **fusion** de plusieurs espaces
   colorimétriques en parallèle (via des DenseNets séparés), pas d'un seul
   espace décorrélé pris isolément — les auteurs notent explicitement que
   "chaque espace colorimétrique donne une exactitude différente selon les
   classes".

2. **Kamann, Güssefeld, Hutmacher, Metzen & Rother, "Increasing the
   Robustness of Semantic Segmentation Models with Painting-by-Numbers"**
   (ECCV 2020, arXiv:2010.05495). https://arxiv.org/abs/2010.05495 — Ce n'est
   pas un travail sur les espaces colorimétriques à proprement parler, mais
   sur le **biais de forme vs couleur** : en forçant le réseau à ignorer la
   couleur réaliste (mélange avec des couleurs aléatoires par classe), les
   auteurs augmentent le biais de forme et améliorent la robustesse aux
   corruptions (gain sur 74 % des cas testés, jusqu'à +25 % sur le bruit).
   C'est un lien indirect mais pertinent : ça suggère que ce qui compte pour
   la robustesse n'est pas *comment* la couleur est encodée (RGB vs LAB) mais
   *combien* le réseau s'appuie dessus.

3. **Chiu, Wang, Kim, Chen & Ma, "ColorSense: A Study on Color Vision in
   Machine Visual Recognition"** (arXiv:2212.08650, 2022).
   https://arxiv.org/abs/2212.08650v2 — Étude à grande échelle montrant que
   les réseaux de vision (CNN et même CLIP) sont sensibles à la
   discriminabilité couleur premier-plan/arrière-plan, et que ni la taille du
   modèle, ni l'architecture, ni les techniques d'augmentation/entraînement
   avancées ne corrigent significativement ce biais (gain moyen de 0,58 point
   seulement). Cela suggère que le choix de l'espace colorimétrique d'entrée
   n'est probablement pas le levier principal de robustesse couleur — un
   argument cohérent avec le résultat quasi ex-æquo du mémoire.

**Lien avec le résultat du mémoire :** ces trois sources convergent vers une
même idée : décorréler l'espace colorimétrique d'entrée (LAB) donne au mieux
un **petit** gain isolé (ColorNet : +1 à +2 points sur CIFAR-10, du même
ordre de grandeur que les 0,03 point du mémoire), et le vrai levier de
robustesse semble être ailleurs — la dépendance du réseau à la couleur en
général (Painting-by-Numbers, ColorSense) plutôt que le format précis dans
lequel cette couleur est présentée. Le résultat du mémoire n'est donc pas une
anomalie : il est cohérent avec une littérature qui trouve des effets de
l'espace colorimétrique réels mais marginaux quand il est utilisé seul (par
opposition à des approches de fusion multi-espaces).

---

## Axe 2 — CIELAB vs RGB en classification d'images : comparaisons empiriques directes

**Sources trouvées :**

1. **Gowda & Yuan, "ColorNet"** (arXiv:1902.00267, 2019) — déjà cité à l'axe
   1, mais c'est la comparaison directe la plus pertinente trouvée : sur
   CIFAR-10 avec un CNN simple, LAB (80,43 %) > RGB/HSV/YUV/YCbCr (~78-79 %).
   https://arxiv.org/abs/1902.00267

2. Résultats rapportés dans des travaux secondaires trouvés par recherche
   (ResearchGate, "Convolutional Neural Network Image Classification Based on
   Different Color Spaces") : sur CIFAR-10, certaines configurations
   obtiennent l'exactitude la plus haute en LAB et en YCrCb (~95-96 %) contre
   RGB (~94 %), mais d'autres études notent que "les niveaux d'exactitude ne
   sont pas très éloignés, ce qui montre que la conversion d'espace
   colorimétrique donne des résultats plus ou moins équivalents". (Source
   secondaire, non vérifiée dans le détail — à citer avec prudence, mais le
   constat qualitatif — écarts faibles et non systématiques — recoupe
   ColorNet.) https://www.researchgate.net/publication/388595768

3. Une évaluation à grande échelle sur ImageNet (citée dans les résultats de
   recherche, "Systematic evaluation of CNN advances on ImageNet",
   arXiv:1606.02228) rapporte que RGB reste l'espace colorimétrique le plus
   adapté en moyenne pour les CNN à grande échelle, ce qui suggère que
   l'avantage de LAB observé sur de petits jeux de données (CIFAR-10) ne se
   généralise pas nécessairement à plus grande échelle.
   https://arxiv.org/abs/1606.02228

**Lien avec le résultat du mémoire :** la littérature ne montre **aucun
consensus fort** en faveur de LAB sur RGB. Les écarts rapportés sont du même
ordre de grandeur que celui du mémoire (1 à 2 points sur CIFAR-10 dans le
meilleur cas pour LAB), jamais un écart transformateur, et au moins une étude
à grande échelle (ImageNet) penche plutôt pour RGB. Le résultat du mémoire
(écart de 0,03 point, donc plus proche de zéro que même les études les plus
favorables à LAB) s'inscrit dans la partie basse, mais pas aberrante, de la
fourchette d'écarts rapportés dans la littérature.

---

## Axe 3 — Choix de l'espace colorimétrique et robustesse aux corruptions communes (ImageNet-C / CIFAR-10-C)

**Sources trouvées :**

1. **Hendrycks & Dietterich, "Benchmarking Neural Network Robustness to
   Common Corruptions and Perturbations"** (ICLR 2019).
   https://arxiv.org/abs/1903.12261 (variante consultée :
   https://openreview.net/forum?id=HJz6tiCqYm) — Papier fondateur d'ImageNet-C
   / CIFAR-10-C (15 types de corruption, 5 niveaux de sévérité, métrique
   mCE). Ce papier ne teste pas l'effet du choix de l'espace colorimétrique
   d'entrée : il prend RGB comme acquis et fait varier les corruptions, pas
   la représentation.

2. **Chiu et al., "ColorSense"** (arXiv:2212.08650, 2022, déjà cité à l'axe
   1) — c'est la source la plus proche d'une étude "espace colorimétrique +
   robustesse", mais elle porte sur la discriminabilité couleur
   premier-plan/arrière-plan et non sur une comparaison RGB vs LAB vs HSV en
   entrée. Elle confirme que la sensibilité à la couleur est un phénomène
   robuste et difficile à corriger par changement d'architecture ou
   d'entraînement.

**Constat de recherche important :** je n'ai trouvé **aucune étude publiée**
qui compare explicitement plusieurs espaces colorimétriques d'entrée (RGB vs
HSV vs LAB vs niveaux de gris) sur ImageNet-C ou CIFAR-10-C avec des
corruptions catégorisées par type (bruit, flou, météo, numérique), et encore
moins qui isole spécifiquement les corruptions de luminosité (brightness,
fog, contrast) comme le fait le mémoire. **Ce constat est en soi une
information utile pour le mémoire** : l'angle spécifique du mémoire
(comparaison systématique d'espaces colorimétriques sur les sous-catégories
de corruptions de CIFAR-10-C, avec une métrique de robustesse spécifique à la
luminosité) semble être une contribution relativement originale, plutôt
qu'une simple réplication d'un résultat déjà établi ailleurs. Cela renforce
la légitimité du travail mais signifie aussi qu'il n'existe pas de résultat
externe direct auquel comparer le chiffre RGB 0,822 vs LAB 0,821.

---

## Axe 4 — Biais de forme vs biais de texture (Geirhos et al.)

**Sources trouvées :**

1. **Geirhos, Rubisch, Michaelis, Bethge, Wichmann & Brendel, "ImageNet-trained
   CNNs are biased towards texture; increasing shape bias improves accuracy
   and robustness"** (ICLR 2019, arXiv:1811.12231).
   https://arxiv.org/abs/1811.12231 — Résultat central : les CNN entraînés
   sur ImageNet classent les images sur la base de la **texture** plutôt que
   de la **forme**, contrairement aux humains (démontré par des images en
   conflit forme/texture, validé par 48 560 essais psychophysiques sur 97
   observateurs). En entraînant le même ResNet-50 sur Stylized-ImageNet (texture
   randomisée par transfert de style, forme préservée), le réseau développe un
   biais de forme, et cela améliore à la fois la détection d'objets et la
   robustesse à un large éventail de distorsions d'image — un bénéfice
   émergent non recherché directement.

2. **Code et modèles associés** : dépôt GitHub officiel
   https://github.com/rgeirhos/texture-vs-shape et
   https://github.com/rgeirhos/Stylized-ImageNet (ICLR 2019 Oral).

3. **Geirhos et al., "The shape and simplicity biases of adversarially
   robust ImageNet-trained CNNs"** (arXiv:2006.09373) —
   https://arxiv.org/abs/2006.09373 travail de suivi montrant qu'un
   entraînement adversarial (une autre forme de contrainte de robustesse)
   augmente aussi le biais de forme, renforçant le lien entre biais de forme
   et robustesse générale (pas seulement aux corruptions communes mais aussi
   aux attaques adversariales).

**Lien avec le résultat du mémoire :** ce corpus est le lien conceptuel le
plus solide trouvé dans cette revue. Il suggère une explication alternative
et complémentaire à l'hypothèse de décorrélation colorimétrique : ce qui
détermine la robustesse d'un CNN n'est pas tant *la représentation
colorimétrique en entrée* que *le biais forme/texture appris pendant
l'entraînement*. Changer RGB en LAB décorrèle statistiquement les canaux mais
ne force pas le réseau à s'appuyer moins sur la texture/couleur locale — d'où
un effet faible. À l'inverse, des méthodes qui perturbent directement la
texture (Stylized-ImageNet, Painting-by-Numbers à l'axe 1) ont un effet plus
marqué sur la robustesse. Cela peut expliquer *pourquoi* le simple choix
RGB/LAB a si peu d'effet dans le mémoire : la décorrélation des canaux de
couleur n'attaque pas directement le biais de texture qui semble être le
facteur causal identifié par Geirhos et al.

---

## Axe 5 — Blanchiment (whitening) implicite en entrée / BatchNorm et invariance à l'espace colorimétrique

**Sources trouvées :**

1. **Ioffe & Szegedy, "Batch Normalization: Accelerating Deep Network
   Training by Reducing Internal Covariate Shift"** (ICML 2015).
   https://proceedings.mlr.press/v37/ioffe15.html — Papier fondateur de
   BatchNorm. Les auteurs partent explicitement de l'idée que l'entraînement
   converge plus vite quand les entrées du réseau sont "blanchies"
   (linéairement transformées à moyenne nulle, variance unité, et
   décorrélées), mais notent qu'un blanchiment complet est coûteux ; BatchNorm
   est une approximation qui normalise chaque caractéristique
   indépendamment (centrage-réduction) sans décorréler complètement.

2. **Huang, Yang, Lang & Deng, "Decorrelated Batch Normalization"** (CVPR
   2018, arXiv:1804.08450). https://arxiv.org/abs/1804.08450 — Ce papier part
   du constat que la BatchNorm standard ne fait que centrer-réduire, **sans
   décorréler** les canaux/caractéristiques, et propose une extension (DBN)
   qui blanchit réellement les activations (via ZCA plutôt que PCA, pour
   éviter le problème d'échange stochastique d'axes). Le simple fait que ce
   papier existe et apporte un gain confirme, en creux, que BatchNorm seule
   ne décorrèle pas complètement — mais elle en normalise déjà l'échelle, ce
   qui réduit une partie de l'asymétrie entre canaux corrélés (RGB) et
   décorrélés (LAB).

3. **Harris, Mihai & Hare, "How Convolutional Neural Network Architecture
   Biases Learned Opponency and Colour Tuning"** (arXiv:2010.02634, 2020).
   https://arxiv.org/abs/2010.02634 — Montre que des CNN entraînés
   développent spontanément, dans leurs premières couches, un **codage
   couleur de type opposant** (proche de ce que fait CIELAB en séparant
   luminance et chrominance), particulièrement quand l'architecture contient
   un goulot d'étranglement (bottleneck). Sans contrainte de bottleneck, le
   réseau apprend un système de couleur non-linéaire plus complexe. C'est une
   preuve empirique que **le réseau peut apprendre lui-même une forme de
   décorrélation/recodage de la couleur**, indépendamment du format d'entrée.

4. **Olah, Mordvintsev & Schubert, "Feature Visualization"** (Distill, 2017,
   DOI: 10.23915/distill.00007). https://distill.pub/2017/feature-visualization/
   — Bien que ce ne soit pas un papier sur l'entraînement mais sur la
   visualisation de caractéristiques, il illustre un fait connexe utile : les
   praticiens appliquent couramment une transformation de décorrélation
   (décomposition de Cholesky sur les corrélations de couleur mesurées dans
   le jeu d'entraînement) *avant* d'optimiser, précisément parce que les
   canaux RGB bruts sont fortement corrélés et rendent l'optimisation moins
   naturelle. C'est une reconnaissance implicite, dans la pratique de terrain,
   que la corrélation RGB est un problème d'optimisation contournable — mais
   contourné *en dehors* du réseau, pas nécessairement appris par lui.

**Lien avec le résultat du mémoire :** ces sources apportent un support
**partiel mais réel** à l'hypothèse théorique du mémoire ("la première couche
convolutive + BatchNorm apprend une décorrélation implicite"). Trois éléments
convergent : (a) BatchNorm a été conçue dès l'origine comme une approximation
du blanchiment (Ioffe & Szegedy), (b) des CNN entraînés développent
spontanément un codage couleur opposant proche de LAB dans leurs premières
couches (Harris et al.), et (c) la pratique de recherche reconnaît la
corrélation RGB comme un problème d'optimisation contournable par une
transformation apprise ou fixée (Olah et al.). En revanche, aucune des
sources trouvées ne teste **directement** l'hypothèse précise du mémoire (un
ResNet-18 entraîné from scratch avec BatchNorm annule-t-il la différence
RGB/LAB par rapport à un réseau gelé/pré-entraîné sans BatchNorm réapprise) —
c'est une lacune de la littérature, pas une contradiction.

---

## Synthèse finale

Globalement, la littérature ne contredit pas le résultat du mémoire — elle le
rend plausible, sans le confirmer complètement faute d'études directement
comparables. Les comparaisons empiriques RGB vs espaces décorrélés qui
existent (ColorNet notamment) rapportent systématiquement des écarts faibles
(1 à 2 points sur CIFAR-10 dans le meilleur cas), du même ordre de grandeur
que l'écart quasi nul du mémoire (0,03 point) ; aucune étude trouvée ne
rapporte un gain massif et robuste d'un espace décorrélé comme LAB sur RGB en
classification. Sur le volet robustesse aux corruptions, aucune étude
publiée ne semble avoir comparé systématiquement des espaces colorimétriques
sur ImageNet-C/CIFAR-10-C par catégorie de corruption — l'angle du mémoire
(et en particulier l'inversion RGB > LAB spécifiquement sur les corruptions
de luminosité) apparaît donc comme une contribution assez originale plutôt
que la réplication d'un résultat déjà connu. La ligne Geirhos et al. sur le
biais forme/texture offre une explication conceptuelle plausible de la
faiblesse de l'effet : la robustesse semble davantage déterminée par la
mesure dans laquelle le réseau s'appuie sur la texture/couleur locale que par
le format précis dans lequel la couleur lui est présentée — décorréler les
canaux ne change pas ce biais fondamental. Enfin, l'explication théorique
avancée dans le mémoire (première couche + BatchNorm apprenant une
décorrélation implicite) trouve un appui indirect mais réel dans la
littérature : BatchNorm a été conçue comme une approximation du blanchiment
dès l'origine (Ioffe & Szegedy, 2015), et des travaux récents montrent que
les CNN développent spontanément des filtres de couleur opposants proches
d'un codage type LAB dans leurs premières couches (Harris et al., 2020).
Aucune source trouvée ne teste cette hypothèse précise dans un cadre
from-scratch vs pré-entraîné comme le fait implicitement le mémoire — c'est
donc une piste de discussion légitime et défendable, mais qu'il faut présenter
comme une hypothèse théorique cohérente avec la littérature existante, pas
comme un résultat déjà démontré ailleurs.
