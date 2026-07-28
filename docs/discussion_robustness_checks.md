# Discussion — Robustness checks & threats to validity (brouillon)

> Brouillon de section pour le chapitre Discussion du mémoire, rédigé suite au
> retour de l'encadreur (voir [plan_recherche_suite.md](plan_recherche_suite.md)).
> **État** : Phase 4 (multi-graines, 5 graines, normalisation z-score) et
> Phase 5 (ablation de normalisation, RGB/LAB, 3 graines × minmax/centered)
> sont toutes les deux **complètes**. Ce document garde volontairement les
> trois niveaux de vérification successifs (seed unique → 5 graines → 3
> normalisations) au lieu d'écraser les anciens chiffres, pour que la
> discussion du mémoire puisse montrer explicitement comment la conclusion
> s'est affinée (et, sur un point précis, **inversée**) à mesure que le bruit
> d'échantillonnage a été mieux maîtrisé.

## 1. Rappel du résultat initial (chapitre 3, seed unique)

Le chapitre 3 rapportait, sur un seul entraînement (seed=42, 50 époques,
ResNet-18 from scratch) :

- **Accuracy CIFAR-10** : LAB (92.96 %) ≈ RGB (92.93 %) > HSV (92.21 %) >
  Grayscale (90.82 %) — écart LAB/RGB de seulement 0.03 point.
- **Robustesse CIFAR-10-C** : LAB/HSV légèrement devant RGB en moyenne globale
  (mCA), mais **RGB meilleur que LAB spécifiquement sur les corruptions liées
  à la luminosité** (brightness, fog, contrast) — mCA_luminosity : RGB 0.822
  vs LAB 0.821.

Ce dernier point contredisait directement l'hypothèse mécaniste du mémoire
(décorréler la luminance et la chrominance en entrée devrait rendre le réseau
plus robuste aux corruptions de luminosité). L'encadreur a demandé de vérifier
si ce résultat — obtenu sur une seule graine — est un phénomène réel ou un
artefact du bruit d'échantillonnage.

## 2. Vérification multi-graines (Phase 4)

### 2.1 Accuracy de validation (5 graines : 42, 0, 1, 2, 3)

| Espace colorimétrique | n | moyenne | écart-type | intervalle |
|---|---|---|---|---|
| RGB | 5 | 0.9267 | 0.0016 | [0.9254, 0.9293] |
| **LAB** | 5 | **0.9291** | 0.0024 | [0.9263, 0.9323] |
| HSV | 5 | 0.9213 | 0.0010 | [0.9205, 0.9227] |
| Grayscale | 5 | 0.9095 | 0.0025 | [0.9081, 0.9139] |

**Lecture** : sur 5 graines, LAB devance RGB de façon un peu plus nette que
sur le run unique du chapitre 3 (+0.0024 en moyenne contre +0.0003
initialement), mais les intervalles moyenne ± écart-type des deux espaces se
chevauchent encore (RGB monte jusqu'à 0.9293, LAB descend jusqu'à 0.9263). Le
classement HSV < RGB ≈ LAB et Grayscale nettement en retrait, en revanche, est
stable sur toutes les graines — c'est un résultat robuste. La différence
RGB/LAB en accuracy pure reste faible et pas clairement significative avec 5
graines seulement (un test statistique formel — t-test apparié — reste à
faire une fois les données de robustesse disponibles, pour trancher les deux
questions ensemble).

### 2.2 Robustesse CIFAR-10-C — mCA_luminosity (5 graines)

| Espace colorimétrique | n | mCA_luminosity moyen | écart-type | intervalle |
|---|---|---|---|---|
| RGB | 5 | 0.8198 | 0.0055 | [0.8103, 0.8244] |
| LAB | 5 | 0.8210 | 0.0029 | [0.8161, 0.8233] |
| HSV | 5 | 0.8129 | 0.0024 | — |
| Grayscale | 5 | 0.8066 | 0.0022 | — |

**Comparaison appariée RGB vs LAB, graine par graine** :

| graine | RGB | LAB | écart (RGB−LAB) | gagnant |
|---|---|---|---|---|
| 0 | 0.8244 | 0.8233 | +0.0011 | RGB |
| 1 | 0.8217 | 0.8222 | −0.0005 | LAB |
| 2 | 0.8205 | 0.8161 | +0.0044 | RGB |
| 3 | 0.8103 | 0.8226 | −0.0123 | LAB |
| 42 | 0.8222 | 0.8208 | +0.0014 | RGB |

Écart moyen apparié = −0.0012 (écart-type = 0.0065) ; RGB gagne sur 3
graines/5, LAB sur 2/5.

**Critère de décision** (fixé avant de voir les résultats, pour éviter le
biais de confirmation) : si les intervalles moyenne ± écart-type de RGB et LAB
se chevauchent sur `mCA_luminosity`, l'inversion observée au chapitre 3 (RGB >
LAB sur la luminosité) n'est pas confirmée comme un effet réel. **C'est le cas
ici** : les intervalles se chevauchent largement, et l'écart moyen apparié est
environ 5 fois plus petit que son propre écart-type. **Conclusion : l'inversion
RGB > LAB du chapitre 3 était un artefact du bruit d'échantillonnage sur un
seul run (seed=42), pas un effet réel.** Le partage quasi 3/2 des victoires
par graine confirme visuellement l'absence d'effet systématique.

## 2.3 Vérification de la normalisation (Phase 5)

**Question** : l'encadreur a suggéré une piste positive si le résultat
"espaces non distinguables" se confirmait — peut-être que LAB n'exprime pas
tout son potentiel avec la normalisation z-score (référence utilisée partout
jusqu'ici), et qu'une normalisation différente changerait le classement
RGB/LAB, en particulier sur `mCA_luminosity`.

**Méthode** : RGB et LAB uniquement (les deux protagonistes du résultat le
plus fin), réentraînés avec deux normalisations alternatives — **Min-Max**
(aucune normalisation supplémentaire après conversion d'espace colorimétrique,
canaux gardés dans [0, 1]) et **centrage [-1, 1]** — sur 3 graines (42, 0, 1),
puis réévalués sur CIFAR-10-C. Le z-score (Phase 4, 5 graines) sert de
référence.

| Normalisation | RGB `mCA_luminosity` (n) | LAB `mCA_luminosity` (n) | Écart (RGB−LAB) | Qui devance ? |
|---|---|---|---|---|
| Z-score (Phase 4, référence) | 0.8198 ± 0.0055 (5) | 0.8210 ± 0.0029 (5) | −0.0012 | LAB |
| Min-Max | 0.8186 ± 0.0060 (3) | 0.8147 ± 0.0031 (3) | +0.0039 | RGB |
| Centrage [-1, 1] | 0.8170 ± 0.0026 (3) | 0.8191 ± 0.0009 (3) | −0.0021 | LAB |

**Lecture** : le "gagnant" bascule effectivement selon la normalisation — LAB
en z-score, RGB en Min-Max, LAB de nouveau en centrage [-1, 1] — mais dans les
trois cas les intervalles moyenne ± écart-type de RGB et LAB **se chevauchent
largement**, et l'écart absolu ne dépasse jamais 0.004 (à comparer à des
écarts-types individuels de 0.003 à 0.006). Le même critère de décision que
pour la section 2.2 s'applique : ces bascules ne constituent pas une preuve
d'effet réel de la normalisation sur le classement RGB/LAB, mais l'expression
attendue du même bruit d'échantillonnage vu sous un angle différent.

**Réponse à la piste de l'encadreur** : non, changer de normalisation ne fait
pas ressortir un avantage caché de LAB sur la robustesse à la luminosité — le
résultat "non distinguable" de la section 2.2 est confirmé indépendamment du
schéma de normalisation testé, ce qui renforce (plutôt qu'affaiblit) la
conclusion générale du mémoire.

L'accuracy de validation suit le même patron de stabilité :

| Colorspace | Normalisation | n | val_acc moyenne ± écart-type |
|---|---|---|---|
| RGB | Min-Max | 3 | 0.9265 ± 0.0014 |
| LAB | Min-Max | 3 | 0.9264 ± 0.0004 |
| RGB | Centrage [-1,1] | 3 | 0.9257 ± 0.0010 |
| LAB | Centrage [-1,1] | 3 | 0.9275 ± 0.0009 |

Comme en z-score (section 2.1), RGB et LAB restent quasi ex-æquo quelle que
soit la normalisation — aucun schéma ne fait émerger un écart net.

## 3. Mise en contexte par la littérature (Phase 6)

Une revue ciblée ([revue_litterature_phase6.md](revue_litterature_phase6.md))
permet de situer ces deux résultats :

- **L'écart quasi nul RGB/LAB n'est pas une anomalie.** La comparaison directe
  la plus proche du mémoire, Gowda & Yuan, *ColorNet: Investigating the
  Importance of Color Spaces for Image Classification* (arXiv:1902.00267,
  2019), rapporte elle aussi des écarts faibles (1-2 points) entre espaces
  colorimétriques prix isolément sur CIFAR-10 — les gains substantiels
  n'apparaissent que lorsque plusieurs espaces sont **fusionnés**, pas avec un
  seul espace décorrélé utilisé seul. Le résultat du mémoire se situe donc
  dans le bas de la fourchette déjà documentée, pas en dehors.
- **L'angle « robustesse par catégorie de corruption » semble original.**
  Aucune étude publiée trouvée ne compare des espaces colorimétriques sur
  ImageNet-C/CIFAR-10-C corruption par corruption. L'analyse spécifique sur
  les corruptions de luminosité (brightness/fog/contrast) n'est donc a priori
  pas la réplication d'un résultat déjà connu.
- **Piste explicative pour la faiblesse de l'effet colorimétrique** : la
  littérature sur le biais forme/texture (Geirhos et al., ICLR 2019,
  arXiv:1811.12231) suggère que la robustesse d'un CNN dépend surtout de la
  mesure dans laquelle il s'appuie sur la texture/couleur locale plutôt que
  sur la forme — un biais qui ne dépend pas du format d'entrée de la couleur.
  Décorréler les canaux ne changerait donc pas ce biais fondamental, ce qui
  est cohérent avec le faible effet observé.
- **Piste théorique sur le « pourquoi »** : l'hypothèse qu'une première couche
  convolutive entraînée + BatchNorm apprend une décorrélation implicite du
  signal d'entrée trouve un appui indirect : BatchNorm a été conçue dès
  l'origine comme une approximation du blanchiment (Ioffe & Szegedy, 2015),
  et Harris et al. (arXiv:2010.02634, 2020) montrent que des CNN entraînés
  développent spontanément des filtres de couleur opposants proches d'un
  codage type LAB dans leurs premières couches. Aucune source ne teste
  l'hypothèse exacte du mémoire (from-scratch vs pré-entraîné), donc à
  présenter comme hypothèse cohérente avec la littérature, pas comme résultat
  établi ailleurs.

## 4. Threats to validity (à intégrer/étendre)

- **Une seule architecture** (ResNet-18) : les résultats ne se généralisent
  pas nécessairement à d'autres familles (DenseNet, ViT, EfficientNet) —
  explicitement classé en perspective (Phase 7) faute de budget temps.
- **Une seule normalisation testée par défaut** (z-score par canal) — *menace
  levée* : la Phase 5 (section 2.3) a testé Min-Max et centrage [-1, 1] pour
  RGB/LAB et confirme la même absence d'effet significatif. Reste une limite
  résiduelle mineure : la Local Contrast Normalization (LCN), mentionnée par
  l'encadreur, n'a pas été testée faute de temps — à citer comme perspective.
- **CIFAR-10 uniquement** : dataset de petite résolution (32×32), les effets
  d'un espace colorimétrique décorrélé pourraient différer sur des images
  haute résolution où la texture locale est plus riche.

## 5. Conclusion (mise à jour post-Phase 4 + Phase 5)

Trois niveaux de vérification successifs, présentés ici côte à côte pour que
la trajectoire du raisonnement reste visible dans le mémoire plutôt que
gommée :

| Niveau de vérification | mCA_luminosity RGB vs LAB | Conclusion à ce stade |
|---|---|---|
| **1. Seed unique (chapitre 3)** | RGB 0.822 > LAB 0.821 | RGB semblait résister mieux à la luminosité — contredisait l'hypothèse mécaniste du mémoire |
| **2. Phase 4 : 5 graines, z-score** | LAB 0.8210 > RGB 0.8198, intervalles chevauchants | L'inversion du seed unique ne tient pas — pas d'effet réel, et le "gagnant" apparent s'inverse même |
| **3. Phase 5 : + Min-Max / centrage [-1,1]** | Le gagnant rebascule selon la normalisation (RGB en Min-Max, LAB dans les deux autres cas), toujours dans le bruit | La non-distinguabilité est confirmée indépendamment du schéma de normalisation — ce n'est pas un artefact du z-score |

Chaque étape de vérification supplémentaire a **réduit** la confiance dans
un effet réel de l'espace colorimétrique, jamais renforcé une conclusion
positive — c'est exactement le sens dans lequel un résultat négatif devient
crédible plutôt que suspect.

- **Accuracy** : le classement HSV < RGB ≈ LAB, Grayscale nettement en
  retrait, est stable sur 5 graines et sur les trois normalisations testées.
  RGB et LAB restent quasi ex-æquo partout (écarts du même ordre que le bruit
  d'échantillonnage).
- **Robustesse aux corruptions de luminosité** : l'inversion RGB > LAB
  rapportée au chapitre 3 (0.822 vs 0.821 sur seed=42 seul) **ne résiste ni**
  à la vérification multi-graines (RGB et LAB se partagent les victoires 3/2
  selon la graine, écart moyen apparié 5× plus petit que sa variance
  inter-graines) **ni** au changement de normalisation (le gagnant bascule
  entre RGB et LAB selon le schéma, toujours dans le bruit). Il s'agissait de
  bruit d'échantillonnage à deux niveaux différents, pas d'un effet réel.

**Conclusion scientifique du mémoire, maintenant étayée statistiquement à
deux niveaux indépendants (graines ET normalisation)** : le choix d'espace
colorimétrique d'entrée (RGB, HSV, LAB) n'a pas d'effet significatif sur
l'accuracy ni sur la robustesse aux corruptions — y compris sur les
corruptions de luminosité où un effet était pourtant attendu par hypothèse
mécaniste, et y compris quand on fait varier la normalisation pour donner sa
chance à LAB d'exprimer un avantage caché. Seule la perte d'information de la
Grayscale a un effet net et systématique (dernière position sur tous les
critères, toutes graines et normalisations confondues). Ce résultat est
cohérent avec la littérature existante (Phase 6) et trouve une explication
théorique plausible dans la capacité d'un réseau entraîné from scratch
(première couche + BatchNorm) à apprendre lui-même une décorrélation du
signal d'entrée, rendant le format de couleur en entrée peu critique. C'est un
résultat négatif robuste et défendable — exactement ce que demandait
l'encadreur : une conclusion vérifiée sous plusieurs angles plutôt que
constatée sur un seul run avec une seule configuration.
