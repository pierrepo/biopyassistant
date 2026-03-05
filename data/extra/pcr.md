# La PCR

## 1. Qu’est-ce que la PCR ?

La **PCR (Polymerase Chain Reaction)**, ou **réaction de polymérisation en chaîne**,
est une technique de biologie moléculaire qui permet **d’amplifier une séquence spécifique d’ADN**.
Autrement dit, elle permet de produire **des millions à des milliards de copies d’un fragment d’ADN à partir d’une très petite quantité initiale**.

La PCR est aujourd’hui une méthode fondamentale en biologie et en médecine. Elle est utilisée pour :

* détecter des agents pathogènes (virus, bactéries),
* analyser des mutations génétiques,
* réaliser du clonage moléculaire,
* préparer des fragments d’ADN pour le séquençage.

La réaction se déroule dans un appareil appelé **thermocycleur**,
qui fait varier la température de manière très précise afin de permettre les différentes étapes de la réaction.

## 2. Les composants nécessaires à la PCR

Une réaction PCR contient généralement :

* **ADN matrice** : la molécule d’ADN contenant la séquence à amplifier
* **Amorces (primers)** : courts fragments d’ADN simple brin (≈18–25 nucléotides) qui délimitent la région à amplifier
* **ADN polymérase thermostable** (ex : Taq polymérase) : enzyme qui synthétise le nouvel ADN
* **dNTPs** : nucléotides nécessaires à la synthèse de l’ADN
* **Tampon réactionnel** et ions Mg²⁺ : nécessaires à l’activité enzymatique

## 3. Les trois phases du cycle PCR

La PCR repose sur la répétition d’un cycle de **trois étapes principales**, généralement répété **25 à 40 fois**.

### 3.1 Dénaturation

Température typique : **94–98 °C**

À haute température, les **deux brins de la double hélice d’ADN se séparent**.
Les liaisons hydrogène entre les bases complémentaires sont rompues.

Résultat :
L’ADN double brin devient **ADN simple brin**, accessible pour les amorces.

### 3.2 Hybridation (ou annealing)

Température typique : **50–65 °C**

Les **amorces se fixent sur les séquences complémentaires de l’ADN matrice**.

Cette étape est très importante car :

* elle détermine **la spécificité de la PCR**
* une température trop basse provoque des fixations non spécifiques
* une température trop élevée empêche la fixation des amorces

La température utilisée dépend principalement de la **température de fusion (Tm)** des amorces.

### 3.3 Élongation (ou extension)

Température typique : **72 °C**

À cette température, l’**ADN polymérase synthétise un nouveau brin d’ADN** à partir de l’amorce en ajoutant des nucléotides complémentaires.

La polymérase progresse le long de la matrice :

* vitesse typique : environ **1000 bases par minute** pour la Taq polymérase.

## 4. Pourquoi la température est-elle si importante ?

La PCR dépend fortement du **contrôle précis de la température**,
car chaque étape correspond à un phénomène moléculaire spécifique.

| Étape        | Température | Rôle                       |
| ------------ | ----------- | -------------------------- |
| Dénaturation | ~95 °C      | séparation des brins d’ADN |
| Hybridation  | ~50–65 °C   | fixation des amorces       |
| Élongation   | ~72 °C      | synthèse du nouvel ADN     |

Si la température est mal choisie :

* les amorces peuvent **ne pas se fixer**
* elles peuvent se fixer **au mauvais endroit**
* la polymérase peut être **moins efficace**

La température d’hybridation est généralement choisie **quelques degrés en dessous de la température de fusion des amorces**.

## 5. Température de fusion des amorces (Tm)

La **température de fusion (Tm)** est la température à laquelle **50 % des duplex ADN (amorce–matrice) sont dissociés**.

Elle dépend de plusieurs facteurs :

* **longueur de l’amorce**
* **composition en bases (A, T, G, C)**
* **concentration en sels**
* **structure de la séquence**

Les bases **G et C** contribuent davantage à la stabilité car elles possèdent **trois liaisons hydrogène**,
contre **deux pour A et T**.

## 6. Modèles simples pour calculer la Tm

### 6.1 Règle de Wallace (méthode simple)

C’est la méthode la plus simple et la plus utilisée pour une estimation rapide.

$$
Tm = 2(A+T) + 4(G+C)
$$

où :

* (A, T, G, C) représentent le nombre de bases dans l’amorce.

#### Exemple

Amorce :

```none
ATGCGTACGA
```

Composition :

* A = 3
* T = 2
* G = 3
* C = 2

Calcul :

$$
Tm = 2(3+2) + 4(3+2)
$$

$$
Tm = 10 + 20 = 30°C
$$

Cette formule fonctionne bien pour des amorces **courtes (≈14–20 bases)**.

### 6.2 Formule pour amorces plus longues

Pour des amorces plus longues (>20 nucléotides) :

$$
Tm = 64.9 + 41 \times \frac{(G+C-16.4)}{N}
$$

où :

* (G+C) = nombre de bases G et C
* (N) = longueur de l’amorce

Cette formule tient mieux compte de la **longueur totale de l’amorce**.

### 6.3 Modèle thermodynamique (Nearest-Neighbor)

Les méthodes modernes utilisent le modèle **Nearest Neighbor**, basé sur les interactions entre **paires de bases adjacentes**.

La formule générale est :

$$
Tm = \frac{\Delta H}{\Delta S + R \ln(C)} - 273.15 + 16.6\log_{10}[Na^+]
$$

où :

* (ΔH) = enthalpie totale
* (ΔS) = entropie totale
* (R) = constante des gaz
* (C) = concentration d’amorce
* ([Na^+]) = concentration en sel

Ce modèle est utilisé dans les logiciels de design d’amorces comme :

* **Primer3**
* **OligoAnalyzer**
* **NCBI Primer-BLAST**

Il est **beaucoup plus précis**, mais nécessite des tables thermodynamiques.

## 7. Choisir la température d’hybridation

Une règle pratique est :

$$
T_{annealing} \approx Tm - 3\text{ à }5°C
$$

Exemple :

Si (Tm = 60°C)

alors

$$
T_{annealing} \approx 55–57°C
$$

Si deux amorces sont utilisées (forward et reverse), on utilise généralement **la Tm la plus basse**.

## 8. Résumé

La **PCR** est une technique essentielle permettant d’amplifier un fragment d’ADN grâce à une succession de cycles thermiques :

1. **Dénaturation** : séparation des brins d’ADN (~95 °C)
2. **Hybridation** : fixation des amorces (~50–65 °C)
3. **Élongation** : synthèse de l’ADN (~72 °C)

La **température de fusion des amorces (Tm)** est un paramètre clé pour déterminer la température d’hybridation.
Elle dépend de la **composition en bases, de la longueur de l’amorce et des conditions chimiques**.

Plusieurs méthodes permettent de la calculer :

* **règle de Wallace** (simple)
* **formules empiriques pour amorces longues**
* **modèles thermodynamiques nearest-neighbor** (les plus précis)

Une bonne estimation de la Tm permet d’optimiser la PCR et d’obtenir **une amplification spécifique et efficace**.
