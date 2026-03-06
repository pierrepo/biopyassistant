# Division entière, quotient et reste

## 1. Définition

La **division entière** est une opération mathématique qui consiste à diviser un nombre entier (le dividende)  par un autre (le diviseur) **sans utiliser de nombres décimaux**.

Elle permet d’obtenir deux résultats :

* **le quotient** : combien de fois le diviseur rentre dans le dividende ;
* **le reste** : ce qui reste après avoir fait ces divisions complètes.

On peut résumer cela par la formule suivante :

$$
dividende = diviseur \times quotient + reste
$$

avec la condition :

$$
0 \le reste < diviseur
$$

Dans une division entière, on distingue donc quatre éléments :

1. **Dividende** : nombre que l'on divise
2. **Diviseur** : nombre par lequel on divise
3. **Quotient** : nombre de divisions complètes possibles
4. **Reste** : ce qui reste après ces divisions

## 2. Exemple

Soit la division :

$$
17 \div 5,
$$

on a alors :

* dividende : **17**
* diviseur : **5**
* quotient : **3**
* reste : **2**

Pourquoi ?

$$
5 \times 3 = 15
$$

et

$$
17 - 15 = 2
$$

Donc :

$$
17 = 5 \times 3 + 2
$$

## 3. Avec Python

Python possède deux opérateurs utiles pour la division entière :

| Opération        | Symbole | Signification       |
| ---------------- | ------- | ------------------- |
| division entière | `//`    | renvoie le quotient |
| modulo           | `%`     | renvoie le reste    |

En reprenant l'exemple précédent :

```python
17 // 5
```

résultat :

```none
3
```

Et :

```python
17 % 5
```

résultat :

```none
2
```

Donc :

```python
17 = 5 * (17 // 5) + (17 % 5)
```

## 4. Applications

### 4.1 Convertir un temps d'incubation en heures et minutes

Soit un temps d'incubation de 215 minutes, on souhaite sa correspondance en heures et minutes.

Approche mathématique :

$$
215 \div 60
$$

$$
60 \times 3 = 180
$$

reste :

$$
35
$$

Donc, ce temps d'incubation correspond à :

* **2 heures**
* **35 minutes**

Approche avec Python :

```python
temps_incubation = 215
heures = temps_incubation // 60
minutes = temps_incubation % 60
print(f"Le temps d'incubation est de {heures} h et {minutes} min")
```

Résultat :

```none
Le temps d'incubation est de 3 h et 35 min
```

## 4.2 Découper une séquence ADN en codons

En biologie moléculaire, les **codons** sont des groupes de **3 nucléotides**.

Si on connaît la longueur d'une séquence ADN, on peut calculer :

* le nombre de **codons complets**
* le nombre de **nucléotides restants**

Exemple pour une séquence de 50 nucléotides.

Approche mathématique :

$$
50 \div 3
$$

$$
3 \times 16 = 48
$$

reste :

$$
2
$$

Donc cette séquence de 50 nucléotides peut être découpée en :

* **16 codons complets**
* **2 nucléotides restants**

Approche avec Python :

```python
longueur_sequence = 50
nombre_nucleotides_par_codon = 3

nombre_codons = longueur_sequence // nombre_nucleotides_par_codon
nombre_nucleotides_restants = longueur_sequence % nombre_nucleotides_par_codon
print(f"Cette séquence est constituéer de {nombre_codons} codons et {nombre_nucleotides_restants} nucléotides restants")
```

Résultat :

```none
Cette séquence est constituéer de 16 codons et 2 nucléotides restants
```

## 4.3 Répartir des échantillons dans des plaques 96 puits

Les plaques de laboratoire contiennent souvent **96 puits**.

Supposons qu'on ait **250 échantillons**.

On veut savoir :

* le nombre de **plaques complètes**
* combien d'échantillons seront présents **sur la dernière plaque**

Approche mathématique :

$$
250 \div 96
$$

$$
96 \times 2 = 192
$$

reste :

$$
58
$$

Donc, les 25O échantillons seront réparties sur :

* 2 plaques pleines ;
* 58 puits sur la troisième.

Approche avec Python :

```python
nombre_echantillons = 250
puits_par_plaque = 96

plaques_entieres = nombre_echantillons // puits_par_plaque
reste_echantillons = nombre_echantillons % puits_par_plaque

print("Plaques complètes:", plaques_entieres)
print("Puits sur la dernière plaque:", reste_echantillons)
```

## Conclusion

Lorsqu'on réalise une division entière, on divise un dividende (le nombre que l'on divise) par un diviseur (le nombre avec lequel on divise) pour obtenir un quotient.

Le reste de la division entière est la différence entre le dividente et le produit du diviseur par le quotient.

Mathématiquement, cela se traduit par :

$$
dividende = quotient \times diviseur + reste
$$

En Python :

```python
quotient = dividende // diviseur
reste = dividende % diviseur
```
