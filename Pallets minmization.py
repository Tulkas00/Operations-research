import pandas as pd
from pulp import LpMaximize, LpProblem, LpVariable, LpBinary, lpSum


excel1 = pd.read_excel("resultat_bac.xlsx", sheet_name="Sheet1")
excel2 = pd.read_excel("caracterisation.xlsx", sheet_name="Palettes")

# Création du problème d'optimisation
prob = LpProblem("Optimisation de stockage de palettes", LpMaximize)

# Paramètres
produits = excel1["Id produit"].tolist()
palettes = excel2["Nom"].tolist()
volume_unite = dict(zip(produits, excel1["Volume Unitaire Picking"]))
longueur = dict(zip(palettes, excel2["Longueur"]))
largeur = dict(zip(palettes, excel2["Largeur"]))
hauteur = dict(zip(palettes, excel2["Hauteur"]))
quantite_existante = dict(zip(palettes, excel2["Quantité existante"]))

# Variables binaires
x = LpVariable.dicts("X", (produits, palettes), 0, 1, LpBinary)
y = LpVariable.dicts("Y", (produits, palettes), 0, 1, LpBinary)

# Fonction objective
penalite = 10000  # Choisissez une valeur appropriée pour la pénalité
prob += lpSum(x[i][j] * volume_unite[i] for i in produits for j in palettes) - \
    penalite * lpSum(y[i][j] for i in produits for j in palettes)

# Contraintes
for i in produits:
    # Chaque produit doit être stocké dans une seule palette
    prob += lpSum(x[i][j] for j in palettes) == 1

for j in palettes:
    # Un seul type de produit par palette   
    prob += lpSum(x[i][j] for i in produits) <= 1

for j in palettes:
    # Capacité maximale des palettes
    prob += lpSum(x[i][j] * volume_unite[i] for i in produits) <= (
        longueur[j] * largeur[j] * hauteur[j] * quantite_existante[j])

for i in produits:
    for k in produits:
        if i != k:
            for j in palettes:
                # Contrainte Y : Si Y(i, j) == 1, alors X(i, j) == 1
                prob += y[i][j] <= x[i][j]

for i in produits:
    for k in produits:
        if i != k:
            for j in palettes:
                # Contrainte Y : Y(i, j) + Y(k, j) <= 1
                prob += y[i][j] + y[k][j] <= 1

# Résolution du problème
prob.solve()

for i in produits:
    for j in palettes:
        if x[i][j].varValue == 1:
            print(f"Produit {i} stocké dans la palette {j}")

print(f"Valeur de la fonction objective : {value(prob.objective)}")
