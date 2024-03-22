import numpy as np

traduction = {
    "chien": "dog",
    "chat": "cat",
    "souris": "mouse",
    "oiseau": "bird",
    "vache": "cow",
    "cochon": "pig",
    "mouton": "sheep",
    "cheval": "horse",
    "poule": "chicken"
}
inventaire = {
    "pommes": 5,
    "bananes": 7,
    "poires": 2
}

dictionnaire3 = {
    "dict1": traduction,
    "dict2": inventaire
}

parametres = {
    "W1": np.random.randn(2, 4),
    "b1": np.random.randn(2, 1),
    "W2": np.random.randn(3, 2),
    "b2": np.random.randn(3, 1)
}

inventaire.values()
inventaire.keys()
len(inventaire)
inventaire["pommes"] = 10
inventaire["cerises"] = 15

# pop permet de supprimer un element du dictionnaire et de le retourner
inventaire.pop("cerises")


def trier(classeur, nombre):
    # classeur : dictionnaire taille 2
    # nombre : int
    # Range nombre dans "positif" ou "negatif" selon sa valeur
    if nombre > 0:
        classeur["positif"].append(nombre)
    else:
        classeur["negatif"].append(nombre)
    return classeur


classeur = {
    "positif": [],
    "negatif": []
}

