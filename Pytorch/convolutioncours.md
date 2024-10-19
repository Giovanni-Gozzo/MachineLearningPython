Première Couche de Convolution (conv1)
nn.Conv2d(1, 6, 3, 1) crée une couche de convolution qui prend des images avec 1 canal (par exemple, des images en niveaux de gris) en entrée.
Le premier paramètre (1) spécifie le nombre de canaux d'entrée. Pour des images en couleur RGB, ce nombre serait 3 (un pour chaque canal de couleur : Rouge, Vert, Bleu).
Le deuxième paramètre (6) est le nombre de filtres (ou noyaux) que la couche va utiliser. Cela signifie que cette couche va produire 6 cartes de caractéristiques différentes en sortie. Chaque filtre détecte une caractéristique spécifique dans l'image, comme des bords orientés dans une certaine direction, des textures, etc.
Le troisième paramètre (3) est la taille de chaque filtre, qui est de 3x3 dans ce cas.
Le quatrième paramètre (1) est le pas (stride), qui est de 1. Cela signifie que le filtre se déplace d'un pixel à la fois lorsqu'il parcourt l'image.


Étape 1: Application de la Convolution
Réduction de Taille (Parfois): La convolution elle-même peut réduire la taille de l'image, mais cela dépend de la taille du filtre, du pas (stride) et du padding utilisé. Si le padding est ajusté pour maintenir la taille de l'image, alors la réduction de taille n'est pas significative à cette étape. Cependant, sans padding ou avec un padding insuffisant, la taille de l'image sera réduite.
Création de Filtres: Pour chaque filtre appliqué à l'image, une carte des caractéristiques (feature map) est créée. Si vous définissez 6 filtres dans votre première couche de convolution, vous obtiendrez 6 cartes des caractéristiques différentes. Chaque carte met en évidence différentes caractéristiques de l'image, comme les bords, les textures, etc.
Étape 2: Activation
ReLU (Rectified Linear Unit): Après avoir appliqué les filtres, la fonction d'activation ReLU est souvent utilisée sur les cartes des caractéristiques. ReLU ajoute de la non-linéarité au modèle, permettant au réseau de capturer des relations complexes dans les données. Elle transforme tous les pixels de valeur négative en zéro et laisse les valeurs positives inchangées, ce qui aide à éliminer les effets négatifs et à accélérer la convergence du réseau.
Étape 3: Pooling (Max Pooling)
Réduction de Taille: Le max pooling est une technique utilisée pour réduire encore plus la taille des cartes des caractéristiques. Par exemple, F.max_pool2d(x, 2, 2) réduit la taille de chaque carte des caractéristiques de moitié dans les deux dimensions. Cela est accompli en prenant le maximum sur chaque fenêtre de 2x2 pixels (définie par le kernel de pooling et le stride) dans les cartes des caractéristiques.
Optimisation: Cette réduction de taille diminue le nombre de paramètres à apprendre et le nombre de calculs à effectuer, ce qui rend le réseau plus rapide à entraîner. Elle aide également à rendre le modèle plus généralisable (moins susceptible de surapprendre) en réduisant le détail et en se concentrant sur les caractéristiques les plus saillantes.
Résumé
En combinant ces étapes, les réseaux de neurones convolutifs (CNN) peuvent efficacement apprendre à identifier et à classer des caractéristiques visuelles complexes à partir d'images. Les convolutions extraient et transforment les caractéristiques, tandis que le pooling réduit la dimensionnalité, ce qui simplifie l'information sans perdre l'essentiel pour la reconnaissance des motifs. Cela permet au réseau d'être à la fois efficace en termes de calcul et efficace pour apprendre des représentations de niveau supérieur des données visuelles.
