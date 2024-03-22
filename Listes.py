
vistes= [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
villes = ['Paris', 'Lyon', 'Marseille', 'Bordeaux', 'Toulouse', 'Lille', 'Nantes', 'Strasbourg', 'Nice', 'Rennes']
liste_2 = [vistes, villes]
liste_3 = []


##INDEXING
dernierelement= villes[-1]
avantdernierelement= villes[-2]

##SLICING
##Tout les elements de la liste entre le 3eme et le 6eme
villes[3:6]

##Tout les elements de la liste a partir du 3eme
villes[3:]

##Tout les elements de la liste jusqu'au 3eme
villes[:3]

##Tout les deux elements de la liste
villes[::2]

##Tout les deux elements de la liste a l'envers
villes[::-1]

##FONCTIONS
##Ajouter un element a la fin de la liste
villes.append('Toulon')

##Ajouter un element a un index precis
villes.insert(2, 'Brest')

## Ajouter une liste a une autre liste a la fin
villes.extend(['Montpellier', 'Grenoble'])

##longueur liste
len(villes)

## Tri de la liste par ordre alphabetique
villes.sort()
villes.sort(reverse=True)

##compter le nombre d'occurence d'un element dans une liste
villes.count('Paris')

if('Paris' in villes):
    print('Paris est dans la liste')
else:
    print('Paris n\'est pas dans la liste')

for ville in villes:
    print(ville)

for index, ville in enumerate(villes):
    print(index, ville)

##pour les zip la liste la plus courte est prioritaire et arrete la boucle
for vistes, ville in zip(vistes, villes):
    print(vistes, ville)


