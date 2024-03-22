liste_1=[]
for i in range(10):
    liste_1.append(i**2)

## List Comprehension
liste_2=[i**2 for i in range(10)]

##Plus rapide et en 1 seule ligne

liste_3= [[i+j for i in range(3)] for j in range(3)]

##Dict comprehension

prenoms = ['Pierre', 'Julien', 'Marie', 'Jean']
dico = {k:v for k,v in enumerate(prenoms)}

ages = [23, 45, 12, 67]
dico = {k:v for k,v in zip(prenoms, ages)}

dico_2 = {k:v for k,v in zip(prenoms, ages) if v > 18}

##Tuple comprehension
tup = tuple(i**2 for i in range(10))


##Exercice

# dictionnaire k:v
#k= 0 - 20
#V = k**2

dicoexo= {k:v for k,v in enumerate([i**2 for i in range(20)])}
print(dicoexo)

carres = {k:k**2 for k in range(20)}
