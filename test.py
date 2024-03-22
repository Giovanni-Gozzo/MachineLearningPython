import numpy

x = 1


def fibonnacci(n):
    nombre = 0
    array = numpy.array([0, 1, 1])
    if(n<2):
        return array
    while (nombre < n):
        longueur=len(array)
        nombre = array[longueur-1] + array[longueur-2]
        array = numpy.append(array, nombre)
    return array[0:len(array)-1]

print(fibonnacci(10))


