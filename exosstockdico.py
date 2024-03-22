import glob

textefile= glob.glob('*.txt')
dict={}

for textefile in textefile:
    with open(textefile,'r') as f:
        dict[textefile]= f.read().splitlines()
print(dict)
