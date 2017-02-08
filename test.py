import numpy as np

l = ["ab", "bc", "cd", "de", "ef", "fg"]
p = np.random.permutation(len(l)) 
m = [ l[i] for i in p]
print m
print l

