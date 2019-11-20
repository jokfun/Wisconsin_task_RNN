from random import randint,shuffle
import numpy as np

def carte(dim=1,feat=2,rule=0):
    #dim : dimension, number of different parameters
    #feat : number of cards / value possible
    #rule : rule to choose/test
    y = [0 for i in range(feat)]
    cop = [y[:] for j in range(feat)]
    for j in range(len(cop)):
        cop[j][j] = 1
    tab = []
    test = []
    lab = []
    for i in range(dim):
        choice = randint(0,len(cop)-1)
        test.append(cop[choice][:])
        lab.append(choice)
        tab+=cop[:]
    tab+=test
    focus = [0 for i in range(feat)]
    try:
        focus[lab[rule]] = 1
    except Exception as e:
        print(focus,lab,rule)
        exit()
    tab = np.concatenate((tab),axis=None)
    test = np.concatenate((test),axis=None)
    return np.array(test),np.array(focus)
        

if __name__=="__main__":
    tab,oth = carte(3,3)
    print(tab,oth)
