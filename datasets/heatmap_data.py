import os
import numpy as np

from PIL import Image

def create_dataset(trials, binary=True):
    folder_name = 'heatmap_plots'
    ext = '.png'
    
    X = []
    Y = []
    
    count = 0

    print('Building dataset...')
    for filename in os.listdir(folder_name):
        
        if count%100==0:
            print(count)
        if count == 5500:
            break
        
        if not filename.endswith(ext):
            continue
        else:
            #get sample
            path = os.path.join(folder_name, filename)
            image = Image.open(path).convert("L")
            arr = np.asarray(image)
            X.append(arr.flatten())
            label = int(filename.split('_')[-1][0]) #filename == '101_12_6.png"
            Y.append(label)
            count += 1
        
    print('Done')
    print()
    
    X, Y = np.array(X), np.array(Y)
    #Y = np.genfromtxt('labels.csv', delimiter=',')
    Y = Y[:count]
    
    Y[Y==2] = 0
    Y[Y==3] = 1
    Y[Y==6] = 2
    
    if binary:
        idx = np.where(Y!=2)
        print(idx)
        X = X[idx]
        Y = Y[idx]
        
    return X, Y
