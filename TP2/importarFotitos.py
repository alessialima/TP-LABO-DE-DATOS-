import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

carpeta = 

X = pd.read_csv(carpeta+"/Fashion-MNIST.csv")

img = np.array(X.iloc[782, 1:785]).reshape((28,28)) # agregue 1:785 asi no me cuenta los indices 
plt.imshow(img, cmap='gray')
plt.show() 

