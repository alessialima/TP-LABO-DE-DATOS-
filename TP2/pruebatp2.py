import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

#%%
carpetaOriginal = os.path.dirname(os.path.abspath(__file__))
ruta_moda = os.path.join(carpetaOriginal, "Fashion-MNIST.csv", ) 
fashion = pd.read_csv(ruta_moda, index_col=0)
#%%
# Separamos los datos en dos subconjuntos
# X para las imagenes, esto es, los valores de sus pixeles
# Y para las etiquetas, (el tipo de prenda al cual pertenece)
X = fashion.drop('label', axis=1)
Y = fashion['label']
#%%
# Plot imagen
img = np.array(X.iloc[12]).reshape((28,28))
plt.imshow(img, cmap='gray')
plt.colorbar()
plt.show()
#%%
# Contamos con estas 10 prendas, viendo el github del Fashion MNIST podemos observar a que numero pertenece
# cada prenda, 0 es Remera, 1 es Pantalon y asi hasta 9 que es Ankle Boot (Botita)
clases = [
    'Remera', 'Pantalón', 'Suéter', 'Vestido', 'Abrigo',
    'Sandalia', 'Camisa', 'Zapatilla', 'Bolso', 'Botita'
]

plt.figure(figsize=(12, 6))
unique, counts = np.unique(Y, return_counts=True)
plt.bar([clases[i] for i in unique], counts)
plt.title("Cantidad Imagenes por Prenda de Fashion MNIST")
plt.xlabel("Prendas de Ropa")
plt.ylabel("Cantidad")
plt.show()
# Se puede observar que hay 7000 imagenes por prenda

#%%
# 2. Imágenes promedio por clase
plt.figure(figsize=(15, 8))
for i in range(10):
    plt.subplot(2, 5, i+1)
    label = (Y == i)
    if np.sum(label) > 0:
        img = np.mean(X[label], axis=0).to_numpy().reshape(28, 28)
        plt.imshow(img, cmap='gray')
        plt.title(clases[i])
        plt.axis('off')
plt.suptitle('Imágenes Promedio por Clase', fontsize=25)
plt.show()
#%%
