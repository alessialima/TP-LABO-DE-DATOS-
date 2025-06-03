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
# Analisis de variabilidad clase 8
plt.figure(figsize=(15, 8))
bolsos = X[Y == 8].sample(20)  # buscamos 20 bolsos aleatorios

for i in range(20):
    plt.subplot(4, 5, i+1)
    img = bolsos.iloc[i].values.reshape(28, 28)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    
plt.suptitle('Algunos Ejemplos de Clase 8', fontsize=25)
plt.tight_layout()
plt.show()

#%% Analisis de variabilidad de todas las clases
for j in range (10):
    plt.figure(figsize=(15, 8))
    bolsos = X[Y == j].sample(20)  # buscamos 20 sandalias aleatorios
    
    for i in range(20):
        plt.subplot(4, 5, i+1)
        img = bolsos.iloc[i].values.reshape(28, 28)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        
    plt.suptitle(f"Algunos Ejemplos de Clase {j}", fontsize=25)
    plt.tight_layout()
    plt.show()
#%%
# Función para mostrar comparación entre dos clases
def compararClases(label1, label2, title):
    clase1 = X[Y == label1].sample(5)
    clase2 = X[Y == label2].sample(5)
    fig, axes = plt.subplots(2, 5, figsize=(20, 4))
    fig.suptitle(title, fontsize=20, y=1.05)
    
    # Usamos dos for in range para obtener imagenes de las dos distintas clases
    for i in range(5):
        img = clase1.iloc[i].values.reshape(28, 28)
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].axis('off')
    for i in range(5):
        img = clase2.iloc[i].values.reshape(28, 28)
        axes[1, i].imshow(img, cmap='gray')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Figurar 1 de ejercicio 1.b
compararClases(2, 1, "Comparación entre Sueter y Pantalón")

# Figura 2 de ejercicio 1.b
compararClases(2, 6, "Comparación entre Sueter y Camisa")
#%%
