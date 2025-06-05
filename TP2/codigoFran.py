import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
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
#%% Punto 3 a


# Separamos los datos en desarrollo dev (80%) y validación heldout (20%)
X_dev, X_heldout, Y_dev, Y_heldout = train_test_split(
    X, Y,
    test_size=0.2,          
    stratify=Y,
    random_state=42
)


#%% b
#Ajustamos un modelo de arbol de decisión y lo entrenamos con distintas profundidades del 1 al 10
train_scores = []
test_scores = []

for profundidad in range(1, 11):
    tree = DecisionTreeClassifier(
        max_profundidad= profundidad,
        random_state=42
    )
    tree.fit(X_dev, Y_dev)
    
    train_acc = tree.score(X_dev, Y_dev)
    test_acc = tree.score(X_heldout, Y_heldout)
    
    train_scores.append(train_acc)
    test_scores.append(test_acc)


plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), train_scores, 'o-', label='Train')
plt.plot(range(1, 11), test_scores, 'o-', label='Test')
plt.xlabel('Profundidad del Árbol')
plt.ylabel('Exactitud')
plt.title('Rendimiento vs Profundidad del Árbol')
plt.xticks(range(1, 11))
plt.legend()
plt.grid(True)
plt.show()

#%% c. Búsqueda de hiperparámetros con validación cruzada
param_grid = {
    'max_depth': [5, 7, 9, 11, 13],      # Rangos optimizados
    'min_samples_split': [5, 10, 20],
    'min_samples_leaf': [2, 4, 8],
    'max_features': ['sqrt', None]        # Reducción para eficiencia
}

tree = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(
    estimator=tree,
    param_grid=param_grid,
    cv=3,                   # 3-fold para mayor velocidad
    scoring='accuracy',
    n_jobs=-1,              # Paralelizar usando todos los núcleos
    verbose=1               # Mostrar progreso
)

grid_search.fit(X_dev, Y_dev)

# Resultados de la búsqueda
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print(f"\nMejores parámetros: {best_params}")
print(f"Mejor accuracy en validación cruzada: {best_score:.4f}")

#%% d. Evaluación final con conjunto held-out
best_tree = DecisionTreeClassifier(**best_params, random_state=42)
best_tree.fit(X_dev, Y_dev)

# Predecir y evaluar
y_pred = best_tree.predict(X_heldout)
heldout_acc = accuracy_score(Y_heldout, y_pred)
print(f"\nAccuracy en conjunto held-out: {heldout_acc:.4f}")

# Matriz de confusión
cm = confusion_matrix(Y_heldout, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, 
    display_labels=clases
)

plt.figure(figsize=(12, 10))
disp.plot(cmap='Blues', values_format='d', xticks_rotation=45)
plt.title('Matriz de Confusión (Conjunto Held-out)')
plt.tight_layout()
plt.show()

# Análisis de errores
error_rates = cm / cm.sum(axis=1)[:, np.newaxis]
np.fill_diagonal(error_rates, 0)  # Eliminar aciertos

max_errors = []
for i in range(10):
    for j in range(10):
        if i != j and error_rates[i, j] > 0.01:  # Filtrar errores significativos
            max_errors.append((i, j, error_rates[i, j]))

max_errors.sort(key=lambda x: x[2], reverse=True)
print("\nPrincipales confusiones:")
for i, j, rate in max_errors[:10]:
    print(f"{clases[i]} → {clases[j]}: {rate:.2%}")
