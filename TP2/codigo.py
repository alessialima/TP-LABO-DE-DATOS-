import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import duckdb as dd
#%%
carpetaOriginal = os.path.dirname(os.path.abspath(__file__))
ruta_moda = os.path.join(carpetaOriginal, "Fashion-MNIST.csv", ) 
fashion = pd.read_csv(ruta_moda, index_col=0)
#%%
# Separamos los datos en dos subconjuntos
# X para las imágenes, esto es, los valores de sus píxeles
# Y para las etiquetas, el tipo de prenda al cual pertenece, que es lo que denominaremos como "clase"
X = fashion.drop('label', axis=1)
Y = fashion['label']

#%%
"""
Contamos con 10 prendas que, con ayuda del github de Fashion-MNIST, 
podemos observar a qué número de clase pertenece cada tipo de prenda:
0 = Remera
1 = Pantalón
2 = Suéter
3 = Vestido
4 = Abrigo
5 = Sandalia
6 = Camisa
7 = Zapatillas
8 = Bolsa
9 = Botita
"""

clases = [
    'Remera', 'Pantalón', 'Suéter', 'Vestido', 'Abrigo',
    'Sandalia', 'Camisa', 'Zapatilla', 'Bolso', 'Botita'
]

# obs: veamos que hay una misma cantidad de prendas por clase 
cantidadPrendasPorClase = fashion['label'].value_counts() 
print(cantidadPrendasPorClase)

# Gráfico que visualiza esta comparación 
plt.figure(figsize=(12, 6))
unique, counts = np.unique(Y, return_counts=True)
plt.bar([clases[i] for i in unique], counts)
plt.title("Cantidad Imagenes por Clase de Fashion MNIST")
plt.xlabel("Prendas de Ropa")
plt.ylabel("Cantidad")
plt.show()
# Se puede observar que hay 7000 imágenes por prenda

#%% Imágenes promedio por clase
plt.figure(figsize=(15, 8))
for i in range(10):
    plt.subplot(2, 5, i+1)
    label = (Y == i)
    if np.sum(label) > 0:
        img = np.mean(X[label], axis=0).to_numpy().reshape(28, 28)
        plt.imshow(img, cmap='bwr')
        plt.title(clases[i], fontsize=20) 
        plt.axis('off')

plt.subplots_adjust(wspace=0.08, hspace=0.08)  

plt.suptitle('Imágenes Promedio por Clase', fontsize=28, y=0.95)
plt.show()
#%%

# Análisis de variabilidad clase 0
plt.figure(figsize=(15, 15))
bolsos = X[Y == 0].sample(100)  # buscamos 100 remeras aleatorias

for i in range(100):
    plt.subplot(10, 10, i+1)
    img = bolsos.iloc[i].values.reshape(28, 28)
    plt.imshow(img, cmap='bwr')
    plt.axis('off')
    
plt.suptitle('Algunos Ejemplos de Clase 0', fontsize=25)
plt.tight_layout()
plt.show()

# Análisis de variabilidad clase 8
plt.figure(figsize=(15, 15))
bolsos = X[Y == 8].sample(100)  # buscamos 100 bolsos aleatorios

for i in range(100):
    plt.subplot(10, 10, i+1)
    img = bolsos.iloc[i].values.reshape(28, 28)
    plt.imshow(img, cmap='bwr')
    plt.axis('off')
    
plt.suptitle('Algunos Ejemplos de Clase 8', fontsize=25)
plt.tight_layout()
plt.show()

#%%
# Función para visualizar una comparación entre dos clases
def compararClases(label1, label2, title):
    clase1 = X[Y == label1].sample(5)
    clase2 = X[Y == label2].sample(5)
    fig, axes = plt.subplots(2, 5, figsize=(10, 3))
    fig.suptitle(title, fontsize=16, y=1.05)
    
    # Usamos dos for in range para obtener imágenes de las dos distintas clases
    for i in range(5):
        img = clase1.iloc[i].values.reshape(28, 28)
        axes[0, i].imshow(img, cmap='bwr')
        axes[0, i].axis('off')
    for i in range(5):
        img = clase2.iloc[i].values.reshape(28, 28)
        axes[1, i].imshow(img, cmap='bwr')
        axes[1, i].axis('off')
    
    
    plt.subplots_adjust(wspace=0.05, hspace=0.05)  
    plt.show()

# Figurar 1 de ejercicio 1.b
compararClases(2, 1, "Comparación entre Sueter y Pantalón")

# Figura 2 de ejercicio 1.b
compararClases(2, 6, "Comparación entre Sueter y Camisa")

# Figura clasificación binaria
compararClases(0, 8, "Comparación entre Remera y Bolso")

#%% ===============================================================================================
# CLASIFICACIÓN BINARIA
# ¿La imagen corresponde a la clase 0 o a la clase 8? 
# =================================================================================================

# A partir del dataframe original, construimos uno nuevo que contenga sólo al subconjunto de imágenes que corresponden a la clase 0 y clase 8

# subconjunto clases 0 y 8
subconjunto_0_8 = dd.sql("""SELECT *
                FROM fashion
                WHERE label = 0 OR label = 8;""").df()
# este subconjunto está balanceado, se tienen 7000 muestras de cada clase (según lo analizado en el punto 1)
# siguiendo la lógica vista, separamos el 85% para train y el restante 15% para test

df_0 = dd.sql("""
    SELECT *
    FROM subconjunto_0_8
    WHERE label = 0
    LIMIT 5950
""").df()

df_8 = dd.sql("""
    SELECT *
    FROM subconjunto_0_8
    WHERE label = 8
    LIMIT 5950
""").df()
#%% gráfico que puse en el informe recién que compara los 2 promedios + la dif  
plt.figure(figsize=(15, 5))    
plt.subplot(1, 3, 1)
label0 = (Y == 0)
img0 = np.mean(X[label0], axis=0).to_numpy().reshape(28, 28)
plt.imshow(img0, cmap='bwr')



plt.subplot(1, 3, 3)
label8 = (Y == 8)
img8 = np.mean(X[label8], axis=0).to_numpy().reshape(28, 28)
plt.imshow(img8, cmap='bwr')



plt.subplot(1,3,2)
remeras = subconjunto_0_8_TRAIN[subconjunto_0_8_TRAIN["label"] == 0].iloc[:, :784].mean()
bolsos = subconjunto_0_8_TRAIN[subconjunto_0_8_TRAIN["label"] == 8].iloc[:, :784].mean()
diferencia = (remeras - bolsos).values.reshape(28, 28)
plt.imshow(diferencia, cmap='bwr')


# espacio entre imágenes
plt.subplots_adjust(wspace=0.12)  
plt.show()
#%%
subconjunto_0_8_TRAIN = pd.concat([df_0, df_8]).sample(frac=1, random_state=1)  # barajamos
# df y train tienen las mismas columnas
diferencia = subconjunto_0_8.merge(subconjunto_0_8_TRAIN, how='outer', indicator=True)
subconjunto_0_8_TEST = diferencia[diferencia['_merge'] == 'left_only'].drop(columns=['_merge'])

#%% KNN
# Promedio de píxeles por clase
remeras = subconjunto_0_8_TRAIN[subconjunto_0_8_TRAIN["label"] == 0].iloc[:, :784].mean()
bolsos = subconjunto_0_8_TRAIN[subconjunto_0_8_TRAIN["label"] == 8].iloc[:, :784].mean()

# Diferencia promedio entre clases
diferencia = (remeras - bolsos).values.reshape(28, 28)

# índices ordenados según el valor al que refieren de menor a mayor 
flat = diferencia.flatten()
indices_ordenados = np.argsort(flat)

# 4 más azules (valores mínimos)
indices_mas_azules = indices_ordenados[:2] #quiero seleccionar los primeros 2 más azules
coordenadas_azules = [divmod(idx, 28) for idx in indices_mas_azules] #para la cuenta recuerdo que hice la transformación, por eso el divmod

# 4 más rojos (valores máximos)
indices_mas_rojos = indices_ordenados[-2:] #los últimos dos corresponden a los dos más rojos
coordenadas_rojos = [divmod(idx, 28) for idx in indices_mas_rojos]

# Convertir coordenadas a números enteros
coordenadas_azules = [(int(x), int(y)) for x, y in coordenadas_azules]
coordenadas_rojos = [(int(x), int(y)) for x, y in coordenadas_rojos]

print("Pixeles más azules:", coordenadas_azules)
print("Pixeles más rojos:", coordenadas_rojos)

plt.imshow(diferencia, cmap='bwr', vmin=-150, vmax=150)
plt.colorbar()
plt.title("Diferencia promedio (remera - bolso)")
ticks = np.arange(0, 28, 2)
plt.xticks(ticks, ticks, rotation=45, fontsize=8)
plt.yticks(ticks, ticks, fontsize=8)

plt.tight_layout()
plt.show()

#%%
# Gracias al gráfico previo podemos ver más fácilmente los atributos que diferencian estas dos clases
# Para facilitar el proceso creamos una función que nos devuelva el índice del píxel según la coordenada deseada
def coordenada_a_indice(fila, columna):
    return fila * 28 + columna

# Píxeles seleccionados más rojos y más azules
p1 = coordenada_a_indice(19, 22)    
p2 = coordenada_a_indice(18, 22)   
p3 = coordenada_a_indice(2, 14)   
p4 = coordenada_a_indice(1, 11)   
combinaciones = {
    "Combo 1 (2 pixeles)": [p1, p3],
    "Combo 2 (3 pixeles)": [p1, p2, p3],
    "Combo 3 (3 pixeles)": [p1, p3, p4],
    "Combo 4 (4 pixeles)": [p1, p2, p3, p4],
}

valores_k = [1, 15, 40, 100]
resultados = []

# Etiquetas
y_train = subconjunto_0_8_TRAIN["label"].values
y_test = subconjunto_0_8_TEST["label"].values

# Entonces evaluamos
for nombre, atributos in combinaciones.items():
    X_train = subconjunto_0_8_TRAIN.iloc[:, atributos].values
    X_test = subconjunto_0_8_TEST.iloc[:, atributos].values

    for k in valores_k:
        modelo = KNeighborsClassifier(n_neighbors=k)
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        resultados.append({
            "combinacion": nombre,
            "k": k,
            "accuracy": acc
        })

# Resultados:
df_resultados = pd.DataFrame(resultados)
print(df_resultados.sort_values(by="accuracy", ascending=False))
# Guiándonos por la exactitud, el mejor modelo de estas combinaciones sería el que toma los 4 píxeles de atributo y un k=100
#%% Matriz de Confusión (chequear pq no se si esta bien)

y_pred = modelo.predict(X_test)

cmKNN = confusion_matrix(y_test,y_pred)
dispKNN = ConfusionMatrixDisplay(confusion_matrix=cmKNN, display_labels=[0,8])

plt.figure(figsize=(12, 10))
dispKNN.plot(cmap='viridis', values_format='d')

dispKNN.ax_.set_xlabel("Predicted", fontsize=12)
dispKNN.ax_.set_ylabel("Actual", fontsize=12)

plt.tight_layout()
plt.show()
#%%
# Entre los experimentos planteados, presentamos uno que nos ha devuelto una menor exactitud. 
# El mismo será el siguiente: 

# Utilizamos como atributo ciertos cálculos con los píxeles, en lugar de individualmente 
# Por ejemplo, definimos una función que nos ayude a identificar si hay una correa de bolso en la parte superior viendo el promedio 
# de píxeles en la parte superior
# También notamos que los bolsos suelen ser más anchos que las remeras, así que buscamos el ancho máximo de píxeles en la imagen
# Pixeles (columnas 0 a 783)
X_train_pixels = subconjunto_0_8_TRAIN.iloc[:, 0:784].values
X_test_pixels = subconjunto_0_8_TEST.iloc[:, 0:784].values

# Etiquetas
y_train = pd.concat([df_0, df_8]).sample(frac=1, random_state=1)["label"].values
y_test = subconjunto_0_8_TEST["label"].values


def calcular_promedio_superior(imagen_flat):
    imagen = imagen_flat.reshape(28, 28)
    franja = imagen[:5, :]  # Primeras 5 filas
    pixeles_no_cero = franja[franja > 0]
    return np.mean(pixeles_no_cero) if len(pixeles_no_cero) > 0 else 0
def calcular_ancho_maximo(imagen_flat):
    imagen = imagen_flat.reshape(28, 28)
    anchos = [np.count_nonzero(fila) for fila in imagen]
    return max(anchos)

# Aplicamos la función a cada imagen
# Para TRAIN
X_train_feat_correa = np.array([calcular_promedio_superior(x) for x in X_train_pixels])
X_train_feat_ancho = np.array([calcular_ancho_maximo(x) for x in X_train_pixels])
X_train_feat = np.column_stack((X_train_feat_correa, X_train_feat_ancho))  # shape: (n_samples, 2)

# Para TEST
X_test_feat_correa = np.array([calcular_promedio_superior(x) for x in X_test_pixels])
X_test_feat_ancho = np.array([calcular_ancho_maximo(x) for x in X_test_pixels])
X_test_feat = np.column_stack((X_test_feat_correa, X_test_feat_ancho))

# Entrenar KNN y evaluar
clasificador = KNeighborsClassifier(n_neighbors=40)
clasificador.fit(X_train_feat, y_train)
y_pred = clasificador.predict(X_test_feat)

accuracy = accuracy_score(y_test, y_pred)
print(f"Exactitud del modelo con 2 atributos (correa + ancho): {accuracy:.4f}")

#%% ===============================================================================================
# CLASIFICACIÓN MULTICLASE
# ¿A cuál de las 10 clases corresponde la imagen? 
# =================================================================================================

# Separamos los datos en desarrollo dev (80%) y validación heldout (20%)
X_dev, X_heldout, Y_dev, Y_heldout = train_test_split(
    X, Y,
    test_size=0.2,          
    stratify=Y,
    random_state=42
)


#%% Ajustamos un modelo de árbol de decisión y lo entrenamos con distintas profundidades del 1 al 10
train_scores = []
test_scores = []

for profundidad in range(1, 11):
    tree = DecisionTreeClassifier(
        max_depth= profundidad,
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

#%% Búsqueda de hiperparámetros con validación cruzada
param_grid = {
    'max_depth': [5, 7, 9, 11, 13],      # Rangos optimizados, OJO ERA HASTA 10 DE PROFUNDIDAD
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

# Resultados de la búsqueda:
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print(f"\nMejores parámetros: {best_params}")
print(f"Mejor accuracy en validación cruzada: {best_score:.4f}")

#%% Evaluación final con conjunto held-out
best_tree = DecisionTreeClassifier(**best_params, random_state=42)
best_tree.fit(X_dev, Y_dev)

# Predecir y evaluar
y_pred = best_tree.predict(X_heldout)
heldout_acc = accuracy_score(Y_heldout, y_pred)
print(f"\nAccuracy en conjunto held-out: {heldout_acc:.4f}")

#%% Matriz de confusión
cm = confusion_matrix(Y_heldout, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clases)

plt.figure(figsize=(12, 10))
disp.plot(cmap='Blues', values_format='d', xticks_rotation=45)

# Ajuste: reducir tamaño del texto en las celdas
for text in disp.text_.ravel():
    text.set_fontsize(7)

# Etiquetas en español
disp.ax_.set_xlabel("Etiqueta Predicha", fontsize=12)
disp.ax_.set_ylabel("Etiqueta Verdadera", fontsize=12)

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

#%% EXTRAS: 
# 1. código del gráfico que utilizamos en el informe para comparar una fracción de prendas de una clase 
plt.figure(figsize=(15, 15))
bolsos = X[Y == 5].sample(20)  

for i in range(20):
    plt.subplot(10, 10, i+1)
    img = bolsos.iloc[i].values.reshape(28, 28)
    plt.imshow(img, cmap='bwr')
    plt.axis('off')
    
plt.suptitle('Algunos Ejemplos de Clase 5', fontsize=25)
plt.tight_layout()
plt.show()

