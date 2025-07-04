#%%
# Nombre del grupo: Amichis_Labo
# Participantes:
#     Francisco Margitic
#     Alessia Lourdes Lima
#     Katerina Lichtensztein
# Contenido del archivo:
#     Importaciones
#     Carga de datos
#     Funciones definidas
#     Separación del dataset en variables explicativas y variables a explicar
#     Exploración de datos
#     Clasificación binaria
#     Selección de Atributos
#     Clasificación Multiclase
#%% ===============================================================================================
# IMPORTACIONES 
# =================================================================================================
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, classification_report
import duckdb 
import random 
#%% ===============================================================================================
# CARGA DE DATOS 
# =================================================================================================
carpetaOriginal = os.path.dirname(os.path.abspath(__file__))
ruta_moda = os.path.join(carpetaOriginal, "Fashion-MNIST.csv", ) 
fashion = pd.read_csv(ruta_moda, index_col=0)
#%% ===============================================================================================
# FUNCIONES DEFINIDAS
# =================================================================================================
# Creamos una función para probar distintos árboles de decisión en la clasificación multiclase, 
# variando su profundidad máxima y analizando su exactitud

def resultado() -> dict():
    res = {}
    
    for profundidad in range(1, 11): #profundidades entre 1 y 10
     tree = DecisionTreeClassifier(
        max_depth = profundidad,
        random_state=42
     )
     tree.fit(X_dev, Y_dev)
    
     y_predict = tree.predict(X_dev)
     score = accuracy_score(Y_dev, y_predict) * 100
    
     res[profundidad] = score
     print(f'Profundidad: {profundidad}, Exactitud: {score:.2f}%') # print para observar el avance
     
     train_acc = tree.score(X_dev, Y_dev)

     train_scores.append(train_acc)
    return res 
#%% ===============================================================================================
# SEPARACIÓN DE VARIABLES 
# =================================================================================================
# X variable explicativa: para las imagenes, esto es, los valores de sus pixeles
# Y variable a explicar: para las clases (el tipo de prenda al cual pertenece)
X = fashion.drop('label', axis=1)
Y = fashion['label']

#%% ===============================================================================================
# EXPLORACIÓN DE DATOS 
# =================================================================================================
"""
Contamos con 10 prendas que, con ayuda del github de Fashion-MNIST, 
podemos observar a qué tipo de prenda pertenece cada número de clase. 
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


# Veamos que hay una misma cantidad de prendas por clase 
cantidadPrendasPorClase = fashion['label'].value_counts() 
print(cantidadPrendasPorClase)

# Gráfico que visualiza la comparación de prendas por clase: 
plt.figure(figsize=(12, 6))
unique, counts = np.unique(Y, return_counts=True)
plt.bar([clases[i] for i in unique], counts)
plt.title("Cantidad Imagenes por Clase de Fashion MNIST", fontsize = 14)
plt.xlabel("Prendas de Ropa", fontsize= 14)
plt.ylabel("Cantidad", fontsize= 14)
plt.show()
# obs: se puede observar que hay 7000 imágenes por prenda

#%% Imágenes promedio por clase
# Para buscar un patrón que describa cada clase, realizamos un promedio de intensidad de píxeles de cada una de ellas
plt.figure(figsize=(15, 8))
for i in range(10):
    plt.subplot(2, 5, i+1)
    label = (Y == i)
    
    if np.sum(label) > 0:
        img = np.mean(X[label], axis=0).to_numpy().reshape(28, 28)
        im = plt.imshow(img, cmap='bwr')
        plt.title(clases[i], fontsize=20) 
        plt.axis('off')
       

plt.subplots_adjust(wspace=0.08, hspace=0.08)  
cax = plt.axes([0.92, 0.15, 0.02, 0.7])
cbar = plt.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=18)

plt.suptitle('Imágenes Promedio por Clase', fontsize=28, y=0.95)

plt.show()



#%% Imágenes desviación estándar por clase
# Para analizar la diferencia entre prendas de una misma clase, analizamos la desviación estándar de intesidad de píxeles de cada una de ellas 
plt.figure(figsize=(15, 8))
for i in range(10):
    plt.subplot(2, 5, i+1)
    label = (Y == i)
    if np.sum(label) > 0:
        img = np.std(X[label], axis=0).to_numpy().reshape(28, 28)
        im = plt.imshow(img, cmap='bwr')
        plt.title(clases[i], fontsize=20) 
        plt.axis('off')

plt.subplots_adjust(wspace=0.08, hspace=0.08)  
cax = plt.axes([0.92, 0.15, 0.02, 0.7])
cbar = plt.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=18)

plt.suptitle('Imágenes Desviación Estándar por Clase', fontsize=28, y=0.95)
plt.show()

#%% Diferencia Promedio por Clases 
# Promedio de píxeles por clase
sueter = fashion[fashion["label"] == 2].iloc[:, :784].mean()
camisa = fashion[fashion["label"] == 6].iloc[:, :784].mean()

# Mostrar diferencia promedio entre clases
diferencia = (sueter - camisa).values.reshape(28, 28)
plt.imshow(diferencia, cmap='bwr', vmin=-150, vmax=150)
plt.colorbar()
plt.title("Diferencia promedio (sueter - camisa)")
ticks = np.arange(0, 28, 2)
plt.xticks(ticks, ticks, rotation=45, fontsize=8)
plt.yticks(ticks, ticks, fontsize=8)
plt.gca().grid(False) 

plt.tight_layout()
plt.show()
#%% ===============================================================================================
# CLASIFICACIÓN BINARIA
# ¿La imagen corresponde a la clase 0 o a la clase 8? 
# =================================================================================================


#%% A partir del dataframe original, construimos uno nuevo que contenga sólo al subconjunto de imágenes que corresponden a la clase 0 y clase 8

# Subconjunto clases 0 y 8
subconjunto_0_8 = duckdb.sql("""
                SELECT *
                FROM fashion
                WHERE label = 0 OR label = 8;
                """).df()
# Este subconjunto está balanceado, se tienen 7000 muestras de cada clase (según lo analizado en el punto 1)
# Separamos el 80% para train y el restante 20% para test, ambos cantidades pares así que hacemos mitad de cada clase
df_0 = duckdb.sql("""
    SELECT *
    FROM subconjunto_0_8
    WHERE label = 0
    LIMIT 5600
    """).df()

df_8 = duckdb.sql("""
    SELECT *
    FROM subconjunto_0_8
    WHERE label = 8
    LIMIT 5600
    """).df()

subconjunto_0_8_TRAIN = pd.concat([df_0, df_8]).sample(frac=1, random_state=1)  # barajamos
# df y train tienen las mismas columnas
diferencia = subconjunto_0_8.merge(subconjunto_0_8_TRAIN, how='outer', indicator=True)
subconjunto_0_8_TEST = diferencia[diferencia['_merge'] == 'left_only'].drop(columns=['_merge'])
#%% ===============================================================================================
# SELECCIÓN DE ATRIBUTOS
# =================================================================================================

# Promedio de píxeles por clase
remeras_promedio = subconjunto_0_8_TRAIN[subconjunto_0_8_TRAIN["label"] == 0].iloc[:, :784].mean()
bolsos_promedio = subconjunto_0_8_TRAIN[subconjunto_0_8_TRAIN["label"] == 8].iloc[:, :784].mean()

# Diferencia promedio entre clases
diferencia = (remeras_promedio - bolsos_promedio).values.reshape(28, 28)

# índices ordenados según el valor al que refieren de menor a mayor 
flat = diferencia.flatten()
indices_ordenados = np.argsort(flat)

# Más azules (valores mínimos)
indices_mas_azules = indices_ordenados[:10] #quiero seleccionar los primeros 2 más azules
coordenadas_azules = [divmod(idx, 28) for idx in indices_mas_azules] #para la cuenta recuerdo que hice la transformación, por eso el divmod

# Más rojos (valores máximos)
indices_mas_rojos = indices_ordenados[-10:] #los últimos dos corresponden a los dos más rojos
coordenadas_rojos = [divmod(idx, 28) for idx in indices_mas_rojos]

# Convertir coordenadas a enteros
coordenadas_azules = [(int(x), int(y)) for x, y in coordenadas_azules]
coordenadas_rojos = [(int(x), int(y)) for x, y in coordenadas_rojos]

print("Pixeles más azules:", indices_mas_azules, "\n", "· Coordenadas:", coordenadas_azules)
print("Pixeles más rojos:", indices_mas_rojos, "\n", "· Coordenadas:",  coordenadas_rojos)


#%%
combo1 = []
for i in range(2):
    combo1.append(int(indices_mas_azules[i]))
for j in range(2):
    combo1.append(int(indices_mas_rojos[j]))
    
print("combo 1", combo1)   

combo2 = []
for i in range(len(indices_mas_azules)):
    combo2.append(int(indices_mas_azules[i]))
for j in range(len(indices_mas_rojos)):
    combo2.append(int(indices_mas_rojos[j]))
    
print("combo 2", combo2)   

#%%

num = random.randint(500, 560)
a = num

combinaciones = { 'combo 1 (65,+ nro random)': [a, combo1[0], combo1[3]],
                 'combo 2 (65, 554)': [combo1[3], combo1[0]], 
                 'combo 3 (65, 526)': [combo1[3], combo1[1]],
                 'combo 4 (526, 43)': [combo1[1], combo1[2]], 
                 'combo 5 (554, 526, 43)': [combo1[0], combo1[1], combo1[2]],
                 'caso todos': combo1
                 
                 }

# notamos que el pixel combo1[3] no es necesario, pero que los demas, en conjunto, si

valores_k = range(1, 20)
resultados = []

# Etiquetas
y_train = subconjunto_0_8_TRAIN["label"].values
y_test = subconjunto_0_8_TEST["label"].values

# Evaluación 
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
            "accuracy": acc * 100
        })

# Resultados:
df_resultados = pd.DataFrame(resultados)
print(df_resultados.sort_values(by="accuracy", ascending=False))

print(a)

#%% Gráfico exactitud por k segun mejores combos 
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.lineplot(
    data=df_resultados,
    x='k',
    y='accuracy',
    hue='combinacion',
    marker='o',
    linewidth=2
)
plt.title('Exactitud según profundidad para cada combinación de píxeles')
plt.xlabel('Profundidad (k)')
plt.ylabel('Exactitud (%)')
plt.xticks(valores_k)
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()


#%% Reporte de Clasificación (buscamos mostrar las métricas en un cuadro)

reporte = classification_report(y_test, y_pred)
lineas = reporte.split("\n")
lineas_filtradas = [linea for linea in lineas if "macro avg" not in linea and "weighted avg" not in linea]
print("Reporte de clasificación:", "\n", "\n".join(lineas_filtradas))

# Guiándonos por la exactitud, el mejor modelo de estas combinaciones sería el que toma los 4 píxeles de atributo y un k=5

#%% Matriz de Confusión de los mejores 4 casos

combos_seleccionados = []
k_seleccionadas = []

for i in range(len(combos_seleccionados)): 
    pixeles_seleccionados = combos_seleccionados[i]
    k = k_seleccionadas[i]
    X_train = subconjunto_0_8_TRAIN.iloc[:, pixeles_seleccionados].values
    X_test = subconjunto_0_8_TEST.iloc[:, pixeles_seleccionados].values

    modelo = KNeighborsClassifier(n_neighbors = k) 
    modelo.fit(X_train, y_train) 
    y_pred = modelo.predict(X_test) 

# Calculo Exactitud 
    accuracy = accuracy_score(y_test, y_pred) 
    print("Mejores Casos","\n",f"Combo: {i+1}",f"Accuracy: {accuracy:.3f}") 

# Gráfico de la matriz de confusión: 
    y_pred = modelo.predict(X_test) 

    cmKNN = confusion_matrix(y_test, y_pred) 
    dispKNN = ConfusionMatrixDisplay(confusion_matrix = cmKNN, display_labels = [0,8])

    plt.figure(figsize=(2.5, 2.5)) 
    dispKNN.plot(cmap='Blues', values_format='d', colorbar = False, ax=plt.gca()) 
    plt.gca().set_aspect('equal', 'box')
    plt.gca().grid(False) 

    dispKNN.ax_.set_xlabel("Predicted", fontsize = 12) 
    dispKNN.ax_.set_ylabel("Actual", fontsize = 12) 

    plt.tight_layout()
    plt.show()


#%% Gráfico que visualiza la comparación entre los promedios de ambas clases
# además de mostrar en los extremos el promedio de cada clase en sí
plt.figure(figsize=(15, 5))    
plt.subplot(1, 3, 1)
label0 = (Y == 0)
img0 = np.mean(X[label0], axis=0).to_numpy().reshape(28, 28)
plt.imshow(img0, cmap='bwr')
plt.colorbar(fraction=0.046, pad=0.04)  
plt.gca().grid(False) 

plt.subplot(1, 3, 3)
label8 = (Y == 8)
img8 = np.mean(X[label8], axis=0).to_numpy().reshape(28, 28)
plt.imshow(img8, cmap='bwr_r')
plt.colorbar(fraction=0.046, pad=0.04)  
plt.gca().grid(False) 

plt.subplot(1,3,2)
remeras = subconjunto_0_8_TRAIN[subconjunto_0_8_TRAIN["label"] == 0].iloc[:, :784].mean()
bolsos = subconjunto_0_8_TRAIN[subconjunto_0_8_TRAIN["label"] == 8].iloc[:, :784].mean()
diferencia = (remeras - bolsos).values.reshape(28, 28)
im_diff = plt.imshow(diferencia, cmap='bwr')
for (fila, columna) in coordenadas_rojos:
    plt.scatter(columna, fila, marker='x', color='black', s=100, linewidth=2)  
for (fila, columna) in coordenadas_azules:
    plt.scatter(columna, fila, marker='x', color='black', s=100, linewidth=2) 
plt.gca().grid(False) 

cbar = plt.colorbar(im_diff, fraction=0.046, pad=0.04)
cbar.set_ticks([diferencia.min(), diferencia.max()])
cbar.set_ticklabels(['BOLSO', 'REMERA']) 
cbar.ax.tick_params(labelsize=10)


plt.subplots_adjust(wspace=0.3, right=0.85) 

plt.show()

#%% Gráfico que visualiza sólo la diferencia de promedios de clase 
plt.imshow(diferencia, cmap='bwr', vmin=-150, vmax=150)
plt.colorbar()
plt.title("Diferencia promedio (remera - bolso)")
ticks = np.arange(0, 28, 2)
plt.xticks(ticks, ticks, rotation=45, fontsize=8)
plt.yticks(ticks, ticks, fontsize=8)
plt.gca().grid(False) 

plt.tight_layout()
plt.show()
#%% Gráficos: desviación de pantalón, diferencia promedio pantalón y zapatilla, desviación de camisa
plt.figure(figsize=(15, 5))    
plt.subplot(1, 3, 1)
label0 = (Y == 1)
img0 = np.std(X[label0], axis=0).to_numpy().reshape(28, 28)
plt.imshow(img0, cmap='bwr')
plt.colorbar(fraction=0.046, pad=0.04)
plt.gca().grid(False) 

plt.subplot(1, 3, 3)
label8 = (Y == 6)
img8 = np.std(X[label8], axis=0).to_numpy().reshape(28, 28)
plt.imshow(img8, cmap='bwr')
plt.colorbar(fraction=0.046, pad=0.04)
plt.gca().grid(False) 

plt.subplot(1,3,2)
zapatilla = fashion[fashion["label"] == 7].iloc[:, :784].mean()
pantalones = fashion[fashion["label"] == 1].iloc[:, :784].mean()
diferencia = (zapatilla - pantalones).values.reshape(28, 28)
plt.imshow(diferencia, cmap='bwr')
plt.colorbar(fraction=0.046, pad=0.04)
plt.gca().grid(False) 

plt.subplots_adjust(wspace=0.3, right=0.85)  
plt.show()
#%% ===============================================================================================
# CLASIFICACIÓN MULTICLASE
# ¿A cuál de las 10 clases corresponde la imagen? 
# =================================================================================================

# Separamos los datos en desarrollo dev (80%) y validación heldout (20%)
X_dev, X_heldout, Y_dev, Y_heldout = train_test_split(
    X, Y,
    test_size=0.2,          
    stratify=Y,
    random_state=20
)

#%% Ajustamos un modelo de arbol de decisión y lo entrenamos con distintas profundidades del 1 al 10

train_scores = []
resultados = resultado() #acá utilizamos la función definida al comienzo del archivo
#%%
print(resultados)
#%% Gráfico Rendimiento vs Profundidad del Árbol

profundidades = list(resultados.keys())
exactitudes = list(resultados.values())

plt.figure(figsize=(8, 5))

plt.plot(profundidades, exactitudes, 'o-', color='blue', markersize=8, linewidth=2, label='Exactitud (Dev)')
plt.xlabel('Profundidad del Árbol', fontsize = 13)
plt.ylabel('Exactitud (%)', fontsize= 13)
plt.title('Rendimiento vs Profundidad del Árbol', fontsize= 13)
plt.xticks(range(1, 11))

plt.grid(True)
plt.show()

#%% Búsqueda de hiperparámetros con validación cruzada 
# Elegimos algunos valores al azar, manteniendo la profundida máxima en 10
# Max_features lo dejamos fijo en None porque al considerar todos los atributos conseguimos una mayor exactitud
# El criterio lo dejamos por default en gini porque es menos costoso y no disminuye de manera considerable la exactitud
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [5, 10, 20],
    'min_samples_leaf': [2, 4, 8],
    'max_features': [None]    
}

tree = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(
    estimator=tree,
    param_grid=param_grid,
    cv=10,                  
    scoring='accuracy',
    n_jobs=-1,              
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

for text in disp.text_.ravel():
    text.set_fontsize(7)

# Etiquetas
disp.ax_.set_xlabel("Etiqueta Predicha", fontsize=12)
disp.ax_.set_ylabel("Etiqueta Verdadera", fontsize=12)

plt.title('Matriz de Confusión (Conjunto Held-out)')
plt.tight_layout()
plt.show()
#%%
# Reporte detallado: precision, recall y F1
print("Reporte de clasificación:\n", classification_report(Y_heldout, y_pred))
#%% Análisis de errores
error_rates = cm / cm.sum(axis=1)[:, np.newaxis]
np.fill_diagonal(error_rates, 0)  # Eliminar aciertos

for i in range(10):
    max_errors = []
    for j in range(10):
        if i != j and error_rates[i, j] > 0.01:  # Filtramos errores significativos
            max_errors.append((i, j, error_rates[i, j]))

max_errors.sort(key=lambda x: x[2], reverse=True)
print("\nPrincipales confusiones:")
for i, j, rate in max_errors[:10]:
    print(f"{clases[i]} → {clases[j]}: {rate:.2%}")

#%% Calculamos presición y recall por clase 

y_pred = best_tree.predict(X_heldout)

precisionScore = precision_score(Y_heldout, y_pred, average=None)
recallScore = recall_score(Y_heldout, y_pred, average=None)

#%% Gráfico que visualiza la precisión y recall por clase  

clases = list(range(10))  # sabemos que van del 0 al 9
grosor = 0.4

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar([c - grosor/2 for c in clases], precisionScore, grosor, label='Precisión', color='lightblue')
ax.bar([c + grosor/2 for c in clases], recallScore, grosor, label='Recall', color='blue')

ax.set_xlabel('Clases', fontsize = 19)
ax.set_ylabel('Valor alcanzado', fontsize = 19)
ax.set_title('Precision & Recall por Clase', fontsize = 20, pad = 20)
ax.set_xticks(clases)
ax.set_xticklabels(clases, fontsize=14)
ax.tick_params(axis='y', labelsize=14)
ax.grid(axis='y', linestyle='--', alpha=0.8)
ax.legend(fontsize=12, handlelength=2, handleheight=1.5)

plt.tight_layout()
plt.show()









