
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 

#%% Ejercicio 1 

# En primer lugar, armamos un dataframe con los datos que obtenemos de Consulta3. Queremos saber la cant_BP por provincia de manera decreciente
consultaSQL = """
              SELECT Provincia,
              COUNT(Cant_BP) AS Cant_BP
              FROM Consulta3
              GROUP BY Provincia
              ORDER BY Cant_BP DESC;
              """

cantBPProv = dd.sql(consultaSQL).df()

print(cantBPProv)

# A partir de ahí armamos un gráfico de barras donde x = cada provincia y height = cantidad de bp que posee cada uno

fig, ax = plt.subplots()

ax.bar(data=cantBPProv,x="Provincia",height="Cant_BP", color='#660033') # es el código ese de html por si quieren cambiar o buscar uno !! elegí violeta

ax.set_title('Cantidad de Bibliotecas Populares por Provincia') # si querés ponerle titulo es aqui

ax.set_xlabel('PROVINCIAS', fontsize = '13', labelpad=8) # labelpad cambia el espacio entre "provincias" y las prov en si !!!!! 
ax.set_ylabel('CANTIDAD DE BP', fontsize = '13', labelpad=8) # fontsize es el tamaño de la letra 

ax.set_ylim(0,567) 
 
ax.set_yticks([]) # saca los numeritos del eje y 
ax.bar_label(ax.containers[0],fontsize=8)

plt.xticks(rotation = 45, ha = "right")
plt.tight_layout()
plt.figure(figsize=(12,6))

ax.spines[["right", "top", "left"]].set_visible(False) # esto saca el rectangulo 



#%% Ejercicio 2 


fig, ax = plt.subplots()

width = 400  # elegir un numero que funque
bins = np.arange((Consulta1["Población Jardin"]).min(), (Consulta1["Población Jardin"]).max()+width, width) # auxilio
# me arma un rango para los bins 

# cuenta que datos se meten en cada bin
counts, bins = np.histogram(Consulta1['Jardines'], bins = bins)

#centro el bin 
center = (bins[:-1]+bins[1:]) /2 


ax.bar(x=center,
       height=counts,
       align="center",
       color="skyblue",
       edgecolor="black")

#titulos ahre
ax.set_title("titulo")
ax.set_xlabel("equis")
ax.set_ylabel("y")

# anota el rango en x

labels = [f'({int(bins[i])},{int(bins[i+1])}]'
          for i in range(len(bins)-1)]

ax.set_xticks(center[::10])
ax.set_xticklabels(labels[::10], rotation=90,fontsize=12)
ax.tick_params(axis="x", length=6,width=2)


plt.tight_layout()
plt.show()

#%% EJERCICIO 2
import matplotlib.pyplot as plt
import numpy as np
# Configuración de colores y transparencia
color_jardin = '#FF8C00'
color_primaria = '#20B2AA'
color_secundaria = '#FF00FF'
transparencia = 0.3 #para poder visualizar las zonas con superposiciones

# Posiciones fijas para cada nivel en el eje X, con jitter aleatorio, esto permite ver todos los valores del gráfico por más de que se superpongan.
# Por ejemplo, si tenemos 3 deptos. con 200 jardines,  estos 3 se dibujan en posiciones como (0.1, 200), (-0.05, 200), (0.15, 200) para que no se vean como un único punto
np.random.seed(0)
x_jardin = np.random.normal(0, 0.2, size=len(Consulta1))
x_primaria = np.random.normal(1, 0.2, size=len(Consulta1))
x_secundaria = np.random.normal(2, 0.2, size=len(Consulta1))

# Crear figura
fig, ax = plt.subplots(figsize=(10, 10))
# Scatter por nivel, el tamaño del punto representa la cantidad de población de manera proporcional
ax.scatter(x_jardin, Consulta1['Jardines'], 
            s=Consulta1['Población Jardin'] / 300, 
            alpha=transparencia, color=color_jardin, label='Jardín')

ax.scatter(x_primaria, Consulta1['Primarias'], 
            s=Consulta1['Población Primaria'] / 300, 
            alpha=transparencia, color=color_primaria, label='Primaria')

ax.scatter(x_secundaria, Consulta1['Secundarios'], 
            s=Consulta1['Población Secundaria'] / 300, 
            alpha=transparencia, color=color_secundaria, label='Secundaria')

# Etiquetas y estética
ax.set_xticks([0, 1, 2], ['Jardín', 'Primaria', 'Secundaria'])
ax.set_ylabel('Cantidad de Establecimientos')
ax.set_xlabel('Nivel Educativo')
ax.set_title('Establecimientos Educativos y Población por Departamento')
ax.set_ylim(bottom=0)
plt.tight_layout()

# Mostrar el gráfico
plt.show()

