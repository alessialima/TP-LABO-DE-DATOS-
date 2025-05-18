
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 

#%% Ejercicio 1 

# En primer lugar, armamos un dataframe con los datos que obtenemos de Consulta3
# Queremos saber la cant_BP por provincia de manera decreciente

consultaSQL = """
              SELECT Provincia,
              SUM(Cant_BP) AS Cant_BP
              FROM Consulta3
              GROUP BY Provincia
              ORDER BY Cant_BP DESC;
              """

cantBPProv = dd.sql(consultaSQL).df()

print(cantBPProv)

# A partir de ahí armamos un gráfico de barras donde x = cada provincia y height = cantidad de bp que posee cada uno

fig, ax = plt.subplots(figsize=(14,7)) 

ax.bar(data=cantBPProv,x="Provincia",height="Cant_BP", color='#908ad1') 

#Etiuetas y estética 
ax.set_title('Cantidad de Bibliotecas Populares por Provincia') 

ax.set_xlabel('', fontsize = '13', labelpad=8) 
ax.set_ylabel('CANTIDAD DE BP', fontsize = '13', labelpad=8)

ax.set_ylim(0,567) 
 
ax.set_yticks([]) 
ax.bar_label(ax.containers[0],fontsize=8)

plt.xticks(rotation = 45, ha = "right")
plt.tight_layout()


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

#%% Ejercicio 3 

#Crear figura 
fig, ax = plt.subplots(figsize=(14,7))

#Buscamos ordenar los bloxplot de las provincias según la mediana 
ordenMediana = Consulta3.groupby("Provincia")["Cant_EE"].median().sort_values().index

sns.boxplot(x="Provincia", y="Cant_EE", data=Consulta3, ax = ax, color = "#8dd18a", order = ordenMediana) 

#Estética y etiquetas
ax.set_title("Cantidad de Establecimientos Educativos por cada departamento de las Provincias", pad = 30)

ax.set_xlabel('')
ax.set_ylabel('Cantidad de Establecimientos Educativos', labelpad=20) 
ax.set_ylim(0,567)


plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
plt.show()

#%% Grafico 4 version entera
import matplotlib.pyplot as plt
# Calcular BP y EE por mil habitantes sin modificar el DataFrame
bp_por_mil = (Consulta3["Cant_BP"] / Consulta3["Población"]) * 1000
ee_por_mil = (Consulta3["Cant_EE"] / Consulta3["Población"]) * 1000
# Crear el gráfico de dispersión
plt.figure(figsize=(10, 10))
plt.scatter(bp_por_mil, ee_por_mil, color="#ae75e4", alpha=0.6)
# Configurar el gráfico
plt.xlabel("Cantidad de BP cada mil habitantes")
plt.ylabel("Cantidad de EE cada mil habitantes")
plt.title("Relación entre BP y EE por cada mil habitantes por Departamento")
plt.grid(True)
plt.xlim(0,2.5)
plt.ylim(0,13)
plt.tight_layout()
plt.show()
#%% Grafico 4 version recortada, un "zoom"
# Calcular BP y EE por mil habitantes sin modificar el DataFrame
bp_por_mil = (Consulta3["Cant_BP"] / Consulta3["Población"]) * 1000
ee_por_mil = (Consulta3["Cant_EE"] / Consulta3["Población"]) * 1000
# Crear el gráfico de dispersión
plt.figure(figsize=(10, 10))
plt.scatter(bp_por_mil, ee_por_mil, color="#ae75e4", alpha=0.6)
# Configurar el gráfico
plt.xlabel("Cantidad de BP cada mil habitantes")
plt.ylabel("Cantidad de EE cada mil habitantes")
plt.title("Relación entre BP y EE por cada mil habitantes por Departamento")
plt.grid(True)
plt.xlim(0,0.6)
plt.ylim(0,10)
plt.tight_layout()
plt.show()
