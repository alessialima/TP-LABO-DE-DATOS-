
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

ax.bar(data=cantBPProv,x="Provincia",height="Cant_BP", color='#660033') 

ax.set_title('Cantidad de Bibliotecas Populares por Provincia') 

ax.set_xlabel('PROVINCIAS', fontsize = '13', labelpad=20)  
ax.set_ylabel('CANTIDAD DE BP', fontsize = '13', labelpad=20) 

ax.set_ylim(0,30) # esto cambia el limite de y creo 

ax.set_yticks([]) # saca los numeritos del eje y 
ax.bar_label(ax.containers[0],fontsize=8) # pone los numeritos sobre cada columna 


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
          
