
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 

#%% Ejercicio 1 

"i) Cantidad de BP por provincia. Mostrarlos ordenados de manera decreciente por dicha cantidad."

# es un dataframe de ejemplo que usé para practicar 
hola = pd.DataFrame({'Provincias': ['Buenos Aires','CABA','Catamarca','Chaco','Chubut'],'Cantidad': [40, 32, 23, 17, 3]
})

# a partir del dataframe del ejercicio iii) podemos primero acá ordenar de manera DECRECIENTE la cantidad de BP por provincia
# luego, gráfico: 

fig, ax = plt.subplots()

ax.bar(data=hola,x='Provincias',height='Cantidad', color='#660033') # es el código ese de html por si quieren cambiar o buscar uno !! elegí violeta

ax.set_title('tituloooo') # si querés ponerle titulo es aqui

ax.set_xlabel('PROVINCIAS', fontsize = '13', labelpad=19) # labelpad cambia el espacio entre "provincias" y las prov en si !!!!! 
ax.set_ylabel('CANTIDAD DE BP', fontsize = '13') # fontsize es el tamaño de la letra 


ax.set_ylim(0,50) # donde puse 50 pone la cantidad maxima + 1 de cant_bp (es el y maximo)

plt.tight_layout() # creo que ajusta los espacios entre columnas(????? segun google qcy)
ax.set_yticks([]) # saca los numeritos del eje y 

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
          
