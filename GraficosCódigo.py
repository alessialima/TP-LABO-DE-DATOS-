
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
ax.spines[['right','top','left']].set_visible(False) # saca rayitas que quedan feas
ax.bar_label(ax.containers[0], fontsize=8) # pone los numeritos arriba de las columnas
#%% 

"""
LOS OTROS DOS HAY QUE PENSARLOS PRIMERO COMO SORONGO ORGANIZAR LA INFO EN PAPEL PQ IS A LOT !!!!!! 
"""
