
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


#necesito, ayudandome con el ej 1, armar un dataframe Ejercicio3(pob_jardin, cant_EE_jardin, pob_primaria, cant_EE_primaria, pob_secu, cant_EE_secu)
# entonces será más sencillo armar esto que sigue: 
# a continuacion van a ver un borrador de lo que puede servir despues. claramente es cualq cosa pq estoy usando un dataframe que existe solo en mi mente 

# puse colores que no se si quedan bien aesthetic juntos veremos cuando ejecute el codigo mas tarde 

fig, ax = plt.subplots()

#grafico jardin 
ax.bar(nombre_grafico['grupo etario de jardin'], nombre_grafico['Cant_EE jardin'], label='Nivel Inicial - Jardín de Infantes', color = "#ffd54f")

# grafico primaria
ax.bar(nombre_grafio['grupo etario de primaria'], nombre_grafico['Cant_EE primaria'], label= 'Primaria', color = "#d41111")

#grafico secundaria
ax.bar(nombre_grafico['grupo etario de secundaria'],nombre_grafico['Cant_EE secundaria'], label='Secundaria', color = "#112bb2")

ax.set_title('titulooo')
ax.set_xlabel('Grupo Etario')
ax.set_ylabel('Cant_EE')


ax.set_xlim(0,xmaximo)
ax.set_ylim(0,ymaximo)
ax.set_xticks(range(1,11,1))


plt.legend()
ax.spines[['right','top','left']].set_visible(False) # saca rayitas que quedan feas
ax.bar_label(ax.containers[0], fontsize=8) # pone los numeritos arriba de las columnas
#%% 


