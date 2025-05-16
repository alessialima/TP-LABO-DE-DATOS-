#%% Ejercicio 2 

# podriamos hacer el mismo grafico en barras pero separando cada nivel por color
# colores primarios(? pero verde en lugar de amarillo pq no se ve

#%% Ejercicio 3

"""
iii) Realizar un boxplot por cada provincia, de la cantidad de EE por cada
departamento de la provincia. Mostrar todos los boxplots en una misma
figura, ordenados por la mediana de cada provincia.
"""

# empiezo pensando como seria 1 solo boxplot para 1 sola provincia 

chau = pd.DataFrame({'Provincia': ['Buenos Aires', 'CABA', 'Catamarca', 'Chaco', 'Buenos Aires', 'Buenos Aires', 'Catamarca'], 'Departamentos': ['D1', 'D1', 'D1', 'D2', 'D2', 'D3', 'D2'], 'Cant_EE': [10, 23, 2, 23, 33, 11, 4]})

fig, ax = plt.subplots()

ax = sns.boxplot(x="Provincia", y="Cant_EE", hue="Departamentos", data=chau, color="#CCCC00")

ax.set_title('')

ax.set_xlabel('PROVINCIA')
ax.set_ylabel('CANT_EE')
ax.set_ylim(0,12) # lo mismo que puse en el del ej 1 aca ahre
ax.legend(title='DEPARTAMENTOS')
#%% Ejercicio 4 
"""
iv) Relación entre la cantidad de BP cada mil habitantes y de EE cada mil
habitantes por departamento.
Importante: En el informe, todos los reportes y gráficos deben ser acompañados por texto
explicativo de lo observado en ellos y con las reflexiones que puedan desarrollar.

"""

