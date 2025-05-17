import pandas as pd
import duckdb as dd
import openpyxl as op
#%%
#elimino el index de mas que se exporto innecesariamente al crear las tablas
carpeta = "/Users/margi/OneDrive/Escritorio/tp1/TablasModelo/"
BP = pd.read_csv(carpeta+"BP.csv", index_col=[0])
DEPARTAMENTO = pd.read_csv(carpeta+"DEPARTAMENTO.csv", index_col=[0])
EE = pd.read_csv(carpeta+"EE.csv", index_col=[0])
MAILS = pd.read_csv(carpeta+"MAILS.csv", index_col=[0])
NIVELES = pd.read_csv(carpeta+"NIVELES.csv", index_col=[0])
NIVELES_EN_EE = pd.read_csv(carpeta+"NIVELES_EN_EE.csv", index_col=[0])
#%%
#en 2017 se creo el departamento de Tolhuin pero los establecimientos educativos no fueron actualizados
#tomamos la decision de mantenerlo como parte de Rio Grande, por lo que para amendar agruparemos la poblacion de ambos deptos
consultaSQL = """
              SELECT id_depto, provincia, nombre_depto, 
              pob_jardin, pob_primaria, pob_secundaria, pob_total
              FROM DEPARTAMENTO WHERE nombre_depto NOT IN ('Tolhuin', 'Río Grande')
              UNION ALL   
              SELECT 
              (SELECT id_depto FROM DEPARTAMENTO WHERE nombre_depto = 'Río Grande') AS id_depto,
              provincia, 'Río Grande' AS nombre_depto,
              SUM(pob_jardin) AS pob_jardin,
              SUM(pob_primaria) AS pob_primaria,
              SUM(pob_secundaria) AS pob_secundaria,
              SUM(pob_total) AS pob_total
              FROM DEPARTAMENTO
              WHERE nombre_depto IN ('Río Grande', 'Tolhuin')
              GROUP BY provincia;
"""

CorreccionTolhuin = dd.sql(consultaSQL).df()
print(CorreccionTolhuin)
#%%
#primero agrupo las escuelas por departamento
consultaSQL = """
              SELECT DISTINCT id_depto,
              COUNT(Cueanexo) AS Cant_EE,
              FROM EE
              GROUP BY id_depto;
              """

cantidadEEporDepto = dd.sql(consultaSQL).df()
print(cantidadEEporDepto)
#%%
consultaSQL = """
                SELECT DISTINCT e.Cueanexo, e.id_depto, n.nivel
                FROM EE AS e
                INNER JOIN NIVELES_EN_EE AS n
                ON n.Cueanexo = e.Cueanexo
                    
"""

EEySusNiveles = dd.sql(consultaSQL).df()
print(EEySusNiveles)
#%%
#juntamos las tablas para obtener provincia y nombre departamento de cada colegio
#ademas de tener la columna nivel donde nos aclara todos los niveles que posee cada establecimiento
consultaSQL = """
              SELECT d.id_depto, e.Cueanexo, e.nivel, d.provincia, d.nombre_depto, d.pob_jardin, d.pob_primaria, d.pob_secundaria,
              FROM CorreccionTolhuin AS d
              INNER JOIN EEySusNiveles as e
              ON d.id_depto = e.id_depto;
              
                    
"""

EEySusNivelesPD = dd.sql(consultaSQL).df()
print(EEySusNivelesPD)
#%%
consultaSQL = """
              SELECT provincia AS Provincia, nombre_depto AS Departamento,
              SUM(CASE WHEN nivel = 'Nivel inicial - Jardín de infantes'
                    THEN 1 ELSE 0 END) AS Jardines,
              pob_jardin AS "Población Jardin",
              SUM(CASE WHEN nivel = 'Primario'
                    THEN 1 ELSE 0 END) AS Primarias,
              pob_primaria AS "Población Primaria",
              SUM(CASE WHEN nivel = 'Secundario'
                    THEN 1 ELSE 0 END) AS Secundarios,
              pob_secundaria AS "Población Secundaria"
              FROM EEySusNivelesPD
              GROUP BY Provincia, Departamento, "Población Jardin","Población Primaria","Población Secundaria"
              ORDER BY Provincia, Departamento DESC;
              
              
                    
"""

ConsignaUno = dd.sql(consultaSQL).df()
print(ConsignaUno)

#%%
















