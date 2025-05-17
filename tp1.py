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

Consulta1 = dd.sql(consultaSQL).df()
print(Consulta1)

#%%
#agrupo las bibliotecas con su respectivo departamento y provincia
consultaSQL = """
              SELECT DISTINCT d.provincia, d.nombre_depto,
              b.nro_conabip, b.fecha_fundacion, b.id_depto
              FROM BP as b
              INNER JOIN DEPARTAMENTO AS d
              ON d.id_depto = b.id_depto;
              """

BPporDepto = dd.sql(consultaSQL).df()
print(BPporDepto)
#%%
consultaSQL = """
              SELECT id_depto, provincia AS Provincia, nombre_depto AS Departamento,
              SUM(CASE WHEN fecha_fundacion >= 1950 THEN 1 ELSE 0 END) AS "Cantidad de BP fundadas desde 1950",
              FROM BPporDepto
              GROUP BY Provincia, Departamento, id_depto
              ORDER BY Provincia, "Cantidad de BP fundadas desde 1950" DESC;
              """

ConsignaDosAux = dd.sql(consultaSQL).df()
print(ConsignaDosAux)
#%%
#generamos los departamentos que no tenian ninguna biblioteca y les agregamos valor 0
#para que figuren en la tabla 
consultaSQL = """
              SELECT d.id_depto, d.provincia AS Provincia, d.nombre_depto AS Departamento,
              '0' AS "Cantidad de BP fundadas desde 1950"
              FROM DEPARTAMENTO AS d
              LEFT JOIN BPporDepto AS b
              ON d.id_depto = b.id_depto
              WHERE b.id_depto IS NULL AND NOT Departamento = 'Tolhuin'; 
"""

ConsignaDosAux2 = dd.sql(consultaSQL).df()
print(ConsignaDosAux2)
#%%
consultaSQL = """
              SELECT Provincia, Departamento,"Cantidad de BP fundadas desde 1950"
              FROM ConsignaDosAux
              UNION ALL
              SELECT Provincia, Departamento,"Cantidad de BP fundadas desde 1950"
              FROM ConsignaDosAux2
              GROUP BY Provincia, Departamento, "Cantidad de BP fundadas desde 1950"
              ORDER BY Provincia, "Cantidad de BP fundadas desde 1950" DESC;;
                    
"""

Consulta2 = dd.sql(consultaSQL).df()
print(Consulta2)
#%%
#primero agrupo las escuelas por departamento
consultaSQL = """
              SELECT DISTINCT provincia, nombre_depto, Cueanexo, id_depto
              FROM EEySusNivelesPD;
              """

EEporDepto = dd.sql(consultaSQL).df()
print(EEporDepto)

#%%
#primero agrupo las escuelas por departamento
consultaSQL = """
              SELECT provincia AS Provincia, nombre_depto AS Departamento, id_depto AS ID,
              COUNT(Cueanexo) As Cant_EE
              FROM EEporDepto
              GROUP BY Provincia, Departamento, ID;
              """

cantEEDepto = dd.sql(consultaSQL).df()
print(cantEEDepto)
#%%
#luego agrupo las bibliotecas por departamento
consultaSQL = """
              SELECT provincia AS Provincia, nombre_depto AS Departamento, id_depto AS ID,
              COUNT(nro_conabip) AS Cant_BP
              FROM BPporDepto
              GROUP BY Provincia, Departamento, ID;
              """

cantBPDepto = dd.sql(consultaSQL).df()
print(cantBPDepto)
#%%
#Con Coalesce garantizamos que aquellos valores nulls tengan un 0
#ejemplo, departamentos sin bibliotecas que no figuren en cantBPDepto
consultaSQL = """
              SELECT d.provincia AS Provincia, d.nombre_depto AS Departamento,
              COALESCE(b.Cant_BP, 0) AS Cant_BP,
              COALESCE(e.Cant_EE, 0) AS Cant_EE,
              d.pob_total AS Población
              FROM DEPARTAMENTO AS d
              LEFT JOIN cantBPDepto AS b 
              ON d.id_depto = b.ID
              LEFT JOIN cantEEDepto e
              ON d.id_depto = e.ID
              WHERE NOT d.nombre_depto = 'Tolhuin'
              ORDER BY Cant_EE DESC, Cant_BP DESC,
              d.provincia ASC, d.nombre_depto ASC;
                    
"""

Consulta3 = dd.sql(consultaSQL).df()
print(Consulta3)
#%%
#existe una biblioteca que tiene su mail escrito 2 veces, vamos a corregirlo para evitar
#posibles errores
#no tenemos en cuenta las bibliotecas sin mail por tanto las quitamos
consultaSQL = """ 
              SELECT nro_conabip,
              (CASE WHEN nro_conabip = 3900
              THEN  'sanestebanbibliotecapopular@yahoo.com.ar'
              ELSE mail END) AS mail
              FROM MAILS
              WHERE NOT mail = 'No posee mail';
                    
"""

mailCorreccion = dd.sql(consultaSQL).df()
print(mailCorreccion)
#%%
#Unimos con BPporDepto para obtener provincia y departamento
consultaSQL = """ 
              SELECT d.nro_conabip, d.provincia AS Provincia, d.nombre_depto AS Departamento, d.id_depto, m.mail
              FROM BPporDepto AS d
              JOIN mailCorreccion AS m
              ON d.nro_conabip = m.nro_conabip;
                    
"""

mailProvincia = dd.sql(consultaSQL).df()
print(mailProvincia)
#%%
# creamos un substring que comience en la posicion siguiente a @
#INSTR nos devuelve la posicion del @,
#el siguiente instr nos devuelve la posicion del primer punto, le restamos 1 para no contarlo
#De esta manera garantizamos obtener el valor entre @ y el primer punto, el dominio de mail
consultaSQL = """
              SELECT Provincia, Departamento, id_depto, nro_conabip,
              SUBSTR(mail, INSTR(mail, '@') + 1, 
              INSTR(SUBSTR(mail, INSTR(mail, '@') + 1), '.') - 1) AS Dominio
              FROM mailProvincia
             
"""
dominios = dd.sql(consultaSQL).df()
print(dominios)
#%%
#Creamos una tabla que nos indique la cantidad de repeticiones de cada dominio,
#logramos esto usando subqueries para comparacion
consultaSQL = """
              SELECT d1.Provincia, d1.Departamento, d1.id_depto,d1.Dominio, d1.Repeticiones
              FROM (SELECT Provincia, Departamento, id_depto, Dominio,
                    COUNT(Dominio) AS Repeticiones
                    FROM dominios
                    GROUP BY Provincia, Departamento, id_depto, Dominio) d1
              WHERE d1.Repeticiones = 
              (SELECT MAX(d2.Repeticiones)
              FROM (SELECT Provincia, Departamento,
              id_depto, Dominio, COUNT(Dominio) AS Repeticiones
              FROM dominios
              GROUP BY Provincia, Departamento, id_depto, Dominio) d2
              WHERE 
              d2.Provincia = d1.Provincia AND
              d2.Departamento = d1.Departamento AND
              d2.id_depto = d1.id_depto)
              ORDER BY d1.Provincia, d1.Departamento;
                    
"""

repeticionesDominio = dd.sql(consultaSQL).df()
print(repeticionesDominio)
#%%
consultaSQL = """
              SELECT Provincia, Departamento, id_depto, 
              MAX(Dominio) AS "Dominio más frecuente en BP"
              FROM repeticionesDominio
              GROUP BY Provincia, Departamento, id_depto;
                    
"""

masFrecuente = dd.sql(consultaSQL).df()
print(masFrecuente)
#%%
#similar a la consulta 4, usamos COALESCE para agregarle dominio ninguno a 
#todos aquellos departamentos que o no tuvieran biblioteca o sus bibliotecas no tuvieran mail
consultaSQL = """
              SELECT d.provincia AS Provincia, d.nombre_depto AS Departamento, 
              COALESCE(m."Dominio más frecuente en BP", 'ninguno') AS "Dominio más frecuente en BP"
              FROM DEPARTAMENTO AS d
              LEFT JOIN masFrecuente AS m
              ON d.id_depto = m.id_depto
              WHERE NOT d.nombre_depto = 'Tolhuin'
              ORDER BY Provincia, Departamento;
              
                    
"""

Consulta4 = dd.sql(consultaSQL).df()
print(Consulta4)








