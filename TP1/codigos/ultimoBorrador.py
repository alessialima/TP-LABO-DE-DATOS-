#%% INTEGRANTES
# Francisco Margitic
# Alessia Lourdes Lima
# Katerina Lichtensztein
# Descripción del contenido y otros datos que consideremos relevantes: *completar*
#cosas a chequear: que se entienda y este minimamente comentada cada seccion,
#que las variables y tablas tengan nombres representativos
#que los nombres y atributos de nuestras tablas sean iguales al modelo relacional


#%% IMPORTACIONES
import pandas as pd
import duckdb as dd
import matplotlib.pyplot as plt
import numpy as np
#%%
carpeta = r"C:\Users\katel\Escritorio\Downloads\TODO tp labo\TablasOriginales"
escuelas = pd.read_excel(carpeta+"/2022_padron_oficial_establecimientos_educativos.xlsX",skiprows=6)
poblacion = pd.read_excel(carpeta+"/padron_poblacion.xlsX", skiprows=([i for i in range(0, 10)]+ [j for j in range(56589, 56700)] ))
poblacion = poblacion.dropna(how='all')
biblio = pd.read_csv(carpeta+"/bibliotecas-populares.csv")
#%%
consultaSQL = """
              WITH numeracion AS (SELECT *,  
              ROW_NUMBER() OVER () AS ordenar
              FROM poblacion),
              departamentos AS (SELECT 
              TRIM(REPLACE("Unnamed: 1", 'AREA #', '')) AS id_departamento,
              "Unnamed: 2" AS departamento, ordenar
              FROM numeracion
              WHERE "Unnamed: 1" LIKE 'AREA #%'),
              datos AS (SELECT ordenar,
              TRIM("Unnamed: 1") AS edad_str,  
              TRIM("Unnamed: 2") AS casos_str
              FROM numeracion)
              SELECT 
              d.id_departamento,
              d.departamento,
              CAST(dt.edad_str AS INTEGER) AS edad,
              CAST(dt.casos_str AS INTEGER) AS cantidad
              FROM departamentos d
              JOIN datos dt 
              ON dt.ordenar > d.ordenar 
              AND dt.ordenar < COALESCE(
              (SELECT MIN(ordenar) 
              FROM departamentos 
              WHERE ordenar > d.ordenar),
              d.ordenar + 200)
              WHERE 
              dt.edad_str ~ '^\\d+$' 
              AND dt.casos_str ~ '^\\d+$'
              ORDER BY d.id_departamento, edad;
              """

dataEdadesYDeptos = dd.sql(consultaSQL).df()
print(dataEdadesYDeptos)
#%%
# recuento de todas las localidades, nos servira para mas adelante unificar las localidades por departamento
#fuerzo conversion a varchar
consultaSQL = """
              SELECT DISTINCT Jurisdicción AS Provincia, CAST("Código de localidad" AS VARCHAR) AS "Código de localidad" , Departamento
              FROM escuelas
              ORDER BY Provincia, Departamento;
              """

dataLocalidades = dd.sql(consultaSQL).df()
print(dataLocalidades)
#%%
# elimino el 0 inicial que pueden tener algunos ids de departamentos
consultaSQL = """
              SELECT DISTINCT 
              CASE WHEN id_departamento LIKE '0%'
              THEN SUBSTR(id_departamento, 2)
              ELSE id_departamento
              END AS ID, 
              departamento
              FROM dataEdadesYDeptos;
              """
              
dataDepartamentosINDEC = dd.sql(consultaSQL).df()
print(dataDepartamentosINDEC)  
#%%
#tomamos a capital federal como un solo departamento de id 2000, eliminamos las comunas
#corregimos error de id que poseia ushuaia y rio grande en padron
consultaSQL = """
              SELECT DISTINCT loc.Provincia, indec.ID, indec.departamento
              FROM dataDepartamentosINDEC AS indec
              INNER JOIN dataLocalidades AS loc
              ON loc."Código de localidad" LIKE indec.ID||'%';
              """
              
dataProvinciasYDeptos1 = dd.sql(consultaSQL).df()
consultaSQL1 = """
              SELECT *
              FROM dataProvinciasYDeptos1
              WHERE Provincia NOT IN ('Ciudad de Buenos Aires');
              """
              
dataProvinciasYDeptos2 = dd.sql(consultaSQL1).df()
faltantes = pd.DataFrame({"Provincia": ["Ciudad de Buenos Aires","Tierra del Fuego","Tierra del Fuego"],"ID": ["2000","94014","94007"],"departamento":["CABA","Ushuaia","Río Grande"]})
dataProvinciasYDeptos = pd.concat([dataProvinciasYDeptos2,faltantes],ignore_index=True)
print(dataProvinciasYDeptos) 
#%%            
#asumimos estos rangos para las edades escolares
consultaSQL = """
              SELECT 
              id_departamento, departamento, 
              SUM(CASE 
              WHEN edad BETWEEN 3 AND 5 THEN cantidad 
              ELSE 0 END) AS pob_jardin,
              SUM(CASE 
              WHEN edad BETWEEN 6 AND 11 THEN cantidad 
              ELSE 0 END) AS pob_primaria,
              SUM(CASE 
              WHEN edad BETWEEN 12 AND 17 THEN cantidad 
              ELSE 0 END) AS pob_secundaria,
              SUM(cantidad) AS pob_total
              FROM dataEdadesYDeptos
              GROUP BY id_departamento, departamento
              ORDER BY id_departamento ASC;
              """

dataPoblacion = dd.sql(consultaSQL).df()
print(dataPoblacion)              
#%%
#juntamos todas las comunas de capital para crear departamento caba
#acorde a como trabajaremos la ciudad
consultaSQL = """ 
              SELECT *
              FROM dataPoblacion
              WHERE departamento LIKE 'Comuna%'
                    
"""

dataCABA = dd.sql(consultaSQL).df()
consultaSQL1 = """
                SELECT '2000' AS id_departamento, 'CABA' AS departamento, 
                SUM(pob_jardin) AS pob_jardin, SUM(pob_primaria) AS pob_primaria,
                SUM(pob_secundaria) AS pob_secundaria, SUM(pob_total) AS pob_total
                FROM dataCABA
                    
"""

dataCABASumada = dd.sql(consultaSQL1).df()
print(dataCABASumada)
#%%
#corregimos el error de id de usuahia y rio grande y eliminamos 0 iniciales en caso de haber
consultaSQL = """
              SELECT
              CASE 
              WHEN departamento = 'Ushuaia' THEN '94014'
              WHEN departamento = 'Río Grande' THEN '94007'
              WHEN id_departamento LIKE '0%' THEN SUBSTR(id_departamento, 2)
              ELSE id_departamento
              END AS id_departamento, departamento, pob_jardin, 
              pob_primaria, pob_secundaria, pob_total
              FROM dataPoblacion
              WHERE departamento NOT LIKE 'Comuna%'
              

                    
"""

dataSinCABA = dd.sql(consultaSQL).df()
print(dataSinCABA)
#%%
#agrupamos todo
consultaSQL = """
              SELECT *
              FROM dataSinCABA
              UNION ALL
              SELECT *
              FROM dataCABASumada
                    
"""

dataPoblacionUnificada = dd.sql(consultaSQL).df()
print(dataPoblacionUnificada)
#%% 
#eliminamos esas 4 tuplas espureas que fueron generadas al colocarle provincia
# a los departamentos
consultaSQL = """
              SELECT pd.ID AS id_depto, pd.Provincia AS provincia, 
              du.departamento AS nombre_depto,du.pob_jardin, 
              du.pob_primaria, du.pob_secundaria, du.pob_total
              FROM dataPoblacionUnificada AS du
              INNER JOIN dataProvinciasYDeptos pd
              ON du.id_departamento = pd.ID 
              WHERE NOT ((provincia = 'Salta' AND nombre_depto = 'Pellegrini')
              OR (provincia = 'Salta' AND nombre_depto = 'Pehuajó')
              OR (provincia = 'Salta' AND nombre_depto = 'Patagones')
              OR (provincia = 'Río Negro' AND nombre_depto = 'Coronel Suárez'))
                    
"""

DEPARTAMENTO = dd.sql(consultaSQL).df()
print(DEPARTAMENTO)
#%%
consultaSQL = """
              SELECT DISTINCT 
              Jurisdicción, Cueanexo, Nombre, "Código de localidad", Departamento,
              CAST(CASE WHEN "Nivel inicial - Jardín de infantes" = '1'
              THEN 1 ELSE 0 END AS INT) AS Jardines,
              CAST(CASE WHEN "Primario" = '1'
              THEN 1 ELSE 0 END AS INT) AS Primarias,
              CAST(CASE WHEN "Secundario" = '1'
              THEN 1 ELSE 0 END AS INT) AS Secundarias
              FROM escuelas;

              """

escuelasCorreccionVARCHAR = dd.sql(consultaSQL).df()
print(escuelasCorreccionVARCHAR)
#%%
consultaSQL = """
              SELECT DISTINCT 
              Jurisdicción, Cueanexo, Nombre, "Código de localidad", Departamento, Jardines, Primarias, Secundarias
              FROM escuelasCorreccionVARCHAR
              WHERE Departamento LIKE 'Comuna%';
"""
escuelaCABA = dd.sql(consultaSQL).df()
consultaSQL1 = """
               SELECT DISTINCT 
               Jurisdicción, Cueanexo, Nombre, '2000' AS "Código de localidad", 'CABA' AS Departamento, Jardines, Primarias, Secundarias
               FROM escuelaCABA
"""


escuelaCABAJunta = dd.sql(consultaSQL1).df()
print(escuelaCABAJunta)
#%%
consultaSQL = """
              SELECT DISTINCT 
              Jurisdicción, Cueanexo, Nombre, CAST(CAST("Código de localidad" AS INT) / 1000 AS INT) AS "Código de localidad", Departamento, Jardines, Primarias, Secundarias
              FROM escuelasCorreccionVARCHAR
              WHERE Departamento NOT LIKE 'Comuna%';



"""


escuelaSINCABA = dd.sql(consultaSQL).df()
print(escuelaSINCABA)
#%%
consultaSQL = """
              SELECT *
              FROM escuelaSINCABA
              UNION ALL
              SELECT *
              FROM escuelaCABAJunta
"""
escuelaJuntas = dd.sql(consultaSQL).df()
print(escuelaJuntas)
#%%
consultaSQL = """
              SELECT Cueanexo, "Código de localidad" AS id_depto, Nombre
              FROM escuelaJuntas
"""
EE = dd.sql(consultaSQL).df()
print(EE)
#%%
consultaSQL = """
              SELECT Cueanexo, CASE 
              WHEN Jardines = '1' THEN 'Nivel inicial - Jardín de infantes' 
              END AS nivel
              FROM escuelasCorreccionVARCHAR
              WHERE nivel = 'Nivel inicial - Jardín de infantes'
              
"""
NIVELESjardin = dd.sql(consultaSQL).df()
consultaSQL1 = """
              SELECT Cueanexo, CASE 
              WHEN Primarias = '1' THEN 'Primario' 
              END AS nivel
              FROM escuelasCorreccionVARCHAR
              WHERE nivel = 'Primario'
              
"""
NIVELESprimaria = dd.sql(consultaSQL1).df()
consultaSQL2 = """
              SELECT Cueanexo, CASE 
              WHEN Secundarias = '1' THEN 'Secundario' 
              END AS nivel
              FROM escuelasCorreccionVARCHAR
              WHERE nivel = 'Secundario'
              
"""
NIVELESsecundaria = dd.sql(consultaSQL2).df()
consultaSQL3 = """
                SELECT *
                FROM NIVELESjardin
                UNION ALL
                SELECT *
                FROM NIVELESprimaria
                UNION ALL
                SELECT *
                FROM NIVELESsecundaria
"""
NIVELES_EN_EE = dd.sql(consultaSQL3).df()
print(NIVELES_EN_EE)

#%%
consultaSQL = """
              SELECT DISTINCT Jurisdicción AS Provincia, "Código de localidad", Departamento,
              COUNT(Cueanexo) AS Escuelas,
              SUM(Jardines) AS Jardines,
              SUM(Primarias) AS Primarias,
              SUM(Secundarias) AS Secundarias,
              FROM escuelasCorreccionVARCHAR
              GROUP BY Provincia,"Código de localidad", Departamento
              ORDER BY Provincia, Departamento

              """

cantidadLocalidadEE = dd.sql(consultaSQL).df()
print(cantidadLocalidadEE)
#%%
consultaProvincia = """
                    SELECT DISTINCT nro_conabip, cod_localidad, id_departamento, provincia, departamento, localidad, mail,
                    CAST(SUBSTRING(fecha_fundacion,0,5) AS INT) AS añoFundacion
                    FROM biblio;
"""

dataAñoAINT = dd.sql(consultaProvincia).df()
print(dataAñoAINT)
#%%
#para no dejar nulls cambiamos dichos valores por 0
consultaProvincia = """
                    SELECT DISTINCT nro_conabip, (CASE
                    WHEN añoFundacion IS NULL THEN '0' ELSE añoFundacion END) AS fecha_fundacion, id_departamento AS id_depto
                    FROM dataAñoAINT;
"""

BP = dd.sql(consultaProvincia).df()
print(BP)
#%%
consultaSQL = """
              SELECT id_departamento AS id, provincia AS Provincia, departamento AS Departamento,
              COUNT(CASE WHEN añoFundacion >= 1950 THEN 1 END) AS "Cantidad de BP fundadas desde 1950",
              FROM dataAñoAINT
              GROUP BY Provincia, Departamento, id
              ORDER BY Provincia, "Cantidad de BP fundadas desde 1950" DESC;
              """

cantidadBP1950 = dd.sql(consultaSQL).df()
print(cantidadBP1950)
#%%
consultaSQL = """
              SELECT DISTINCT d.Provincia AS Provincia, d.departamento AS Departamento, bp."Cantidad de BP fundadas desde 1950" AS "Cantidad de BP fundadas desde 1950"
              FROM dataProvinciasYDeptos AS d
              LEFT JOIN cantidadBP1950 AS bp
              ON d.ID = bp.id AND TRIM(LOWER(d.departamento)) = TRIM(LOWER(bp.Departamento))
              """

consignaDos1 = dd.sql(consultaSQL).df()
consultaSQL1 = """
              SELECT DISTINCT Provincia, Departamento,
              CAST(CASE WHEN "Cantidad de BP fundadas desde 1950" IS NULL 
                   THEN 0 ELSE "Cantidad de BP fundadas desde 1950" END AS INT) AS "Cantidad de BP fundadas desde 1950"
              FROM consignaDos1
              GROUP BY Provincia, Departamento, "Cantidad de BP fundadas desde 1950"
              ORDER BY Provincia, "Cantidad de BP fundadas desde 1950" DESC;
              """

consignaDos = dd.sql(consultaSQL1).df()
print(consignaDos)


#%%
consultaSQL = """
              SELECT departamento
              FROM dataProvinciasYDeptos
              EXCEPT
              SELECT Departamento
              FROM consignaDos;

"""

consignaDos = dd.sql(consultaSQL).df()
print(consignaDos)

#%%
consultaSQL = """
              SELECT DISTINCT nro_conabip, CASE 
              WHEN mail IS NULL THEN 'No posee mail' ELSE mail
              END AS mail
              FROM biblio
              WHERE nro_conabip = 3900



"""

mails = dd.sql(consultaSQL).df()
print(mails)

#%%
consultaSQL = """
              SELECT DISTINCT 'Nivel inicial - Jardín de infantes' AS Nivel FROM escuelas
              UNION ALL
              SELECT DISTINCT 'Primario' FROM escuelas
              UNION ALL 
              SELECT DISTINCT 'Secundario' FROM escuelas;
"""

NIVELES = dd.sql(consultaSQL).df()
print(NIVELES)
#%%
DEPARTAMENTO.to_csv("DEPARTAMENTO.csv",index=False) 
BP.to_csv("DEPARTAMENTO.csv",index=False)
EE.to_csv("DEPARTAMENTO.csv",index=False)
mails.to_csv("DEPARTAMENTO.csv",index=False)
NIVELES.to_csv("DEPARTAMENTO.csv",index=False)
NIVELES_EN_EE.to_csv("DEPARTAMENTO.csv",index=False)

#%%
#elimino el index de mas que se exporto innecesariamente al crear las tablas
carpeta = r"C:\Users\katel\Escritorio\Downloads\TODO tp labo\TablasModelo"
BP = pd.read_csv(carpeta+"/BP.csv", index_col=[0])
DEPARTAMENTO = pd.read_csv(carpeta+"/DEPARTAMENTO.csv", index_col=[0])
EE = pd.read_csv(carpeta+"/EE.csv", index_col=[0])
MAILS = pd.read_csv(carpeta+"/MAILS.csv", index_col=[0])
NIVELES = pd.read_csv(carpeta+"/NIVELES.csv", index_col=[0])
NIVELES_EN_EE = pd.read_csv(carpeta+"/NIVELES_EN_EE.csv", index_col=[0])
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
#%% GRAFICOS
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

ax.bar_label(ax.containers[0],fontsize=8)

plt.xticks(rotation = 45, ha = "right")
plt.tight_layout()


#%% EJERCICIO 2
# Configuración de colores y transparencia
color_jardin = '#ae75e4'
color_primaria = '#75e4ab'
color_secundaria = '#ffb35a'
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
# Ordenamos provincias según la mediana de Cant_EE
ordenMediana = Consulta3.groupby("Provincia")["Cant_EE"].median().sort_values().index

# Preparamos datos para cada provincia en el orden deseado
data_ordenada = [Consulta3.loc[Consulta3["Provincia"] == prov, "Cant_EE"].values for prov in ordenMediana]

# Crear figura
fig, ax = plt.subplots(figsize=(14,7))

# Dibujar boxplot con matplotlib (los labels se pasan con la lista de provincias ordenadas)
bp = ax.boxplot(data_ordenada, patch_artist=True)
# 1. Boxes (caja rellena)
for patch in bp['boxes']:
    patch.set_facecolor("#8dd18a")    # color relleno
    patch.set_edgecolor("grey")      # borde caja
    patch.set_linewidth(1)          # grosor borde

# 2. Median lines (línea mediana)
for median in bp['medians']:
    median.set(color='grey', linewidth=1)

# 3. Whiskers (bigotes)
for whisker in bp['whiskers']:
    whisker.set(color='grey', linewidth=1)

# 4. Caps (líneas de tope de los bigotes)
for cap in bp['caps']:
    cap.set(color='grey', linewidth=1)

# 5. Fliers (outliers, puntos)
for flier in bp['fliers']:
    flier.set(marker='o', color='grey', alpha=0.6)
# Etiquetas y estética
ax.set_title("Cantidad de Establecimientos Educativos por cada departamento de las Provincias", pad=30)

ax.set_xlabel('')
ax.set_ylabel('Cantidad de Establecimientos Educativos', labelpad=20)
ax.set_ylim(0,567)

# Poner etiquetas en el eje x con rotación
ax.set_xticklabels(ordenMediana, rotation=45, ha='right')

plt.show()


#%% Grafico 4 version entera
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









