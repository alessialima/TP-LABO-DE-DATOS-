#%% INTEGRANTES
# Francisco Margitic
# Alessia Lourdes Lima
# Katerina Lichtensztein
#El archivo se encuentra dividido en tres secciones principales,
#Limpieza de Datos: de las tablas originales para crear las de nuestro modelo
#Manejo de Tablas Modelo: para generar las 4 consultas solicitadas
#Graficos: utilizando las herramientas de visualizacion 
#Nuestras Tablas para descargar, las mismas se encuentran en TablasModelo
#Consultas Extra: para las métricas de GQM o el anexo
#%% IMPORTACIONES
import os
import pandas as pd
import duckdb as dd
import matplotlib.pyplot as plt
import numpy as np
#%%
#importamos las tablas originales
carpetaOriginal = os.path.dirname(os.path.abspath(__file__))
ruta_escuelas = os.path.join(carpetaOriginal,"TablasOriginales" ,"2022_padron_oficial_establecimientos_educativos.xlsx")
ruta_poblacion = os.path.join(carpetaOriginal,"TablasOriginales" , "padron_poblacion.xlsx")
ruta_bibliotecas = os.path.join(carpetaOriginal,"TablasOriginales" , "bibliotecas-populares.csv")
escuelas = pd.read_excel(ruta_escuelas, skiprows=6)
#eliminamos estas primeras filas para obtener nombres utiles de columna
poblacion = pd.read_excel(ruta_poblacion,skiprows=([i for i in range(0, 10)]+ [j for j in range(56589, 56700)]))
poblacion = poblacion.dropna(how="all")
#eliminamos las primeras y ultimas filas para facilitar lectura y ademas eliminamos
#todas las filas nulas
biblio = pd.read_csv(ruta_bibliotecas)
#%%
#Importamos nuestras correspondientes al modelo relacional
carpetaModelo = os.path.dirname(os.path.abspath(__file__))
ruta_BP = os.path.join(carpetaModelo,"TablasModelo" , "BP.csv")
ruta_DEPARTAMENTO = os.path.join(carpetaModelo,"TablasModelo" , "DEPARTAMENTO.csv")
ruta_EE = os.path.join(carpetaModelo,"TablasModelo" , "EE.csv")
ruta_MAILS = os.path.join(carpetaModelo,"TablasModelo" , "MAILS.csv")
ruta_NIVELES = os.path.join(carpetaModelo,"TablasModelo" , "NIVELES.csv")
ruta_NIVELES_EN_EE = os.path.join(carpetaModelo,"TablasModelo" , "NIVELES_EN_EE.csv")
BP = pd.read_csv(ruta_BP)
DEPARTAMENTO = pd.read_csv(ruta_DEPARTAMENTO)
EE = pd.read_csv(ruta_EE)
MAILS = pd.read_csv(ruta_MAILS)
NIVELES = pd.read_csv(ruta_NIVELES)
NIVELES_EN_EE = pd.read_csv(ruta_NIVELES_EN_EE)
#%%
#LIMPIEZA DE DATOS Y CREACION TABLAS MODELO

#Creamos la tabla temporal numeracion y usamos ROW_NUMBER para
#numerar y ordenar las filas
#Creamos la tabla temporal departamentos, la misma elimina el texto 'AREA #' y 
#los espacios en blanco de la casilla, y nos devuelve las casillas de departamento e id
#para esto nos ayudamos de la funcion replace y trim
#tales que incluyan 'AREA #' en la columna 1
#Creamos la tabla temporal datos que selecciona las filas que incluyen las edades
#y cantidad de poblacion de cada departamento de las columnas 1 y 2, con trim garantizamos
#eliminar espacios en blanco 
#Finalmente hacemos un JOIN entre departamento y datos, estos se colocan ordenados segun
#numeracion hasta el proximo departamento, o con un toque de 200 en caso de ser el ultimo
#con CAST forzamos la conversion a INTEGER
#finalmente TRANSLATE (ejemplo, '0123456789', '') remplaza los valores numericos
#de ejemplo por vacios, por lo tanto filtramos con el WHERE todas aquellas filas
#que no posean solo numeros 
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
              TRIM("Unnamed: 1") AS edades,  
              TRIM("Unnamed: 2") AS casos
              FROM numeracion)
              SELECT 
              d.id_departamento,
              d.departamento,
              CAST(dt.edades AS INTEGER) AS edad,
              CAST(dt.casos AS INTEGER) AS cantidad
              FROM departamentos AS d
              JOIN datos AS dt 
              ON dt.ordenar > d.ordenar 
              AND dt.ordenar < COALESCE(
              (SELECT MIN(ordenar) 
              FROM departamentos 
              WHERE ordenar > d.ordenar),
              d.ordenar + 200)
              WHERE TRANSLATE(dt.edades, '0123456789', '') = ''
              AND TRANSLATE(dt.casos, '0123456789', '') = ''
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
#como el código de localidad es el código de departamento con un agregado final,
#usamos la comparación LIKE indec.ID||'%' para encontrar el id depto de las localidades
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
              WHERE departamento LIKE 'Comuna%';                    
"""

dataCABA = dd.sql(consultaSQL).df()
consultaSQL1 = """
               SELECT '2000' AS id_departamento, 'CABA' AS departamento, 
               SUM(pob_jardin) AS pob_jardin, SUM(pob_primaria) AS pob_primaria,
               SUM(pob_secundaria) AS pob_secundaria, SUM(pob_total) AS pob_total
               FROM dataCABA;                   
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
              WHERE departamento NOT LIKE 'Comuna%';
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
              FROM dataCABASumada;                   
"""

dataPoblacionUnificada = dd.sql(consultaSQL).df()
print(dataPoblacionUnificada)
#%% 
#eliminamos esas 4 tuplas espureas que fueron generadas al colocarle provincia
# a los departamentos, luego obtenemos la tabla DEPARTAMENTO de nuestro Modelo Relacional
#las misas fueron halladas ya que el tamaño de dataPoblacionUnificada diferia por 4 filas
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
              OR (provincia = 'Río Negro' AND nombre_depto = 'Coronel Suárez'));                    
"""

DepartamentoConTolhuin = dd.sql(consultaSQL).df()
print(DepartamentoConTolhuin)
#%%
#en 2017 se creo el departamento de Tolhuin pero los establecimientos educativos no fueron actualizados
#tomamos la decision de mantenerlo como parte de Rio Grande, por lo que para amendar agruparemos la poblacion de ambos deptos
consultaSQL = """
              SELECT id_depto, provincia, nombre_depto, 
              pob_jardin, pob_primaria, pob_secundaria, pob_total
              FROM DepartamentoConTolhuin WHERE nombre_depto NOT IN ('Tolhuin', 'Río Grande')
              UNION ALL   
              SELECT 
              (SELECT id_depto FROM DepartamentoConTolhuin WHERE nombre_depto = 'Río Grande') AS id_depto,
              provincia, 'Río Grande' AS nombre_depto,
              SUM(pob_jardin) AS pob_jardin,
              SUM(pob_primaria) AS pob_primaria,
              SUM(pob_secundaria) AS pob_secundaria,
              SUM(pob_total) AS pob_total
              FROM DepartamentoConTolhuin
              WHERE nombre_depto IN ('Río Grande', 'Tolhuin')
              GROUP BY provincia;
"""

DEPARTAMENTO = dd.sql(consultaSQL).df()
print(DEPARTAMENTO)
#%%
#cambiamos los nulls por 0
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
#agrupamos los ee de capital federal para trabajarlo como un unico departamento
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
               FROM escuelaCABA;
"""

escuelaCABAJunta = dd.sql(consultaSQL1).df()
print(escuelaCABAJunta)
#%%
#sacamos todos los ee de capital federal para despues poder unificarlos correctamente
consultaSQL = """
              SELECT DISTINCT 
              Jurisdicción, Cueanexo, Nombre, CAST(CAST("Código de localidad" AS INT) / 1000 AS INT) AS "Código de localidad", Departamento, Jardines, Primarias, Secundarias
              FROM escuelasCorreccionVARCHAR
              WHERE Departamento NOT LIKE 'Comuna%';
"""

escuelaSINCABA = dd.sql(consultaSQL).df()
print(escuelaSINCABA)
#%%
#agrupamos ambas tablas 
consultaSQL = """
              SELECT *
              FROM escuelaSINCABA
              UNION ALL
              SELECT *
              FROM escuelaCABAJunta;
"""

escuelaJuntas = dd.sql(consultaSQL).df()
print(escuelaJuntas)
#%%
#Finalmente creamos la tabla EE de nuestro Modelo Relacional
consultaSQL = """
              SELECT Cueanexo, "Código de localidad" AS id_depto, Nombre AS nombre_EE
              FROM escuelaJuntas;
"""

EE = dd.sql(consultaSQL).df()
print(EE)
#%%
#Creamos la tabla NIVELES_EN_EE de nuestro Modelo Relacional
#esta posee el cueanexo del establecimiento y el nivel que posee como valor atomico,
#en caso de poseer los tres niveles el cueanexo estara en tres filas, una asociada a cada nivel
consultaSQL = """
              SELECT Cueanexo, CASE 
              WHEN Jardines = '1' THEN 'Nivel inicial - Jardín de infantes' 
              END AS nivel
              FROM escuelasCorreccionVARCHAR
              WHERE nivel = 'Nivel inicial - Jardín de infantes';
              
"""
NIVELESjardin = dd.sql(consultaSQL).df()
consultaSQL1 = """
              SELECT Cueanexo, CASE 
              WHEN Primarias = '1' THEN 'Primario' 
              END AS nivel
              FROM escuelasCorreccionVARCHAR
              WHERE nivel = 'Primario';
              
"""
NIVELESprimaria = dd.sql(consultaSQL1).df()
consultaSQL2 = """
              SELECT Cueanexo, CASE 
              WHEN Secundarias = '1' THEN 'Secundario' 
              END AS nivel
              FROM escuelasCorreccionVARCHAR
              WHERE nivel = 'Secundario';
              
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
                FROM NIVELESsecundaria;
"""

NIVELES_EN_EE = dd.sql(consultaSQL3).df()
print(NIVELES_EN_EE)
#%%
#forzamos conversion de fecha fundacion a tipo int para facilidad de manipulacion,
#ademas, solo tomamos el año de dicha fecha, pues es el unico valor que nos interesa
consultaProvincia = """
                    SELECT DISTINCT nro_conabip, cod_localidad, id_departamento, provincia, departamento, localidad, mail,
                    CAST(SUBSTRING(fecha_fundacion,0,5) AS INT) AS añoFundacion
                    FROM biblio;
"""

dataAñoAINT = dd.sql(consultaProvincia).df()
print(dataAñoAINT)
#%%
#creamos la tabla BP de nuestro Modelo Relacional,
#para no dejar nulls cambiamos dichos valores por 0
consultaProvincia = """
                    SELECT DISTINCT nro_conabip, (CASE
                    WHEN añoFundacion IS NULL THEN '0' ELSE añoFundacion END) AS fecha_fundacion, id_departamento AS id_depto
                    FROM dataAñoAINT;
"""

BP = dd.sql(consultaProvincia).df()
print(BP)
#%%
#Generamos la tabla MAILS de nuestro Modelo Relacional,
#cambiamos los nulls de establecimientos que no posean mail por 
# 'No posee mail'
#existe una biblioteca que tiene su mail escrito 2 veces, vamos a corregirlo para evitar
#posibles errores
consultaSQL = """
              SELECT DISTINCT nro_conabip, CASE 
              WHEN mail IS NULL THEN 'No posee mail'
              WHEN nro_conabip = 3900
              THEN  'sanestebanbibliotecapopular@yahoo.com.ar'
              ELSE mail
              END AS mail
              FROM biblio;
"""

MAILS = dd.sql(consultaSQL).df()
print(MAILS)
#%%
#Creamos una mini tabla llamada niveles de nuestro Modelo Relacional,
#la misma incluye los tres niveles a utilizar de la modalidad comun
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
#GENERACION DE CONSULTAS
#Juntamos los cuenaexos con sus respectivos departamentos
consultaSQL = """
                SELECT DISTINCT e.Cueanexo, e.id_depto, n.nivel
                FROM EE AS e
                INNER JOIN NIVELES_EN_EE AS n
                ON n.Cueanexo = e.Cueanexo;
"""

EEySusNiveles = dd.sql(consultaSQL).df()
print(EEySusNiveles)
#%%
#juntamos las tablas para obtener provincia y nombre departamento de cada colegio
#ademas de tener la columna nivel donde nos aclara todos los niveles que posee cada establecimiento
consultaSQL = """
              SELECT d.id_depto, e.Cueanexo, e.nivel, d.provincia, d.nombre_depto, d.pob_jardin, d.pob_primaria, d.pob_secundaria,
              FROM DEPARTAMENTO AS d
              INNER JOIN EEySusNiveles as e
              ON d.id_depto = e.id_depto;
"""

EEySusNivelesPD = dd.sql(consultaSQL).df()
print(EEySusNivelesPD)
#%%
# ANALISIS DE DATOS 1
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
              ORDER BY Provincia, Primarias DESC;
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
#garantizamos contar solo aquellas bp fundadas en 1950 o despues
consultaSQL = """
              SELECT id_depto, provincia AS Provincia, nombre_depto AS Departamento,
              SUM(CASE WHEN fecha_fundacion >= 1950 THEN 1 ELSE 0 END) AS "Cantidad de BP fundadas desde 1950",
              FROM BPporDepto
              GROUP BY Provincia, Departamento, id_depto
              ORDER BY Provincia, "Cantidad de BP fundadas desde 1950" DESC;
"""

cantidadBP1950 = dd.sql(consultaSQL).df()
print(cantidadBP1950)
#%%
#generamos los departamentos que no tenian ninguna biblioteca y les agregamos valor 0
#para que figuren en la tabla 
consultaSQL = """
              SELECT d.id_depto, d.provincia AS Provincia, d.nombre_depto AS Departamento,
              '0' AS "Cantidad de BP fundadas desde 1950"
              FROM DEPARTAMENTO AS d
              LEFT JOIN BPporDepto AS b
              ON d.id_depto = b.id_depto
              WHERE b.id_depto IS NULL; 
"""

departamentosSinBP = dd.sql(consultaSQL).df()
print(departamentosSinBP)
#%%
#ANALISIS DE DATOS 2
consultaSQL = """
              SELECT Provincia, Departamento,"Cantidad de BP fundadas desde 1950"
              FROM cantidadBP1950
              UNION ALL
              SELECT Provincia, Departamento,"Cantidad de BP fundadas desde 1950"
              FROM departamentosSinBP
              GROUP BY Provincia, Departamento, "Cantidad de BP fundadas desde 1950"
              ORDER BY Provincia, "Cantidad de BP fundadas desde 1950" DESC;                    
"""

Consulta2 = dd.sql(consultaSQL).df()
print(Consulta2)
#%%
#Seleccionamos los datos relevantes 
consultaSQL = """
              SELECT DISTINCT provincia, nombre_depto, Cueanexo, id_depto
              FROM EEySusNivelesPD;
"""

EEporDepto = dd.sql(consultaSQL).df()
print(EEporDepto)
#%%
#Luego sumamos cuanexos para obtener la cantidad de ee de cada departamento
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
#ANALISIS DE DATOS 3
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
              ORDER BY Cant_EE DESC, Cant_BP DESC,
              d.provincia ASC, d.nombre_depto ASC;
                    
"""

Consulta3 = dd.sql(consultaSQL).df()
print(Consulta3)
#%%
#Unimos con BPporDepto para obtener provincia y departamento
consultaSQL = """ 
              SELECT d.nro_conabip, d.provincia AS Provincia, d.nombre_depto AS Departamento, d.id_depto, m.mail
              FROM BPporDepto AS d
              JOIN MAILS AS m
              ON d.nro_conabip = m.nro_conabip 
              WHERE NOT mail = 'No posee mail';
                    
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
              FROM mailProvincia;
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
#Obtenemos el dominio mas frecuente de cada departamento, optamos por usar MAX
#para que en caso de empate devuelva el dominio de inicial mas grande
# ejemplo, google y yahoo empatan, devuelve yahoo pues y>g
consultaSQL = """
              SELECT Provincia, Departamento, id_depto, 
              MAX(Dominio) AS "Dominio más frecuente en BP"
              FROM repeticionesDominio
              GROUP BY Provincia, Departamento, id_depto;                    
"""

masFrecuente = dd.sql(consultaSQL).df()
print(masFrecuente)
#%%
#ANALISIS DE DATOS 4
#similar a la consulta 3, usamos COALESCE para agregarle dominio ninguno a 
#todos aquellos departamentos que o no tuvieran biblioteca o sus bibliotecas no tuvieran mail
consultaSQL = """
              SELECT d.provincia AS Provincia, d.nombre_depto AS Departamento, 
              COALESCE(m."Dominio más frecuente en BP", 'ninguno') AS "Dominio más frecuente en BP"
              FROM DEPARTAMENTO AS d
              LEFT JOIN masFrecuente AS m
              ON d.id_depto = m.id_depto
              ORDER BY Provincia, Departamento;
"""

Consulta4 = dd.sql(consultaSQL).df()
print(Consulta4)
#%% GRAFICOS
#Ejercicio 1 

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

#%%
# NUESTRAS TABLAS
#para descargar las tablas de nuestro Modelo Relacional en formato csv
DEPARTAMENTO.to_csv("DEPARTAMENTO.csv",index=False) 
BP.to_csv("BP.csv",index=False)
EE.to_csv("EE.csv",index=False)
MAILS.to_csv("MAILS.csv",index=False)
NIVELES.to_csv("NIVELES.csv",index=False)
NIVELES_EN_EE.to_csv("NIVELES_EN_EE.csv",index=False)
#%%
#Tablas de consultas 
Consulta1.to_csv("Consulta1.csv",index=False)
Consulta2.to_csv("Consulta2.csv",index=False)
Consulta3.to_csv("Consulta3.csv",index=False)
Consulta4.to_csv("Consulta4.csv",index=False)
#%%
#importacion de tablas consulta
carpetaConsulta = os.path.dirname(os.path.abspath(__file__))
ruta_C1 = os.path.join(carpetaModelo,"ConsultasSQL" , "Consulta1.csv")
ruta_C2 = os.path.join(carpetaModelo,"ConsultasSQL" , "Consulta2.csv")
ruta_C3 = os.path.join(carpetaModelo,"ConsultasSQL" , "Consulta3.csv")
ruta_C4 = os.path.join(carpetaModelo,"ConsultasSQL" , "Consulta4.csv")
Consulta1 = pd.read_csv(ruta_C1)
Consulta2 = pd.read_csv(ruta_C2)
Consulta3 = pd.read_csv(ruta_C3)
Consulta4 = pd.read_csv(ruta_C4)
#%%
#CONSULTAS Y TABLAS DE ANEXO
consultaSQL = """
              SELECT Provincia,
              SUM(Cant_BP) AS BP,
              SUM(Cant_EE) AS EE,
              SUM(Población) AS Población
              FROM Consulta3
              GROUP BY Provincia
              ORDER BY Población DESC;                    
"""

CantPorProvincias = dd.sql(consultaSQL).df()
consultaSQL1 = """
              SELECT Provincia, BP,
              ROUND((BP * 100000.0) / Población, 2) AS "BP cada 100k habitantes",
              EE,
              ROUND((EE * 100000.0) / Población, 2) AS "EE cada 100k habitantes",
              Población
              FROM CantPorProvincias
              ORDER BY Población DESC;                    
"""

EEyBPcada100 = dd.sql(consultaSQL1).df()
print(EEyBPcada100)
#%%
#GENERAR TABLA ANEXO
EEyBPcada100.to_csv("EEyBPcada100.csv",index=False)
#%%
#importacion tabla anexo
carpetaAnexo = os.path.dirname(os.path.abspath(__file__))
ruta_Anexo = os.path.join(carpetaAnexo,"Anexo" , "EEyBPcada100.csv")
Anexo = pd.read_csv(ruta_Anexo)
#%%

longitud = Anexo.head(24)
plt.figure(figsize=(15, 4))
ax = plt.subplot(111, frame_on=False)
ax.axis("off")
col_widths = [0.2, 1, 0.2,1,0.2,1,1,1]
color_encabezado = "#f5f5f5"
tabla = plt.table(
    cellText=longitud.values,
    colLabels=longitud.columns,
    colWidths=[0.2] * len(longitud.columns),
    cellLoc="left",
    loc="center")

for (i, nombre_columna) in enumerate(longitud.columns):
    celda_encabezado = tabla.get_celld()[(0, i)]
    celda_encabezado.set_facecolor(color_encabezado)
    celda_encabezado.set_text_props(weight='bold')  
tabla.auto_set_font_size(False)
tabla.set_fontsize(12)
tabla.scale(1, 1.5)
plt.show()
#%% Consulta extra realizada a partir de la Consulta 1 para relacionar de forma más clara la cantidad de EE y población segun el nivel y rango etario
consultaSQL = """
SELECT 
  Provincia, 
  Departamento,
  ROUND(NULLIF("Población Jardin", 0) / NULLIF(Jardines, 0), 1) AS poblacion_por_jardin,
  ROUND(NULLIF("Población Primaria", 0) / NULLIF(Primarias, 0), 1) AS poblacion_por_primaria,
  ROUND(NULLIF("Población Secundaria", 0) / NULLIF(Secundarios, 0), 1) AS poblacion_por_secundaria
FROM Consulta1
ORDER BY Provincia, poblacion_por_jardin ASC;
"""

Consulta1EXTRA = dd.sql(consultaSQL).df()
print(Consulta1EXTRA)

#%% Descargo una imagen de la tabla para agregar al anexo
longitud = Consulta1EXTRA.head(24)
plt.figure(figsize=(20, 4))
ax = plt.subplot(111, frame_on=False)
ax.axis("off")
col_widths = [0.2, 1, 0.2,1,0.2,1,1,1]
color_encabezado = "#f5f5f5"
tabla = plt.table(
    cellText=longitud.values,
    colLabels=longitud.columns,
    colWidths=[0.2] * len(longitud.columns),  # Ajusta según necesidad
    cellLoc="left",
    loc="center")

for (i, nombre_columna) in enumerate(longitud.columns):
    celda_encabezado = tabla.get_celld()[(0, i)]
    celda_encabezado.set_facecolor(color_encabezado)
    celda_encabezado.set_text_props(weight='bold')
tabla.auto_set_font_size(False)
tabla.set_fontsize(12)
tabla.scale(1, 1.5) 
plt.show()
#%%
#Gráfica de dispersión, donde podemos observar a simple viste que no hay relación alguna entre
#cantidad de bp y cantidad de ee
plt.figure(figsize=(14, 10))
ax = plt.gca()
scatter = ax.scatter(
    x=Anexo["BP cada 100k habitantes"],
    y=Anexo["EE cada 100k habitantes"],
    s=Anexo["Población"]/30000,
    c=Anexo["Población"],
    cmap="viridis",
    alpha=0.8,
    edgecolor="black",
    linewidth=0.5)

for i, row in Anexo.iterrows():
    ax.annotate(
        row["Provincia"],
        (row["BP cada 100k habitantes"], row["EE cada 100k habitantes"]),
        fontsize=9,
        alpha=0.85,
        xytext=(5, 5),
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.5))

z = np.polyfit(Anexo["BP cada 100k habitantes"], Anexo["EE cada 100k habitantes"], 1)
p = np.poly1d(z)
trendline, = ax.plot(
    Anexo["BP cada 100k habitantes"],
    p(Anexo["BP cada 100k habitantes"]),
    color="red",
    linestyle="--",
    linewidth=1.5,
    label=f'Tendencia (y = {z[0]:.2f}x + {z[1]:.2f})')

plt.title("Relación entre Bibliotecas Populares y Establecimientos Educativos\npor cada 100,000 habitantes", 
          fontsize=14, pad=20)
plt.xlabel("Bibliotecas Populares por 100k habitantes", fontsize=12)
plt.ylabel("Establecimientos Educativos por 100k habitantes", fontsize=12)
plt.grid(True, alpha=0.2)

sizes = [500000, 1000000, 5000000]
labels = [f'{s/1000000:.1f}M hab' for s in sizes]
handles = [plt.scatter([], [], s=s/30000, color="grey", edgecolor="black", alpha=0.7)
           for s in sizes]

legend = ax.legend(
    handles=handles,
    labels=labels,
    title="Tamaño = Población",
    loc="upper right",
    frameon=True,
    framealpha=0.9)

plt.xlim(0, 22)
plt.ylim(40, 250)

plt.tight_layout()
plt.show()
#%%