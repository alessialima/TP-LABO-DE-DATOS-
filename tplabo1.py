import pandas as pd
import duckdb as dd
import openpyxl as op
#%%
carpeta = "/Users/margi/OneDrive/Escritorio/TPlabo1/"
escuelas = pd.read_excel(carpeta+"2022_padron_oficial_establecimientos_educativos.xlsX",skiprows=6)
poblacion = pd.read_excel(carpeta+"padron_poblacion.xlsX", skiprows=([i for i in range(0, 10)]+ [j for j in range(56589, 56700)] ))
poblacion = poblacion.dropna(how='all')
biblio = pd.read_csv(carpeta+"bibliotecas-populares.csv")
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



              
