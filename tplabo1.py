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
consultaSQL = """
              SELECT DISTINCT Jurisdicción AS Provincia, CAST("Código de localidad" AS VARCHAR) AS "Código de localidad" , Departamento
              FROM escuelas
              ORDER BY Provincia, Departamento;
              """

dataLocalidades = dd.sql(consultaSQL).df()
print(dataLocalidades)
#%%
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
              FROM edad_casos_deptos
              GROUP BY id_departamento, departamento
              ORDER BY id_departamento ASC;
              """

dataPoblacion = dd.sql(consultaSQL).df()
print(dataPoblacion)              
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
              SELECT DISTINCT Jurisdicción AS Provincia, "Código de localidad", Departamento,
              COUNT(Cueanexo) AS Escuelas,
              SUM(Jardines) AS Jardines,
              SUM(Primarias) AS Primarias,
              SUM(Secundarias) AS Secundarias,
              FROM escuelasCorreccionVARCHAR
              GROUP BY Provincia,"Código de localidad", Departamento
              ORDER BY Provincia, Departamento;

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
              LEFT OUTER JOIN cantidadBP1950 AS bp
              ON d.ID = bp.id
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
              SELECT Provincia, Departamento
              FROM consignaDos
              EXCEPT
              SELECT Provincia, departamento
              FROM dataProvinciasYDeptos;



"""

consignaDos = dd.sql(consultaSQL).df()
print(consignaDos)

#%%



              