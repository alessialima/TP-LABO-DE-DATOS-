import pandas as pd
import duckdb as dd
import openpyxl as op
#%%
carpeta = "/Users/margi/OneDrive/Escritorio/TPlabo1/"
ee = pd.read_excel(carpeta+"2022_padron_oficial_establecimientos_educativos.xlsX",skiprows=6)
poblacion = pd.read_excel(carpeta+"padron_poblacion.xlsX", skiprows=10)
poblacion = poblacion.dropna(how='all')
bp = pd.read_csv(carpeta+"bibliotecas-populares.csv")
#%%
###departamento
consultaDepto = """
                SELECT DISTINCT 
                CASE 
                WHEN TRIM(REPLACE("Unnamed: 1", 'AREA #', '')) LIKE '0%'
                THEN SUBSTR(TRIM(REPLACE("Unnamed: 1", 'AREA #', '')), 2) 
                ELSE TRIM(REPLACE("Unnamed: 1", 'AREA #', '')) 
                END AS id_departamento,
                "Unnamed: 2" AS departamento
                FROM poblacion
                WHERE "Unnamed: 1" LIKE 'AREA #%'
                ORDER BY departamento;
"""

dataframeDepartamento = dd.sql(consultaDepto).df()
print(dataframeDepartamento)
#%%
consultaEdadCasos = """
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

edad_casos_deptos = dd.sql(consultaEdadCasos).df()
print(edad_casos_deptos)
#%%
consultaProvincia = """
                    SELECT DISTINCT
                    id_provincia, provincia
                    FROM bp
                    ORDER BY provincia;
"""

dataframeProvincia = dd.sql(consultaProvincia).df()
print(dataframeProvincia)

#%%        
consultaSQL = """
              SELECT DISTINCT 'Nivel inicial - Jardín de infantes' AS Nivel FROM ee
              UNION ALL
              SELECT DISTINCT 'Primario' FROM ee
              UNION ALL 
              SELECT DISTINCT 'Secundario' FROM ee;
"""

dataframeNivel = dd.sql(consultaSQL).df()
print(dataframeNivel)
#%%
consultaSQL = """
              SELECT DISTINCT id_departamento, nombre, mail, fecha_fundacion, nro_conabip
              FROM bp
              ORDER BY id_departamento;
"""

dataframebp = dd.sql(consultaSQL).df()
print(dataframebp)
#%%

consultaSQL = """
              SELECT DISTINCT 
              Cueanexo, Nombre, Mail
              FROM ee;

              """

dataframeee = dd.sql(consultaSQL).df()
print(dataframeee)
#%%


















