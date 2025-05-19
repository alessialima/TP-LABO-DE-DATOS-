# Primer métrica: 

CantFechaFundacionNull = BP["fecha_fundacion"].isna().sum()
CantFechaFundacionTotal = len(BP)

print((CantFechaFundacionNull/CantFechaFundacionTotal)*100)

# Segunda métrica:

biblio.sort_values(by='localidad', ascending = True)
escuelas.sort_values(by='Localidad', ascending = True)

comparamosLocalidad = escuelas['Localidad'].equals(biblio['localidad']) 

comparamosCodigos = escuelas['Código de localidad'].equals(biblio['cod_localidad']) 

print(comparamosCodigos) 
