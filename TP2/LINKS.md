informe: https://docs.google.com/document/d/1pNZVzNHwQR6B3ZaX80kaiz_KHs8x11hNQX-i1oNlWIM/edit?usp=sharing

info sobre el database: https://www.kaggle.com/datasets/zalando-research/fashionmnist

## 
* cada fila es una imagen
* label = clase (última columna)
* unnamed 0 = index 
* el resto son pixeles
* valor oscuridad del 1 al 255 
##
✅ Atributos recomendados (útiles para distinguir clases 0 vs. 8)
Cantidad de píxeles no negros
 → Las carteras suelen ser más compactas y ocupan menos superficie que las remeras.


Altura y ancho del bounding box del objeto (no negro)
 → Las remeras suelen ser más verticales y anchas, mientras que muchas carteras son más cuadradas o apaisadas.


Relación de aspecto (alto / ancho)
 → Las remeras suelen tener proporciones más estables, mientras que las carteras varían más.


Centro de masa (centroide)
 → El centroide de la remera puede estar más alto (debido al cuello) que el de una cartera.


Histograma de intensidades
 → Las carteras pueden tener más contraste o más píxeles blancos por los brillos del material. Las remeras suelen ser más homogéneas.


Número de regiones conectadas (conectividad de píxeles blancos)
 → Puede ayudar a detectar los tirantes o correas de las carteras.


Perfil vertical u horizontal de intensidad
 → Las remeras tienen “hombros”, las carteras suelen tener una base más plana o simétrica.


Proporción de píxeles blancos cerca del borde superior (correas)
 → Algunas carteras tienen tiras finas en la parte de arriba que podrían ser capturadas así.



❌ Atributos poco útiles o descartables
Píxeles del borde (esquinas)
 → Suelen ser negros en casi todas las imágenes.


Posición absoluta del objeto
 → No es confiable porque las prendas pueden estar ligeramente desplazadas.
