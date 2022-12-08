# OpenAlexGroup - Filtro de artículos asociados a palabra clave de institución

Esta aplicación permite consultar los artículos asociados a una institución basándose en una palabra clave, la cual es ingresada como un parámetro de URL.

# Ejemplo de uso

Ingresar por url: http://127.0.0.1:8000/ devolverá una lista de todos los artículos que estén asociados a una institución en la base de datos.

Ingresar parámetro por URL: http://127.0.0.1:8000/?affiliation=nombre%20de%20institución devolverá todos los artículos donde al menos uno de los autores esté asociado a una institución que en su nombre lleve el valor del parámetro 'affiliation'.

El parámetro 'affiliation' puede ser una palabra clave, como facom. No es case-sensitive, por lo que http://127.0.0.1:8000/?affiliation=facom equivale a http://127.0.0.1:8000/?affiliation=FACom
