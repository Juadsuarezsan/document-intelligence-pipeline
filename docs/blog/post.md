# Reglas primero, modelo después: cómo construí un pipeline de Document Intelligence que sabe cuándo no sabe

*Borrador para Medium / Dev.to. ~1 900 palabras.*

Hay una escena que se repite en cualquier equipo de cuentas por pagar, en
cualquier aseguradora, en cualquier hospital: alguien abre un PDF, busca con
los ojos el número de factura, la fecha, el total, y los teclea en otro
sistema. Multiplíquelo por miles de documentos al mes y por la variedad de
formatos que llegan de cientos de proveedores. Ese trabajo es el que las
empresas de *Intelligent Document Processing* (Hyperscience, Rossum, Klarity)
venden como producto, y es el problema que quise resolver de punta a punta en
mi portafolio: un sistema que recibe un PDF, lo lee por la ruta que
corresponda, extrae un JSON validado y, sobre todo, **decide con honestidad
si el resultado se puede aprobar solo**.

Este artículo cuenta las decisiones de diseño, lo que medí sin gastar un solo
token y lo que aprendí de los errores.

## El problema no es extraer, es saber cuándo la extracción es fiable

Un modelo de lenguaje con visión extrae campos de una factura con una
facilidad que hace cinco años parecía ciencia ficción. Pero en producción la
pregunta no es "¿puede el modelo leer esto?" sino "¿qué hago con los 30 000
documentos del mes?". Si todos van a un revisor humano, el sistema no ahorra
nada. Si todos se aprueban solos, el primer total mal leído termina en una
transferencia equivocada.

La arquitectura que implementé sigue el patrón que usan los productos
comerciales:

1. **Parsing por capas.** Un PDF con capa de texto se lee con `pdfplumber` en
   18 ms. Un escaneo se rasteriza y pasa por Tesseract (0,7 s por página). Un
   documento que ni el OCR resuelve va a Claude Vision.
2. **Clasificación** en cinco tipos (factura, recibo, contrato, formulario,
   reporte), porque cada tipo tiene su esquema Pydantic de campos.
3. **Extracción guiada por esquema**: el modelo (o las reglas) devuelve
   exactamente los campos definidos, con valores normalizados.
4. **Validación cruzada**: campos requeridos, fechas ISO, `subtotal + IVA =
   total`, y el dígito de verificación del NIT colombiano con el algoritmo
   módulo 11 de la DIAN, no con un regex.
5. **Confianza y enrutamiento**: ≥ 0,90 se aprueba solo, ≥ 0,70 va a
   revisión humana, por debajo se reprocesa con el modelo de visión.

Todo esto vive en un `StateGraph` de LangGraph con seis nodos y una arista
condicional. Cada nodo registra su entrada, su salida y su duración bajo el
`trace_id` de la petición; la respuesta HTTP devuelve tokens, costo en
dólares y el registro de nodos.

## La decisión que lo cambió todo: un extractor de reglas que no es un stub

La tentación en un proyecto así es escribir un `if not api_key: return []` y
dejar que el modelo haga todo. Decidí lo contrario: el extractor heurístico
(`HeuristicExtractor`) es un componente real. Conoce ocho formas de escribir
"número de factura" en inglés y español, normaliza `1.234,56` y `1,234.56`,
entiende cinco formatos de fecha, lee cláusulas de contrato en prosa ("the
initial term of this Agreement is 24 months") y parte líneas con varias celdas
como las que produce un OCR ("To:  Mr. Smith  Date:  9/3/92").

Tres razones:

- **Es la primera fila de la tabla comparativa.** El spec pedía comparar
  "solo parsing" contra "parsing + fallback con visión" contra "visión
  directa". Sin una línea base real no hay contra qué comparar.
- **Es el primer paso del pipeline real, no un plan B.** Con reglas, el 7 %
  de mis documentos sintéticos se aprueba solo y el 62 % va a revisión con
  campos ya rellenados. Solo el 33 % necesita al modelo. Eso es dinero: a 30
  000 documentos al mes, usar Claude en todos cuesta unos 390 USD; usarlo solo
  donde las reglas fallan, unos 130.
- **Hace la evaluación reproducible.** `python -m eval.run` corre en un
  contenedor sin red, sin GPU y sin llave, y produce los mismos números cada
  vez.

## Medir sin llave: datos sintéticos con semilla y cinco formularios reales

No tenía llave de API en el entorno de desarrollo y la red bloqueaba la
descarga de FUNSD y DocVQA. Podía dejar la evaluación como "pendiente" o
construir algo medible. Hice lo segundo, con dos reglas: **todo lo sintético
se declara como tal** y **ningún número se publica si no sale de un archivo
versionado en `eval/runs/`**.

El generador (`src/synth/documents.py`) produce 100 documentos de cinco tipos
con 15 plantillas de layout. Las primeras plantillas eran "fáciles": listas
de `Etiqueta: valor`. El extractor las resolvía al 100 %, y eso me pareció
sospechoso más que satisfactorio. Así que añadí variantes adversas: facturas
escritas en prosa ("Invoice X was issued by Y on March 14, 2026…"), contratos
en forma de carta, memos con `RE:` y `FROM:`, y formularios con etiquetas
separadas por espacios en lugar de dos puntos, que es exactamente lo que
produce un OCR sobre un formulario real. A eso sumé las cinco anotaciones de
FUNSD que sí tenía, convertidas a texto en orden de lectura con sus 281 pares
pregunta → respuesta como verdad de terreno.

Cada documento se evalúa de tres maneras: como texto ya extraído, como PDF
con capa de texto (generado con reportlab) y como PDF escaneado (el mismo
PDF rasterizado y reincrustado como imagen, sin texto, para que el pipeline
tome la ruta OCR de verdad). Y 30 tablas, con y sin líneas, para medir la
extracción de tablas a nivel de celda.

## Lo que salió

Fila heurística, modo PDF: **Field F1 82,9 %, Doc Accuracy 71,0 %, Table F1
96,7 %**, 18 ms por documento, 0 USD. Por modo de entrada: 83,4 % en texto,
82,9 % en PDF, 81,0 % en escaneado. El clasificador acierta el 99,1 %.

Los números por plantilla son lo interesante. Todo lo que tiene forma de
`Etiqueta: valor` sale al 100 %. Las facturas en prosa caen al 18 %, las
cartas de contrato al 44 %, los formularios sin dos puntos al 0 %. Y en los
formularios reales de FUNSD el recall de pares es del 8 %: las respuestas
están en otra línea, en casillas, o son varias por pregunta.

Eso no es un fracaso del sistema, es la señal correcta. El scorer envía
precisamente esos documentos a `vlm_fallback`. La precisión, en cambio, es
alta en todos los tipos (≥ 95 %): cuando las reglas extraen algo, casi
siempre es correcto. El problema es recall, y el recall es lo que el modelo
de visión debe cerrar.

Las dos filas con Claude quedan como "pendiente (requiere
ANTHROPIC_API_KEY)". El código está: cliente sobre el SDK `anthropic` con
timeout, backoff exponencial solo en 429/5xx y accounting de tokens; bloques
`image` en base64 para la transcripción y la extracción por visión; el juez
LLM con una rúbrica de cinco criterios numerados. Todo probado con
respuestas simuladas, incluido el camino en que el modelo devuelve basura y el
pipeline conserva los campos anteriores en lugar de romperse.

## Tres errores que me enseñaron algo

**El regex de respaldo del total capturaba el subtotal.** En las facturas en
prosa, "the amount of USD 1,760.55 (subtotal 1,455.00 plus tax…)" hacía que
`total[^\d]{0,25}(\d…)` tomara el primer número después de *subtotal*. Siete
falsos positivos, todos del mismo bug. Un regex que "funciona" en las
plantillas fáciles es una deuda que solo aparece con datos adversos.

**Una etiqueta de dos letras rompía los reportes.** `re` (de "RE: asunto")
como etiqueta de título hacía prefijo con "report date" y llenaba el título
con una fecha. La solución fue un orden estricto: coincidencia exacta con
cualquier etiqueta antes que cualquier prefijo, prefijos solo con etiquetas
de tres o más caracteres, y nunca sobre una clave que sea la etiqueta exacta
de otro campo (`tax` nunca hace prefijo con `tax id`).

**El generador inyectaba errores donde no debía.** Para probar el validador
aritmético, el 20 % de las facturas tiene un total alterado a propósito. La
primera versión reemplazaba el string del total en todas las líneas, y cuando
el IVA era 0 % el subtotal (idéntico al total) también cambiaba. El ground
truth decía una cosa y el texto otra. Se detectó porque la evaluación
señalaba un subtotal "mal extraído" que en realidad estaba bien leído. La
verdad de terreno también tiene bugs, y solo la encuentra una evaluación que
se mira caso por caso.

## Lo que haría con una llave

Correr `make eval` para completar la tabla; publicar 30 trazas en LangSmith
(el grafo ya está instrumentado por variables de entorno); y comparar dos
estrategias de reprocesamiento: extraer de nuevo desde las imágenes o
transcribir con visión y volver a pasar las reglas. Mi hipótesis es que la
segunda es más barata y casi igual de buena para facturas, y peor para
formularios como los de FUNSD.

Después, datos reales: los 50 formularios de prueba de FUNSD para reportar el
F1 de pares oficial, y DocVQA para document-level accuracy. Y una mejora que
no necesita modelo: extraer pares por proximidad geométrica usando las cajas
del OCR en lugar de solo por línea, que es la causa principal del recall del
8 %.

## Cierre

El valor de un pipeline de documentos no está en el modelo más caro sino en
la fontanería alrededor: en saber qué ruta tomar, en validar con reglas de
negocio, en calibrar una confianza que refleje cobertura y no solo promedio,
y en medir con datos que expongan las debilidades en lugar de esconderlas.
El repositorio está en
[github.com/Juadsuarezsan/document-intelligence-pipeline](https://github.com/Juadsuarezsan/document-intelligence-pipeline):
`make install && make test && make eval` reproduce todo lo que acabo de
contar, sin llave.
