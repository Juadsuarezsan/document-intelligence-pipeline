# Muestra de FUNSD

Cinco archivos de anotación del conjunto FUNSD (*Form Understanding in Noisy
Scanned Documents*, Jaume, Ekenel y Thiran, 2019). El conjunto completo tiene
199 formularios (149 de entrenamiento, 50 de prueba), pesa ~17 MB con imágenes
y se distribuye en https://guillaumejaume.github.io/FUNSD/dataset.zip bajo
una licencia de investigación no comercial.

Aquí solo se incluyen las anotaciones (sin imágenes). Se usan para:

- construir pares pregunta → respuesta reales para la evaluación
  (`src/synth/funsd.py`, 281 pares en total), y
- poblar la galería de la demo con formularios reales.

| Archivo | Entidades | Pares Q→A |
|---|---|---|
| `0000971160.json` | 24 | 8 |
| `0000989556.json` | 100 | 42 |
| `0000990274.json` | 31 | 13 |
| `0000999294.json` | 181 | 195 |
| `0001118259.json` | 31 | 23 |

## Esquema

```json
{
  "form": [
    {
      "id": 0,
      "text": "Date:",
      "box": [482, 268, 518, 282],
      "linking": [[3, 12]],
      "label": "question",
      "words": [{"text": "Date:", "box": [482, 268, 518, 282]}]
    }
  ]
}
```

Etiquetas: `header`, `question`, `answer`, `other`. `linking` enlaza
pregunta con respuesta por `id`.

## Conjunto completo

```bash
python scripts/download_data.py --dataset funsd   # descarga, verifica SHA-256 y extrae en data/raw/funsd
```

Ningún número del repositorio proviene del conjunto completo: la corrida
registrada en `eval/RESULTS.md` usa únicamente estas cinco anotaciones más
los documentos sintéticos.
