# Quadratic Regression (Python)

Proyecto migrado a **Python puro** y optimizado para procesamiento incremental de puntos `(x, y)`.

## Características
- Arquitectura orientada a objetos con separación de responsabilidades.
- Patrón **Strategy** para desacoplar el motor de ajuste cuadrático.
- Enfoque AOP con decoradores para métricas de rendimiento.
- Código simple y mantenible siguiendo principios SOLID/DRY/KISS/YAGNI.
- Salida tabular con estas columnas:
  `x_start, x_end, a, b, c, y_calc, slope, parabola_side`.
- Gráfica generada con **matplotlib** (PNG), evitando Highcharts.

## Requisitos
```bash
python3 -m pip install -r requirements.txt
```

## Uso
```bash
python3 main.py \
  --input input.txt \
  --output output.txt \
  --plot-output regression_plot.png \
  --tolerance 0.12 \
  --max-failed-predictions 4 \
  --significant-digits 6 \
  --min-points 5
```

## Nota de migración
Este repositorio ahora está centrado en Python y ya no depende de implementación Java.
