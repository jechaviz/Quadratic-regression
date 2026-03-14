# Quadratic Regression (Python)

Proyecto migrado a **Python puro** y optimizado para procesamiento incremental de puntos `(x, y)`.

## Características
- Arquitectura orientada a objetos con separación de responsabilidades.
- Patrón **Strategy** para desacoplar el motor de ajuste cuadrático.
- Enfoque AOP con decoradores para métricas de rendimiento.
- Código simple y mantenible siguiendo principios SOLID/DRY/KISS/YAGNI.
- Salida tabular con estas columnas:
  `x_start, x_end, a, b, c, y_calc, y_pred, slope, parabola_side`.
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
  --validation-mode avg \
  --significant-digits 6 \
  --min-points 5 \
  --prediction-horizon 1 \
  --markov-lookback 30 \
  --hurst-min-points 20 \
  --predictive-strength 0.65
```


## Uso en Jupyter Notebook
También puedes usar la versión de notebook `quadratic_regression.ipynb` para ejecutar el flujo paso a paso en Jupyter.

```bash
jupyter notebook quadratic_regression.ipynb
```
### Modos de validación
- `avg`: valida por distancia absoluta promedio (`|y_real - y_parabola| <= tolerance`).
- `envelope`: considera fallo cuando el precio queda **fuera** de la parábola (para `a>0`, debajo; para `a<0`, encima), con tolerancia.
- `inner`: considera fallo cuando el precio queda **dentro** de la parábola; útil para detectar rechazo externo.

## Pine Script para TradingView
Se añadió `tradingview_quadratic_regression.pine` para probar el concepto visualmente en TradingView con los mismos modos (`avg`, `envelope`, `inner`) y filtros de tolerancia/fallos máximos.

## Predicción y suavizado
- `y_pred` usa una proyección **predictiva** basada en una cadena de Markov de régimen (sube/baja) + exponente de Hurst (persistencia/mean-reversion).
- `predictive-strength` controla cuánto pesa el componente predictivo sobre el horizonte base.
- `markov-lookback` y `hurst-min-points` controlan la memoria estadística del predictor.
- En las visualizaciones, la tendencia alcista se pinta en azul suave y la bajista en rosa suave para lectura rápida.
- Se mantiene el enfoque incremental con estadísticas acumuladas y operaciones de frontera en tiempo constante para el ajuste.

## Nota de migración
Este repositorio ahora está centrado en Python y ya no depende de implementación Java.
