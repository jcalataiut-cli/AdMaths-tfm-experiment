# tfm-experiments

Experimentación numérica para TFM sobre generalización del teorema NNGP/NTK.

## Contenido

- **01_clt_convergence.py** — Verificación del Teorema Central del Límite en espacio de parámetros
- **02_kernel_comparison.py** — Comparación kernels NNGP analítico vs. empírico (FC, CNN)
- **compile_report.py** — Generador de informes LaTeX unificados
- **figuras** — Resultados visuales de los experimentos

## Hallazgo principal

CNN con peso compartido rompe convergencia GP ingenua (correlación ~0.416 vs 0.958 FC).

