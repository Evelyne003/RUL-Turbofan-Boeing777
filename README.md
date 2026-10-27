# Estimación de vida útil restante (RUL) en motores turbofan con LSTM

**Trabajo Terminal No. 2027-A190 · ESCOM-IPN · Lic. en Ciencia de Datos**
*Prototipo basado en la aplicación de aprendizaje profundo para estimar el tiempo de vida útil restante en motores turbofan de aeronave comercial Boeing 777.*

Autores: Rojas Vasquez Evelyne, Onofre Rosales Rodrigo · Directoras: Morales Guitron Sandra Luz, López Ruiz Gabriela de Jesús.

El prototipo aplica la arquitectura LSTM de Zheng et al. (2017) sobre los conjuntos NASA C-MAPSS (FD001–FD004) siguiendo CRISP-DM, y expone los resultados en un tablero Streamlit.

## 1. Declaración de datos 
Para garantizar la trazabilidad y reproducibilidad del proyecto, se establece la siguiente declaración sobre el origen, tratamiento y limitaciones de los insumos informativos:
* **Fuente primaria:** Conjunto de datos público *Commercial Modular Aero-Propulsion System Simulation* (C-MAPSS), administrado por el *NASA Prognostics Data Repository* (Ames Research Center).
* **Estructura multivariada:** Consta de subconjuntos experimentales (`FD001` a `FD004`) que registran 21 variables de sensores (temperaturas, presiones y velocidades de rotación del compresor) bajo diferentes condiciones de operación y modos de falla simulados[cite: 7].
* **Política de privacidad y acceso:** Al tratarse de simulaciones termodinámicas públicas para investigación académica, no se vulneran normativas de propiedad industrial de aerolíneas ni restricciones de confidencialidad comercial.
* **Limitación explícita:** Los datos carecen de canales directos de acelerometría o vibración de alta frecuencia (Frederick et al., 2007; Saxena et al., 2008). Su uso se limita estrictamente a la validación algorítmica del prototipo, deslindando su aplicación directa en producción sin una fase previa de calibración con telemetría real de flotas comerciales.

---

## 2. Criterios de administración y gestión de proyectos de TI
El desarrollo del Trabajo Terminal se rige bajo estándares formales de gestión de proyectos tecnológicos y MLOps:
* **Metodología de desarrollo:** Estandarizada mediante **CRISP-DM** (*Cross Industry Standard Process for Data Mining*), estructurada en cuatro etapas ejecutables: (1) Ingesta y Preparación, (2) Modelado e Hiperparámetros, (3) Evaluación y Validación, y (4) Despliegue en Tablero.
* **Control de versiones y calidad de código:** Repositorio alojado en GitHub bajo flujos de trabajo profesionales, asegurando revisión por pares, commits descriptivos y pruebas unitarias automatizadas (`pytest` con 24 pruebas sobre componentes sintéticos para garantizar robustez sin requerir descarga pesada).
* **Gestión de riesgos técnicos:** Mitigación de desfasamientos mediante partición estricta por motor (80/20) para evitar fuga de datos (*data leakage*) y uso de validación interna independiente del conjunto de prueba final.

---

## 3. Políticas éticas y de seguridad operacional
Dado que el mantenimiento predictivo impacta directamente en industrias críticas para la vida humana (aviación comercial), el prototipo adopta un marco estricto de ética y responsabilidad algorítmica:
* **Seguridad en la aviación comercial:** El sistema está concebido como una **herramienta de apoyo a la toma de decisiones** para los ingenieros del departamento de mantenimiento aeronáutico, y no como un sistema de control autónomo o absoluto.
* **Mitigación de falsos negativos y subestimación:** En concordancia con la función de puntuación asimétrica de Saxena et al. (2008), se prioriza penalizar con mayor peso las estimaciones tardías (cuando el modelo predice una vida útil mayor a la real), previniendo fallas imprevistas en servicio.
* **Transparencia y explicabilidad:** El tablero de visualización expone no solo la predicción numérica del RUL, sino la evolución temporal de las variables críticas y alertas de umbral, permitiendo al operador auditar el comportamiento del modelo LSTM.
* **Sostenibilidad operativa:** El uso de mantenimiento predictivo contribuye a la reducción de la huella de carbono y el consumo innecesario de recursos al evitar el reemplazo prematuro de componentes aeronáuticos que aún conservan vida útil operativa.

---

## Arquitectura modular (Figura 1 del protocolo)

| Módulo del protocolo | Archivo | Responsabilidad |
|---|---|---|
| 1. Adquisición y limpieza | `src/ingestion.py`, `src/preprocessing.py` | Descarga/lectura, sensores constantes, normalización, etiquetas de RUL |
| 2. Ventanas temporales | `src/preprocessing.py` | Ventanas deslizantes, partición por motor |
| 3. Modelo LSTM | `src/model.py`, `src/train.py` | Arquitectura, Adam + ReduceLROnPlateau, entrenamiento |
| 4. Resultados y alertas | `src/evaluation.py`, `src/inference.py`, `dashboard/app.py` | RMSE, MAE, Score, gráficas, alerta de falla |

## Estructura del repositorio

```
rul-turbofan-tt/
├── data/
│   ├── raw/                 # C-MAPSS original (train_/test_/RUL_FD00X.txt) — no versionado
│   └── processed/           # artefactos intermedios — no versionado
├── notebooks/
│   ├── 01_EDA.ipynb         # análisis exploratorio (CRISP-DM 2–3)
│   └── 02_experimentacion.ipynb  # cuadrícula de hiperparámetros y evaluación (CRISP-DM 4–5)
├── src/
│   ├── config.py            # rutas, constantes de C-MAPSS, TrainConfig
│   ├── ingestion.py         # descarga y lectura validada
│   ├── preprocessing.py     # Preprocessor, RUL, ventanas, split por motor
│   ├── model.py             # LSTM apilada + callbacks
│   ├── train.py             # pipeline completo y CLI
│   ├── grid_search.py       # búsqueda en cuadrícula (validación interna)
│   ├── evaluation.py        # métricas y gráficas
│   └── inference.py         # carga de artefactos, curva de RUL, alertas
├── models/<FD00X>/          # model.keras, preprocessor.joblib, config.json, metrics.json, figures/
├── dashboard/app.py         # tablero Streamlit
├── tests/                   # pytest (24 pruebas; datos sintéticos, sin descarga)
├── Makefile · requirements*.txt · pytest.ini · ruff.toml · .gitignore
```

## Inicio rápido

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
make setup                                             # dependencias (TensorFlow ≥ 2.16)
make data                                              # descarga C-MAPSS en data/raw/
make test                                              # 24 pruebas
make train SUBSET=FD001 WINDOW=30                      # entrena y evalúa un subconjunto
make train-all                                         # FD001–FD004
make grid SUBSET=FD001                                 # búsqueda en cuadrícula
make dashboard                                         # tablero en http://localhost:8501
```

Si `make data` falla (la NASA ha cambiado la ubicación del archivo en el pasado), descargue manualmente el conjunto desde el *NASA Prognostics Data Repository* y copie los 12 archivos `train_FD00X.txt`, `test_FD00X.txt` y `RUL_FD00X.txt` en `data/raw/`; o use `python -m src.ingestion --download --url <URL_VIGENTE>`.

Sin GPU el entrenamiento es viable (series cortas, modelo pequeño). Fije `--seed` para reproducibilidad; el pipeline usa `keras.utils.set_random_seed`.

## Decisiones de diseño y su justificación

1. **RUL lineal por tramos.** `RUL(t) = ciclo_final − t`, limitada a `rul_cap` (por defecto 125; la literatura usa valores del orden de 125–130, verifique el valor de cada fuente). Supone degradación despreciable en la fase sana (Heimes, 2008). El tope se trata como hiperparámetro.
2. **Normalización min-max por régimen de operación, ajustada solo con entrenamiento.** En FD002/FD004 los seis regímenes desplazan la escala de los sensores; se agrupan con K-Means sobre los ajustes operacionales. Se evita la normalización por motor individual (ver observaciones).
3. **Selección de sensores guiada por datos.** Se descarta un sensor si su desviación estándar dentro de cada régimen es < 0.01 en todos ellos; `01_EDA.ipynb` contrasta el resultado con el conjunto de 14 sensores habitual en FD001.
4. **Ventanas deslizantes con paso 1.** Etiqueta = RUL del último ciclo de la ventana. En prueba se usa la última ventana de cada motor; las trayectorias más cortas que la ventana se rellenan replicando la primera lectura.
5. **Partición 80/20 por motor** (no por ventana): las ventanas solapadas de un mismo motor son casi idénticas, y separarlas entre entrenamiento y validación infla la métrica de validación.
6. **Arquitectura.** LSTM apiladas → capas densas ReLU → salida lineal, con *dropout* (Srivastava et al., 2014) y capa `Rescaling` por `rul_cap` para estabilidad numérica. **Tamaños por defecto (64-32 celdas, 32-16 densas, dropout 0.2) son un punto de partida**; deben contrastarse con Zheng et al. (2017) y refinarse con la cuadrícula.
7. **Optimización.** Adam, tasa inicial 1e-3 (Kingma y Ba, 2015), `ReduceLROnPlateau` (factor 0.5, paciencia 5, mínimo 1e-5) y parada temprana con restauración del mejor peso.
8. **Selección de hiperparámetros sin tocar el conjunto de prueba.** `grid_search.py` optimiza el RMSE de validación; la prueba se evalúa una vez.
9. **Métricas.** RMSE, MAE y función de puntuación de Saxena et al. (2008). Se reportan contra la RUL real sin tope (`RUL_FD00X.txt`) y también con la RUL real topada, porque la literatura no es uniforme; indique cuál usa al comparar.
10. **Alertas.** Estado NORMAL / PRECAUCIÓN / CRÍTICO según umbrales configurables (por defecto 60 y 30 ciclos). Son parámetros de política de mantenimiento, no valores de la literatura.

## Limitaciones

* Resultados sobre datos simulados; no se garantiza generalización a flotas reales.
* Los umbrales de alerta no están calibrados con costos operativos reales.

## Referencias (APA 7.ª edición)

Frederick, D. K., DeCastro, J. A. y Litt, J. S. (2007). *User's guide for the Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)* (NASA/TM—2007-215026). NASA Glenn Research Center.

Goodfellow, I., Bengio, Y. y Courville, A. (2016). *Deep learning*. MIT Press. https://www.deeplearningbook.org

Heimes, F. O. (2008). Recurrent neural networks for remaining useful life estimation. En *2008 International Conference on Prognostics and Health Management* (pp. 1–6). IEEE.

Hochreiter, S. y Schmidhuber, J. (1997). Long short-term memory. *Neural Computation, 9*(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735

Kingma, D. P. y Ba, J. (2015). Adam: A method for stochastic optimization. En *3rd International Conference on Learning Representations (ICLR 2015)*. https://arxiv.org/abs/1412.6980

Li, X., Ding, Q. y Sun, J.-Q. (2018). Remaining useful life estimation in prognostics using deep convolution neural networks. *Reliability Engineering & System Safety, 172*, 1–11. https://doi.org/10.1016/j.ress.2017.11.021

Saxena, A., Goebel, K., Simon, D. y Eklund, N. (2008). Damage propagation modeling for aircraft engine run-to-failure simulation. En *2008 International Conference on Prognostics and Health Management*. IEEE. https://doi.org/10.1109/PHM.2008.4711414

Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I. y Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research, 15*(56), 1929–1958.

Wirth, R. y Hipp, J. (2000). CRISP-DM: Towards a standard process model for data mining. En *Proceedings of the 4th International Conference on the Practical Applications of Knowledge Discovery and Data Mining* (pp. 29–39).

Zheng, S., Ristovski, K., Farahat, A. y Gupta, C. (2017). Long short-term memory network for remaining useful life estimation. En *2017 IEEE International Conference on Prognostic and Health Management (ICPHM)* (pp. 88–95). IEEE.
