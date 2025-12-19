
# Proyecto Final: Análisis del Sentimiento en Criptomonedas y su Relación con el Tipo de Cambio

### Curso: **Introducción a Machine Learning con Python**
### Grupo: **N° 8**

## Integrantes
- **Luis Ángel Alejandro Arrieta Feria**
- **Mirelli Thais Jimenez Pulache**
- **Néstor Julio Rivero Escobar**

---

# 1. Introducción

El objetivo central de este proyecto es evaluar si existe una relación significativa entre la evolución del sentimiento en el mercado de criptomonedas —medido mediante el **Fear & Greed Index (FGI)**— y el comportamiento del **tipo de cambio USD/PEN**.

La investigación busca determinar si factores financieros *no tradicionales* y expectativas globales en mercados digitales pueden servir como indicadores de volatilidad para analistas y hacedores de política monetaria.

La motivación surge de la creciente interconexión entre mercados financieros globales. Aunque tradicionalmente el tipo de cambio responde a factores macroeconómicos, es plausible que indicadores alternativos relacionados al **sentimiento global de riesgo** afecten indirectamente economías emergentes como la peruana.

Se construyó un dataset consolidando series del criptomercado, indicadores globales y el tipo de cambio USD/PEN. A lo largo de los Trabajos 1–4 se aplicaron herramientas de estadística, series de tiempo, machine learning y análisis causal.

---

# TRABAJO 1 – Principales Resultados

## 2. Importación y Preparación del Dataset

Se construyó un dataset consolidado que integra series de tiempo de diversas fuentes, asegurando la homogeneidad temporal y la corrección de inconsistencias originales. El dataset final abarca un rango de fechas desde el **01 de junio de 2018** hasta el **30 de junio de 2025**.

Variables seleccionadas para el estudio:

- **FGI (Fear & Greed Index):** Sentimiento del mercado cripto  
- **USD/PEN (Venta):** Tipo de cambio local  
- **DXY:** Índice del dólar global  
- **VIX:** Índice de volatilidad y termómetro del miedo global  
- **BTC/USD:** Precio del Bitcoin  
- **Commodities y Tasas:** Precio del oro, T-Bills 13w y Treasury 5y  

Estas variables permiten desarrollar:

- Modelos ARX
- Modelos ML
- Análisis causal con DAGs
- Estadística exploratoria

---

## 3. Análisis Exploratorio (EDA)

El análisis descriptivo permite identificar patrones de estabilidad, episodios de volatilidad y posibles comovimientos entre activos.

### 3.1 Evolución Histórica y estabilidad

Se graficaron USD/PEN, FGI, BTC, Oro y VIX.  

Hallazgos:

- El tipo de cambio muestra tendencia suave.
- BTC es altamente volátil.
- FGI presenta episodios de "miedo extremo".
- El VIX sube en periodos de estrés global.
- **USD/PEN:**  
  El sol peruano mostró una marcada depreciación entre 2018 y 2021 debido a la incertidumbre política interna y el impacto del COVID-19, alcanzando picos cercanos a **S/ 4.10**. Posteriormente, ingresó en una fase de estabilización gracias a la intervención del Banco Central de Reserva del Perú (BCRP).
- **Bitcoin vs. Oro:**  
  Ambos activos muestran una correlación visual positiva desde 2020, actuando como reservas de valor ante la liquidez global. Sin embargo, el Bitcoin exhibe movimientos mucho más erráticos y explosivos en comparación con la trayectoria suave del oro.

### 3.2 Histogramas de Retornos

- **USD/PEN:**  
  Sus retornos diarios presentan una curva estrecha con un pico pronunciado en cero, lo que refleja una baja volatilidad estructural y un mercado cambiario predecible sujeto a un marco institucional sólido.

- **Bitcoin:**  
  Posee una distribución de retornos mucho más ancha con *colas gordas* (*fat tails*). Sus movimientos diarios suelen oscilar entre **±5%**, evidenciando su naturaleza especulativa y la ausencia de una autoridad que estabilice su demanda.

### 3.3 Gráficos de Dispersión

- Validación de Hipótesis:
  La correlación de Pearson entre el FGI y el tipo de cambio USD/PEN es de **0.07**, confirmando la ausencia de una relación lineal significativa entre el sentimiento
  cripto y la moneda peruana. Por el contrario, Bitcoin muestra una correlación moderada de **0.52** con el USD/PEN, influenciada principalmente por el periodo de coincidencia alcista de 2020–2021.

### 3.4 Mapa de Calor de Correlaciones

- FGI y USD/PEN → correlación baja.
- Oro correlaciona con VIX y DXY.
- BTC correlaciona moderadamente con riesgo global.

### 3.5 Bimodalidad del FGI

El FGI no sigue una distribución uniforme; se observan dos picos en los extremos (*Miedo* y *Avaricia*). El mercado pasa más tiempo en estados de *Fear* y *Extreme Fear* (más de 1200 días acumulados) que en estados de neutralidad.

---

# TRABAJO 2 – Principales Resultados

## 4. Modelo Dinámico Autoregresivo con Variable Exógena (ARX)

Previo a la selección final, se evaluaron múltiples especificaciones, incluyendo modelos multivariados con mayor número de controles macroeconómicos y distintas estructuras temporales. Sin embargo, tras comparar la capacidad predictiva fuera de la muestra, el presente modelo demostró el mejor desempeño, superando a las versiones más complejas (que presentaron problemas de sobreajuste) y a las más simples. Por tanto, se seleccionó esta especificación por su robustez y minimización del error.

Se implementó un modelo dinámico para evaluar si el retorno del tipo de cambio (\(ret\_USD\)) puede ser explicado por su propia inercia y por el sentimiento global (FGI) como shock externo.

### Especificación del Modelo

ret_USD_t = α + β1 · ret_USD_{t-1} + β2 · FGI_{t-1} + ε_t

### 4.1 Resultados del Modelado

El modelo fue ajustado utilizando **Mínimos Cuadrados Ordinarios (OLS)** con errores robustos (**HAC**) para corregir la autocorrelación.

- **Inercia del Mercado (Memoria):**  
  El coeficiente de \(ret\_USD_{t-1}\) resultó positivo y altamente significativo (**p = 0.002**). Esto indica que el tipo de cambio peruano presenta persistencia: si el dólar sube hoy, existe una probabilidad estadística de que mantenga esa tendencia al día siguiente.

- **Influencia del Sentimiento:**  
  El coeficiente del \(FGI_{t-1}\) no fue estadísticamente significativo (**p = 0.623**). Esto refuerza la conclusión de que las condiciones de miedo o avaricia en el mercado cripto no tienen un efecto inmediato sobre la variación diaria del sol peruano.

- **Métrica de Error:**  
  El modelo alcanzó un **Error Cuadrático Medio (MSE)** de **0.092062** en el conjunto de prueba, superando el desempeño de modelos lineales simples previos.
  
---

# TRABAJO 3 – Principales Resultados del Modelado Avanzado

## 5. Modelos Avanzados de Regresión

En esta sección se implementaron modelos de aprendizaje automático con mayor capacidad predictiva para evaluar si el retorno del tipo de cambio (\(ret\_USD\)) puede ser anticipado mediante el uso de variables rezagadas y factores exógenos. Se comparó el desempeño de algoritmos lineales regularizados (**Ridge**) contra modelos de ensamble no lineales (**Random Forest** y **XGBoost**).

---

## 5.1 Feature Engineering y Preparación del Dataset

Para capturar la dinámica temporal del mercado, se amplió el dataset original generando características adicionales basadas en rezagos temporales:

- **Rezagos calculados:**  
  Se generaron retardos de orden \(t-1\) (memoria inmediata), \(t-2\) (persistencia de corto plazo) y \(t-7\) (dinámica semanal).

- **Variables rezagadas:**  
  Se aplicó este procedimiento al tipo de cambio (USD/PEN), precio de Bitcoin (BTC), sentimiento de mercado (FGI) y variables macroeconómicas globales (DXY, VIX, Oro, T-Bills y Treasury 5y).

- **Limpieza:**  
  Se eliminaron los valores nulos generados por el desplazamiento temporal para asegurar la consistencia del entrenamiento.

---

## 5.2 Estrategia de Validación: TimeSeriesSplit

Debido a la naturaleza secuencial de los datos financieros, se utilizó una validación cruzada específica para series de tiempo:

- **Hold-out final:**  
  Se reservó el último 10% de los datos cronológicos exclusivamente para la prueba final del modelo.

- **Validación cruzada:**  
  Se implementó un **TimeSeriesSplit** con 5 particiones y un desfase (*gap*) de un día para evitar la filtración de información futura y la dependencia inmediata entre entrenamiento y validación.

---

## 5.3 Comparación General de Desempeño (MSE)

Resultados del **Error Cuadrático Medio (MSE)** obtenidos en el conjunto de prueba:

| Modelo | MSE Test | Interpretación del Desempeño |
|------|----------|------------------------------|
| XGBoost | 0.074090 | Presenta el menor error, logrando capturar mejor la relación entre sentimiento cripto y tipo de cambio |
| Ridge | 0.074256 | Desempeño muy similar a XGBoost; sugiere que la serie no posee patrones no lineales complejos |
| Random Forest | 0.076369 | Menor precisión acumulada al suavizar excesivamente los movimientos diarios |

---

## 5.7 Importancia de Variables (Random Forest)

El análisis de importancia relativa permite identificar qué factores tienen mayor peso en la formación del tipo de cambio diario:

- **Dominancia de la Inercia:**  
  El factor más determinante es el retorno del día anterior (ret\_USD\_lag1), con una importancia cercana a **0.20**. Esto confirma que el mercado cambiario peruano tiene una persistencia interna significativa.

- **Riesgo Global:**  
  Los rezagos del índice de volatilidad (**VIX\_lag7** y **VIX\_lag1**) muestran una relevancia secundaria, indicando que el riesgo internacional influye gradualmente en el sol peruano.

- **Sentimiento Cripto (FGI):**  
  La importancia del FGI es marginal. No es un determinante principal, aunque aporta información mínima detectada por modelos flexibles.

---

## 5.8 Validación Visual: Tendencias Acumuladas

Al comparar la trayectoria real contra la predicha en los últimos 100 días, se observa lo siguiente:

- **Ajuste de XGBoost:**  
  Es el modelo que más se aproxima a la pendiente general de la tendencia real.

- **Limitaciones:**  
  Ningún modelo captura con precisión los saltos abruptos o reversiones repentinas del mercado. Estos movimientos son atribuidos a factores externos impredecibles o intervenciones directas del BCRP para contener la volatilidad.

---

## Enfoque Opcional: Modelo en Dos Etapas (Two-Stage Model)

## 5.9 Separación entre Inercia y Shocks

Este enfoque busca aislar el componente estructural del tipo de cambio para modelar únicamente lo inesperado:

- **Etapa 1 (Inercia):**  
  Un modelo autorregresivo AR(1) captura la dinámica interna del tipo de cambio.

- **Etapa 2 (Shocks):**  
  Un Random Forest intenta predecir los residuos (lo que la inercia no explicó) utilizando variables exógenas como BTC, FGI y VIX.

- **Resultados:**  
  El MSE obtenido (**0.074193**) no mostró mejoras sustanciales respecto a los modelos directos. Se confirmó que los retornos diarios están dominados por ruido y que la inercia tiene un poder explicativo limitado en el muy corto plazo.

---

## 5.10 Importancia en los Shocks

Al analizar qué variables explican los movimientos inesperados (*shocks*), se identificaron los siguientes impulsores principales:

- **Ajuste Semanal:**  
  El rezago **USD_PEN_Venta_lag7** es el predictor más fuerte de los residuos, sugiriendo patrones de cierre contable o intervenciones estacionales del Banco Central.

- **Señales de Liquidez:**  
  Los **T-Bills a 13 semanas** y el **VIX** explican shocks que no están correlacionados con el comportamiento pasado del tipo de cambio.

- **Sentimiento Cripto:**  
  El FGI casi no contribuye a explicar estos movimientos bruscos, confirmando que su influencia no es sistemática.

---

## Puntos más importantes del análisis

- **Predicibilidad Limitada:**  
  Los modelos lineales (Ridge) y no lineales (XGBoost) obtienen resultados casi idénticos. Esto indica que el tipo de cambio USD/PEN se comporta de manera altamente aleatoria en el corto plazo, sin estructuras no lineales aprovechables.

- **Influencia del Banco Central:**  
  La estabilidad del sol y la dificultad de los modelos para capturar quiebres bruscos reflejan la política de intervención del BCRP, que suaviza la volatilidad y reduce los patrones persistentes.

- **Desconexión del Sentimiento:**  
  El Fear & Greed Index (FGI) carece de poder predictivo estadísticamente significativo para el sol peruano. El tipo de cambio responde a su propia inercia y a shocks macroeconómicos amplios, no a la euforia o pánico del mercado cripto.

# TRABAJO 4: Análisis Causal y Modelado de Redes Neuronales

## 6. Análisis Causal mediante Gráficos Acíclicos Dirigidos (DAG)

En esta fase se construyó un **Gráfico Acíclico Dirigido (DAG)** para representar de forma explícita las relaciones causales asumidas entre las variables del estudio. El objetivo primordial fue identificar caminos causales que podrían generar sesgo si no se controlan, permitiendo visualizar la interacción entre los shocks globales, el sentimiento del mercado y la política monetaria.

### Estructura del Mecanismo Causal

- **Variables Observadas:**  
  Se incluyeron indicadores como el **Fear & Greed Index (FGI)**, **Bitcoin (BTC)**, el tipo de cambio **USD/PEN**, y variables macro-financieras como **VIX**, **DXY**, **Oro**, **T-Bills** y **Treasury 5y**.

- **Variables Latentes (No Observadas):**  
  Se incorporaron nodos críticos que influyen en el sistema pero no están presentes en el dataset, tales como la **Política Monetaria del BCRP**, **Shocks Globales** (pandemias o crisis financieras) y los **Flujos de Capital**.

- **Dinámica de Interacción:**  
  El DAG representa cómo el sentimiento global (FGI) puede movilizar la demanda por dólares y el precio del Bitcoin simultáneamente, influyendo en los retornos cambiarios.

### Interpretación Económica del DAG

El análisis revela un sistema donde la mayoría de las interacciones entre los predictores observados se explican por **causas comunes omitidas**. Los shocks globales actúan como variables de confusión (*confounders*) al afectar al mismo tiempo al FGI, al VIX, al oro y al Bitcoin.

Por otro lado, factores domésticos como la política monetaria local y los flujos de capital operan como canales directos que presionan el precio del dólar. Finalmente, el modelo incorpora formalmente la persistencia del retorno del dólar mediante un bucle autorregresivo, reforzando la tesis de que la dinámica cambiaria depende tanto de shocks externos como de su propia inercia interna.

---

## 7. Modelo de Redes Neuronales (MLP)

Se implementó un Perceptrón Multicapa (MLPRegressor) con una estrategia de optimización de hiperparámetros (GridSearchCV) para evaluar si una arquitectura de aprendizaje profundo podía capturar relaciones no lineales complejas que los modelos estadísticos tradicionales no detectan.

## Arquitectura y Configuración del Modelo

A diferencia de las pruebas iniciales, se utilizó una validación cruzada temporal para encontrar la arquitectura óptima1. Los mejores hiperparámetros encontrados fueron:

- **Estructura de Capas:** Tres capas ocultas densas de (128, 64, 32) neuronas.
- **Función de Activación:** Logistic (Sigmoide), lo que sugiere una mejor adaptación a la naturaleza acotada de los retornos que la función ReLU.
- **Regularización:** Alpha de 0.001 (penalización L2).
- **Optimizador:** Adam con una tasa de aprendizaje inicial de 0.001.

## Desempeño y Comparativa

La optimización mejoró drásticamente el desempeño del modelo. A diferencia de las iteraciones previas donde el modelo no generalizaba, el MLP optimizado alcanzó niveles de error competitivos frente a los modelos de ensamble y lineales:

| Modelo         | MSE (Test) | Interpretación del Desempeño                         |
|---------------|------------|-----------------------------------------------------|
| XGBoost       | 0.074090   | Mejor desempeño (por margen mínimo).                |
| Ridge (Lineal)| 0.074256   | Alta competitividad pese a su simplicidad.          |
| MLP (Optimizado) | 0.074719 | Convergencia exitosa al nivel de los otros modelos. |
| Random Forest | 0.076369   | Desempeño ligeramente inferior.                     |

## Interpretación del Análisis MLP

**Convergencia de Modelos:**  
El hallazgo más revelador es que una red neuronal profunda compleja (MLP) converge prácticamente al mismo error cuadrático medio (MSE ~0.0747) que una regresión lineal penalizada (Ridge, MSE ~0.0742).

**Ausencia de No Linealidad Explotable:**  
El hecho de que la complejidad adicional de la red neuronal no se traduzca en una mejora predictiva sugiere que la serie de retornos del USD/PEN no contiene patrones no lineales ocultos significativos.

**Eficiencia del Mercado:**  
Los resultados refuerzan la hipótesis de que la dinámica del tipo de cambio está dominada por el ruido de alta frecuencia y la inercia lineal, haciendo que los modelos sofisticados "colapsen" hacia soluciones lineales.

# 8. Conclusiones Generales

El desarrollo de este análisis, que integra modelos estáticos, dinámicos (ARX) y de machine learning avanzado, permite establecer conclusiones robustas sobre la dinámica del mercado cambiario peruano frente al ecosistema cripto:

**Insignificancia del Sentimiento Cripto:**  
El Fear & Greed Index (FGI) no presenta un efecto estadísticamente significativo sobre los retornos diarios del USD/PEN, ni en modelos lineales ni en redes neuronales. El sentimiento global del mercado digital no se transmite al mercado cambiario peruano en el corto plazo.

**Dominancia de la Inercia (Memoria):**  
El tipo de cambio peruano exhibe una fuerte persistencia autorregresiva. El predictor más potente en todos los modelos fue consistentemente el retorno rezagado del propio tipo de cambio (ret_USD_lag1), confirmando que el sol se mueve principalmente por su propia dinámica histórica e intervenciones de suavización.

**Equivalencia Predictiva:**  
Todos los modelos evaluados (Ridge, XGBoost y MLP) convergieron a un desempeño predictivo casi idéntico (MSE ≈ 0.075). Esto indica que el error restante corresponde a ruido de mercado irreductible y shocks estocásticos, y no a una falta de capacidad del modelo.

**Rol del Bitcoin:**  
Aunque existe una correlación contemporánea positiva y moderada, el Bitcoin tiene un aporte marginal en la predicción futura. Actúa más como un termómetro de liquidez global coincidente que como un predictor adelantado de la moneda peruana.

---

# 9. Discusión Económica

Los hallazgos de esta investigación son consistentes con la literatura empírica sobre mercados emergentes y la microestructura del mercado cambiario peruano.

### Relación entre Sentimiento y Activos

La ausencia de impacto del FGI sobre el USD/PEN valida la tesis de **Baker y Wurgler (2007)**, quienes sostienen que el sentimiento de los inversionistas tiene una influencia determinante en activos con alta incertidumbre fundamental y carácter especulativo. El mercado cambiario peruano, caracterizado por fundamentos sólidos, no encaja en esta descripción.

### Naturaleza del Bitcoin

La limitada capacidad predictiva del Bitcoin sobre el sol peruano coincide con los estudios de **Baur et al. (2018)** y **Corbet et al. (2019)**. Estos autores destacan que las criptomonedas responden a shocks globales de liquidez y apetito por riesgo, pero no mantienen vínculos causales directos con los mercados cambiarios tradicionales.

### El Rol de la Política Monetaria

La fuerte persistencia observada en los retornos del USD/PEN es coherente con los reportes del **Banco Central de Reserva del Perú (BCRP)**. La estabilidad de la moneda peruana se explica por:

- Mecanismos de intervención esterilizada del BCRP que suavizan la volatilidad.  
- Una microestructura de mercado que prioriza la previsibilidad y reduce los movimientos erráticos.  
- Un régimen de metas explícitas de inflación que ancla las expectativas de los agentes económicos.

### Machine Learning en Series Financieras

El hecho de que modelos sofisticados como **XGBoost** no superen de manera sustancial a modelos lineales como **Ridge** confirma que, en mercados con fuerte presencia institucional, la mayor parte de la variación diaria es ruido y no información aprovechable. Los modelos complejos capturan interacciones interesantes, pero no logran vencer la eficiencia del mercado cuando la señal fundamental es débil.

En conjunto, los resultados refuerzan la idea de que el **USD/PEN** es un activo de baja volatilidad y alta resiliencia institucional, cuya transmisión frente al sentimiento global especulativo es significativamente más débil que en otros activos financieros de la región.

# 10. Referencias Bibliográficas

## Sentimiento y mercados financieros

Baker, M., & Wurgler, J. (2007). Investor sentiment in the stock market. *Journal of Economic Perspectives*, 21(2), 129–151.

García, D. (2013). Sentiment during recessions. *Journal of Finance*, 68(3), 1267–1300.

## Bitcoin, riesgo global y mercados financieros

Baur, D. G., Hong, K., & Lee, A. D. (2018). Bitcoin: medium of exchange or speculative assets? *Journal of International Financial Markets, Institutions & Money*, 54, 177–189.

Corbet, S., Lucey, B., & Yarovaya, L. (2019). The financial market effects of cryptocurrency energy prices. *Energy Economics*, 84, 104502.

## Tipo de cambio y macroeconomía peruana

Banco Central de Reserva del Perú (BCRP). (2023). *Reporte de Inflación*. Lima, Perú.

Rossini, A. (2010). Intervenciones cambiarias y mecanismos de transmisión en economías emergentes. Banco Central de Reserva del Perú.

## Series de tiempo y modelos ARX

Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press.

Stock, J. H., & Watson, M. W. (2016). *Introduction to Econometrics*. Pearson.

## Machine Learning y eficiencia de mercados financieros

Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *Review of Financial Studies*, 33(5), 2223–2273.
