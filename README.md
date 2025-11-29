
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

La motivación surge de la creciente interconexión entre mercados financieros globales. Aunque tradicionalmente el tipo de cambio responde a factores macroeconómicos, es plausible que indicadores alternativos relacionados al **sentimiento global de riesgo** afecten indirectamente economías emergentes como la peruana.

Se construyó un dataset consolidando series del criptomercado, indicadores globales y el tipo de cambio USD/PEN. A lo largo de los Trabajos 1–4 se aplicaron herramientas de estadística, series de tiempo, machine learning y análisis causal.

---

# TRABAJO 1 – Principales Resultados

## 2. Importación y Preparación de Datos

Se realizó limpieza, alineación temporal y estandarización del dataset.  
Variables incluidas:

- FGI (Fear & Greed Index)
- USD/PEN (venta)
- DXY
- VIX
- BTC/USD
- Oro
- Treasury Bills 13w
- Treasury 5y

Estas variables permiten desarrollar:

- Modelos ARX
- Modelos ML
- Análisis causal con DAGs
- Estadística exploratoria

---

## 3. Análisis Exploratorio (EDA)

### 3.1 Evolución Histórica

Se graficaron USD/PEN, FGI, BTC, Oro y VIX.  
Hallazgos:

- El tipo de cambio muestra tendencia suave.
- BTC es altamente volátil.
- FGI presenta episodios de "miedo extremo".
- El VIX sube en periodos de estrés global.

### 3.2 Histogramas de Retornos

- USD/PEN tiene retornos concentrados en 0 (alta estabilidad).
- Bitcoin presenta colas pesadas.

### 3.3 Gráficos de Dispersión

- No hay relación lineal fuerte entre FGI y ret_USD.
- BTC muestra asociación moderada.

### 3.4 Mapa de Calor de Correlaciones

- FGI y USD/PEN → correlación baja.
- Oro correlaciona con VIX y DXY.
- BTC correlaciona moderadamente con riesgo global.

### 3.5 Bimodalidad del FGI

Clasificación en 5 categorías: Extreme Fear → Extreme Greed.  
Predominan estados moderados.

---

# TRABAJO 2 – Principales Resultados

## 4. Modelo Dinámico ARX

Modelo estimado:

ret_USD_t = α + β * ret_USD_(t-1) + γ * FGI_(t-1) + u_t

yaml
Copiar código

Objetivos:

- Detectar inercia del tipo de cambio.
- Evaluar aporte informativo del FGI.

Resultados:

- **El coeficiente autoregresivo es significativo.**
- **El FGI NO es significativo.**
- El ARX reduce ligeramente el MSE respecto al AR.

---

# TRABAJO 3 – Principales Resultados

## 5. Modelos Avanzados de Regresión

Implementados:

- Ridge
- Random Forest
- XGBoost
- MLP (Red Neuronal)

### 5.1 Feature Engineering

Se generaron rezagos t−1, t−2, t−7 de:

- USD/PEN  
- BTC  
- FGI  
- indicadores globales

### 5.2 Validación: TimeSeriesSplit

- 5 divisiones  
- Test final = último 10 %

### 5.3 Ridge Regression

- Requiere estandarización.
- MSE moderado (modelo lineal).

### 5.4 Random Forest

- Captura no linealidades.
- Mejor que Ridge.
- FGI tiene baja importancia.

### 5.5 XGBoost

- **Mejor desempeño general (MSE más bajo).**

### 5.6 Comparación General

`XGBoost > Random Forest > Ridge > MLP`

### 5.7 Importancia de Variables

- Rezagos del USD/PEN dominan.
- BTC tiene relevancia media.
- FGI es de las menos importantes.

### 5.8 Tendencias Acumuladas

Los mejores modelos siguen la dirección del tipo de cambio pero no capturan todos los shocks externos.

---

# Enfoque Opcional: Two-Stage Model

## 5.9 Separación entre Inercia y Shocks

**Etapa 1:** AR(1) captura dinámica interna.  
**Etapa 2:** RF predice los residuos (shocks).

Resultados:

- Existen shocks externos
- **FGI no explica los shocks**
- VIX, DXY y BTC sí tienen algún aporte.

## 5.10 Importancia en los Shocks

- VIX, DXY y BTC explican mejor las innovaciones.
- FGI casi no contribuye.

## 5.11 Gráfico Final de Componentes

Incluye:

- Retorno total  
- Shock real  
- Shock predicho

---

# TRABAJO 4 – Análisis Causal

## 6. DAG (Directed Acyclic Graph)

Incluye:

- Variables observadas: FGI, BTC, USD/PEN, VIX, DXY, Oro, T-Bills, Treasury 5y
- Variables no observadas: política monetaria, flujos de capital, shocks globales

Permite:

- Entender relaciones causales
- Identificar confounders
- Distinguir rutas abiertas/cerradas

---

# 7. Modelo MLP (Neural Network)

- Arquitectura 64–32, ReLU, Adam
- Early stopping
- R² negativo en test

Conclusión: **no capta estructura temporal.**

---

# 8. Conclusiones Generales

- **FGI NO afecta significativamente los retornos USD/PEN.**
- USD/PEN responde principalmente a:
  - su propia inercia,
  - shocks globales amplios (VIX, DXY),
  - factores macroestructurales.
- BTC tiene relación marginal con el tipo de cambio.
- **XGBoost y Random Forest son los mejores modelos.**
- MLP no generaliza con pocos datos.

---

# 9. Discusión Económica

Los resultados coinciden con literatura empírica:

- El sentimiento afecta activos especulativos (Baker & Wurgler, 2007).  
- Bitcoin correlaciona con apetito global por riesgo (Baur et al., 2018).  
- En economías emergentes estables, el tipo de cambio responde a variables globales amplias: VIX, DXY, tasas del Tesoro, commodities.

En conjunto:  
**El tipo de cambio peruano es estable y poco sensible al sentimiento del criptomercado.**
