# 📊 Sistema Completo de Regresión Lineal OLS con Validación de Modelo

Sistema web profesional para realizar regresión lineal múltiple por Mínimos Cuadrados Ordinarios (OLS) con **validación completa de supuestos del modelo**.

## ✨ Características Principales

### Análisis OLS Completo
✅ Estimación de coeficientes β̂ = (X'X)⁻¹X'y  
✅ Errores estándar, estadísticos t, p-values  
✅ Intervalos de confianza al 95%
✅ R², R² ajustado  
✅ Determinación de significancia estadística  

### Validación de Supuestos
✅ **Normalidad de Residuos** - Prueba de Jarque-Bera  
✅ **Homocedasticidad** - Prueba de Breusch-Pagan  
✅ **Autocorrelación** - Prueba de Durbin-Watson  
✅ **Multicolinealidad** - Factor de Inflación de Varianza (VIF)  

### Diagnóstico Visual
✅ Histograma de residuos con curva normal  
✅ Q-Q Plot para normalidad  
✅ Residuos vs Valores Ajustados  
✅ Residuos vs Índice de observación  

### Interface Web Profesional
✅ Interface intuitiva con Flask  
✅ Carga de archivos CSV/Excel  
✅ Visualización interactiva de resultados  
✅ Descarga de resultados en Excel  

## 📁 Estructura del Proyecto

```
proyecto_ols_flask/
├── app.py                  # Aplicación Flask principal
├── requirements.txt        # Dependencias
├── README_FLASK.md        # Esta guía
├── templates/
│   └── index.html         # Interfaz web
├── uploads/               # Carpeta para archivos subidos (se crea automáticamente)
└── mini.csv               # Archivo de prueba
```

## 🚀 Instalación y Configuración

### Paso 1: Clonar o descargar el proyecto
```bash
mkdir proyecto_ols_flask
cd proyecto_ols_flask
```

### Paso 2: Crear entorno virtual (recomendado)
```bash
# En Windows
python -m venv venv
venv\Scripts\activate

# En Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Paso 3: Instalar dependencias
```bash
pip install -r requirements.txt
```

### Paso 4: Crear estructura de carpetas
```bash
mkdir templates
mkdir uploads
```

### Paso 5: Colocar archivos
- Guardar `app.py` en la raíz del proyecto
- Guardar `index.html` dentro de la carpeta `templates/`
- Guardar `mini.csv` en la raíz (opcional, para pruebas)

## ▶️ Ejecutar la Aplicación

```bash
python app.py
```

La aplicación estará disponible en: **http://localhost:5000**

## 📖 Uso del Sistema

### 1. Acceder a la aplicación
Abrir navegador en `http://localhost:5000`

### 2. Cargar datos
- **Archivo:** Seleccionar CSV o Excel (máximo 5000 filas)
- **Variable Y:** Nombre exacto de la variable dependiente
- **Variables X:** Nombres separados por comas (1-10 variables)
- **Intercepto:** Sí/No
- **Nivel α:** 0.01, 0.05 o 0.10

### 3. Ejecutar análisis
Click en "🚀 Ejecutar Regresión y Validación"

### 4. Interpretar resultados

#### **Tabla de Coeficientes**
Muestra para cada variable:
- **Beta (β̂):** Coeficiente estimado
- **Error Estándar:** SE(β̂)
- **t:** Estadístico t = β̂/SE
- **P-value:** Significancia estadística
- **IC 95%:** Intervalo de confianza
- **Significativo:** Sí/No según α

#### **Resumen del Modelo**
- **n:** Número de observaciones
- **k:** Número de parámetros
- **RSS:** Suma de cuadrados de residuos
- **σ²:** Varianza del error
- **R²:** Bondad de ajuste
- **R² ajustado:** R² penalizado por variables

#### **Validación de Supuestos**

**🔔 Prueba de Jarque-Bera (Normalidad)**
- **H₀:** Los residuos siguen distribución normal
- **Interpretación:** 
  - p-value > α → No se rechaza H₀ (residuos normales) ✅
  - p-value < α → Se rechaza H₀ (residuos no normales) ❌

**📊 Prueba de Breusch-Pagan (Homocedasticidad)**
- **H₀:** Varianza constante (homocedasticidad)
- **Interpretación:**
  - p-value > α → Varianza constante ✅
  - p-value < α → Heterocedasticidad ❌

**🔄 Prueba de Durbin-Watson (Autocorrelación)**
- **Estadístico DW:** Rango [0, 4]
- **Interpretación:**
  - DW ≈ 2 → No autocorrelación ✅
  - DW < 1.5 → Autocorrelación positiva ❌
  - DW > 2.5 → Autocorrelación negativa ❌

**🔗 Factor de Inflación de Varianza (Multicolinealidad)**
- **Regla:** VIF < 10 es aceptable
- **Interpretación:**
  - VIF < 5 → Baja multicolinealidad ✅
  - 5 < VIF < 10 → Multicolinealidad moderada ⚠️
  - VIF > 10 → Multicolinealidad severa ❌

#### **Gráficos de Diagnóstico**
1. **Histograma de residuos:** Debe aproximarse a curva normal
2. **Q-Q Plot:** Puntos deben estar cerca de la línea diagonal
3. **Residuos vs Ajustados:** Dispersión aleatoria sin patrones
4. **Residuos vs Índice:** No debe haber tendencias

### 5. Descargar resultados
Click en "💾 Descargar Resultados Completos (Excel)"

El archivo Excel incluye:
- **Hoja 1:** Coeficientes estimados
- **Hoja 2:** Resumen del modelo
- **Hoja 3:** Resultados de pruebas diagnósticas

## 📝 Ejemplo con mini.csv

### Archivo mini.csv
```csv
y,x1,x2
1.20,1,2
1.70,2,1
1.95,3,3
1.95,4,5
2.70,5,4
```

### Configuración
- **Archivo:** mini.csv
- **Variable Y:** y
- **Variables X:** x1,x2
- **Intercepto:** Sí
- **Nivel α:** 0.05

### Resultados Esperados
```
Coeficientes:
- Intercepto: β₀ ≈ 1.025 (significativo)
- x1: β₁ ≈ 0.458 (significativo)
- x2: β₂ ≈ -0.167 (no significativo)

Modelo:
- n = 5
- k = 3
- R² ≈ 0.95
```

## 🔧 Especificaciones Técnicas

### Fórmulas Implementadas

#### Estimación OLS
```
β̂ = (X'X)⁻¹X'y
ŷ = Xβ̂
e = y - ŷ
RSS = e'e
σ² = RSS/(n-k)
Var(β̂) = σ²(X'X)⁻¹
SE(β̂) = √diag(Var(β̂))
t = β̂/SE(β̂)
p-value = 2×P(T > |t|), df = n-k
IC 95% = β̂ ± t₀.₉₇₅ × SE(β̂)
```

#### Jarque-Bera
```
S = E[(e - μ)³]/σ³  (skewness)
K = E[(e - μ)⁴]/σ⁴  (kurtosis)
JB = (n/6)[S² + (K-3)²/4]
JB ~ χ²(2)
```

#### Breusch-Pagan
```
1. e² = Xγ + u  (regresión auxiliar)
2. BP = n×R²_aux
3. BP ~ χ²(k-1)
```

#### Durbin-Watson
```
DW = Σ(eₜ - eₜ₋₁)²/Σeₜ²
Rango: [0, 4]
```

#### VIF
```
Para cada Xⱼ:
1. Xⱼ = X₋ⱼδ + u  (regresión auxiliar)
2. VIFⱼ = 1/(1 - R²ⱼ)
```

### Restricciones
- **Observaciones:** 1 ≤ n ≤ 5000
- **Variables X:** 1 ≤ p ≤ 10
- **Datos:** Solo valores numéricos
- **Formato:** CSV o Excel (.xls, .xlsx)

### Manejo de Datos Faltantes
- Las filas con NA en Y o cualquier X se eliminan automáticamente
- El sistema reporta cuántas filas fueron eliminadas
- Si no quedan observaciones válidas, se muestra error

## ⚙️ Configuración Avanzada

### Cambiar Puerto
En `app.py`, línea final:
```python
app.run(debug=True, port=5000)  # Cambiar 5000 por el puerto deseado
```

### Aumentar Límite de Tamaño de Archivo
En `app.py`:
```python
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
```

### Modo Producción
```bash
# Desactivar modo debug
app.run(debug=False, host='0.0.0.0', port=5000)

# O usar servidor WSGI (recomendado)
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## 🐛 Solución de Problemas

### Error: ModuleNotFoundError
```bash
# Reinstalar dependencias
pip install -r requirements.txt
```

### Error: Template not found
```bash
# Verificar que index.html esté en templates/
mkdir templates
# Mover index.html a templates/
```

### Error: Archivo muy grande
```bash
# Aumentar límite en app.py
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB
```

### Error: Puerto en uso
```bash
# Cambiar puerto en app.py o liberar puerto
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:5000 | xargs kill -9
```

## 📊 Interpretación de Resultados

### ¿Cuándo es válido el modelo?

El modelo OLS es **válido** si cumple:

1. ✅ **Coeficientes significativos** (p-value < α)
2. ✅ **Residuos normales** (Jarque-Bera p > α)
3. ✅ **Homocedasticidad** (Breusch-Pagan p > α)
4. ✅ **No autocorrelación** (DW ≈ 2)
5. ✅ **No multicolinealidad** (VIF < 10)
6. ✅ **Buen ajuste** (R² alto, preferible > 0.7)

### ¿Qué hacer si falla alguna prueba?

❌ **Residuos no normales:**
- Verificar outliers
- Transformar variable Y (log, sqrt)
- Aumentar tamaño muestral

❌ **Heterocedasticidad:**
- Usar errores estándar robustos
- Transformar variables
- Mínimos cuadrados ponderados

❌ **Autocorrelación:**
- Incluir rezagos de Y
- Modelo de series temporales
- Mínimos cuadrados generalizados

❌ **Multicolinealidad:**
- Eliminar variables correlacionadas
- Regresión en componentes principales
- Aumentar muestra

## 📚 Referencias Teóricas

- **Greene, W.H.** (2018). *Econometric Analysis* (8th ed.)
- **Wooldridge, J.M.** (2019). *Introductory Econometrics* (7th ed.)
- **Davidson, R. & MacKinnon, J.G.** (2004). *Econometric Theory and Methods*

## 🤝 Contribuciones

Para reportar bugs o sugerir mejoras, contactar al desarrollador.

## 📄 Licencia

Este sistema es de uso educativo para cursos de econometría y estadística.

---

**Desarrollado con:** Python 3.x + Flask + NumPy + SciPy + Matplotlib  
**Última actualización:** Noviembre 2025  
**Versión:** 2.0 (con validación completa)

## 🎯 Resumen Rápido

```bash
# 1. Instalar
pip install -r requirements.txt

# 2. Ejecutar
python app.py

# 3. Abrir navegador
http://localhost:5000

# 4. Subir datos y ejecutar
```

¡Listo para usar! 🚀