"""
Sistema de Regresión Lineal OLS con Validación de Modelo
Flask Web Application
Incluye: OLS + Pruebas de Normalidad, Homocedasticidad, Multicolinealidad, Autocorrelación
"""

from flask import Flask, render_template, request, send_file, jsonify, session
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from io import BytesIO
import base64
import os
from werkzeug.utils import secure_filename
import json
import collections.abc
import uuid
import json

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-in-production'
app.config['UPLOAD_FOLDER'] = 'uploads'
TMP_FOLDER = "tmp_results"
os.makedirs(TMP_FOLDER, exist_ok=True)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Crear carpeta uploads si no existe
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class OLSValidation:
    """Clase extendida con validación de supuestos"""
    
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.beta = None
        self.se = None
        self.t_stats = None
        self.p_values = None
        self.ci_lower = None
        self.ci_upper = None
        self.n = None
        self.k = None
        self.rss = None
        self.sigma2 = None
        self.r2 = None
        self.r2_adj = None
        self.var_names = None
        self.rows_dropped = 0
        self.residuals = None
        self.fitted_values = None
        self.X = None
        self.y = None
        
    def fit(self, X, y, intercept=True, var_names=None):
        """Ajusta el modelo OLS"""
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float).reshape(-1, 1)
        
        # Añadir intercepto
        if intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])
            if var_names:
                self.var_names = ['Intercepto'] + list(var_names)
            else:
                self.var_names = ['Intercepto'] + [f'X{i}' for i in range(1, X.shape[1])]
        else:
            self.var_names = var_names if var_names else [f'X{i}' for i in range(1, X.shape[1] + 1)]
        
        self.X = X
        self.y = y
        self.n = X.shape[0]
        self.k = X.shape[1]
        
        if self.n <= self.k:
            raise ValueError(f"Insuficientes observaciones: n={self.n}, k={self.k}")
        
        # Calcular β̂ = (X'X)^(-1)X'y
        XtX = X.T @ X
        XtX_inv = np.linalg.inv(XtX)
        Xty = X.T @ y
        self.beta = XtX_inv @ Xty
        
        # Predicciones y residuos
        self.fitted_values = X @ self.beta
        self.residuals = y - self.fitted_values
        
        # RSS y σ²
        self.rss = float((self.residuals.T @ self.residuals)[0, 0])
        df = self.n - self.k
        self.sigma2 = self.rss / df
        
        # Varianza de β̂
        var_beta = self.sigma2 * XtX_inv
        self.se = np.sqrt(np.diag(var_beta)).reshape(-1, 1)
        
        # Estadísticos t
        self.t_stats = self.beta / self.se
        
        # P-values (dos colas)
        self.p_values = 2 * (1 - stats.t.cdf(np.abs(self.t_stats), df))
        
        # Intervalos de confianza
        t_crit = stats.t.ppf(1 - self.alpha/2, df)
        self.ci_lower = self.beta - t_crit * self.se
        self.ci_upper = self.beta + t_crit * self.se
        
        # R² y R² ajustado
        tss = np.sum((y - np.mean(y))**2)
        self.r2 = 1 - (self.rss / tss) if tss > 0 else 0
        self.r2_adj = 1 - (1 - self.r2) * (self.n - 1) / (self.n - self.k)
    
    def jarque_bera_test(self):
        """Prueba de normalidad de Jarque-Bera"""
        resid = self.residuals.flatten()
        n = len(resid)
        
        # Calcular skewness y kurtosis
        mean_resid = np.mean(resid)
        std_resid = np.std(resid, ddof=1)
        
        skewness = np.mean(((resid - mean_resid) / std_resid) ** 3)
        kurtosis = np.mean(((resid - mean_resid) / std_resid) ** 4)
        
        # Estadístico JB
        jb_stat = (n / 6) * (skewness**2 + (kurtosis - 3)**2 / 4)
        jb_pvalue = 1 - stats.chi2.cdf(jb_stat, df=2)
        
        return {
            'statistic': jb_stat,
            'p_value': jb_pvalue,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'normal': jb_pvalue > self.alpha,
            'conclusion': 'Los residuos siguen una distribución normal' if jb_pvalue > self.alpha 
                         else 'Los residuos NO siguen una distribución normal'
        }
    
    def breusch_pagan_test(self):
        """Prueba de homocedasticidad de Breusch-Pagan"""
        resid_sq = self.residuals.flatten() ** 2
        
        # Regresión auxiliar: e² sobre X
        X_aux = self.X
        XtX = X_aux.T @ X_aux
        XtX_inv = np.linalg.inv(XtX)
        beta_aux = XtX_inv @ (X_aux.T @ resid_sq.reshape(-1, 1))
        
        fitted_aux = X_aux @ beta_aux
        tss_aux = np.sum((resid_sq - np.mean(resid_sq))**2)
        rss_aux = np.sum((resid_sq - fitted_aux.flatten())**2)
        r2_aux = 1 - (rss_aux / tss_aux) if tss_aux > 0 else 0
        
        # Estadístico BP = n * R²
        bp_stat = self.n * r2_aux
        bp_pvalue = 1 - stats.chi2.cdf(bp_stat, df=self.k - 1)
        
        return {
            'statistic': bp_stat,
            'p_value': bp_pvalue,
            'homoscedastic': bp_pvalue > self.alpha,
            'conclusion': 'Los errores son homocedásticos (varianza constante)' if bp_pvalue > self.alpha
                         else 'Hay evidencia de heterocedasticidad (varianza no constante)'
        }
    
    def durbin_watson_test(self):
        """Prueba de autocorrelación de Durbin-Watson"""
        resid = self.residuals.flatten()
        diff_resid = np.diff(resid)
        
        dw_stat = np.sum(diff_resid**2) / np.sum(resid**2)
        
        # Interpretación: DW ≈ 2 indica no autocorrelación
        # DW < 2 indica autocorrelación positiva
        # DW > 2 indica autocorrelación negativa
        
        if 1.5 <= dw_stat <= 2.5:
            conclusion = 'No hay evidencia significativa de autocorrelación'
            autocorrelated = False
        elif dw_stat < 1.5:
            conclusion = 'Hay evidencia de autocorrelación positiva'
            autocorrelated = True
        else:
            conclusion = 'Hay evidencia de autocorrelación negativa'
            autocorrelated = True
        
        return {
            'statistic': dw_stat,
            'autocorrelated': autocorrelated,
            'conclusion': conclusion
        }
    
    def vif_test(self):
        """Factor de Inflación de Varianza (VIF) para multicolinealidad"""
        if self.k <= 2:  # Solo intercepto + 1 variable
            return {
                'vif_values': [],
                'multicollinear': False,
                'conclusion': 'Menos de 2 variables, no aplica VIF'
            }
        
        X_no_intercept = self.X[:, 1:]  # Remover intercepto
        n_vars = X_no_intercept.shape[1]
        vif_values = []
        
        for i in range(n_vars):
            # Regresión de Xi sobre las demás X
            X_i = X_no_intercept[:, i]
            X_others = np.delete(X_no_intercept, i, axis=1)
            
            # Añadir intercepto a X_others
            X_others = np.column_stack([np.ones(X_others.shape[0]), X_others])
            
            # Calcular R² de la regresión auxiliar
            XtX = X_others.T @ X_others
            XtX_inv = np.linalg.inv(XtX)
            beta = XtX_inv @ (X_others.T @ X_i.reshape(-1, 1))
            fitted = X_others @ beta
            
            tss = np.sum((X_i - np.mean(X_i))**2)
            rss = np.sum((X_i.reshape(-1, 1) - fitted)**2)
            r2 = 1 - (rss / tss) if tss > 0 else 0
            
            # VIF = 1 / (1 - R²)
            vif = 1 / (1 - r2) if r2 < 0.9999 else 999
            vif_values.append({
                'variable': self.var_names[i + 1],  # +1 por el intercepto
                'vif': vif
            })
        
        max_vif = max([v['vif'] for v in vif_values]) if vif_values else 0
        multicollinear = max_vif > 10
        
        return {
            'vif_values': vif_values,
            'multicollinear': multicollinear,
            'conclusion': 'No hay problemas graves de multicolinealidad (VIF < 10)' if not multicollinear
                         else 'Hay problemas de multicolinealidad (VIF > 10 en alguna variable)'
        }
        
    def white_test(self):
        """
        Prueba de heterocedasticidad de White
        Usa regresión auxiliar con términos X, X^2 y productos cruzados.
        """
        try:
            resid_sq = self.residuals.flatten()
            X_no_intercept = self.X[:, 1:]  # sacar intercepto
            if X_no_intercept.shape[1] == 0:
                return {
                    'statistic': None,
                    'p_value': None,
                    'df': 0,
                    'efficient': True,
                    'conclusion': 'No hay variables independientes para realizar la prueba White'
                }
            resid_sq = self.residuals.flatten()
            X_no_intercept = self.X[:, 1:]  # sacar intercepto para generar términos
            
            n, k = X_no_intercept.shape
            Z = []

            # 1) agregar X
            Z.append(X_no_intercept)

            # 2) agregar X^2
            Z.append(X_no_intercept ** 2)

            # 3) agregar productos cruzados X_i * X_j (solo cuando i < j)
            cross_terms = []
            for i in range(k):
                for j in range(i + 1, k):
                    cross_terms.append((X_no_intercept[:, i] * X_no_intercept[:, j]).reshape(-1, 1))
            if cross_terms:
                Z.append(np.hstack(cross_terms))

            # Unir todos los términos en una sola matriz
            Z = np.hstack(Z)

            # Agregar intercepto
            Z = np.column_stack([np.ones(n), Z])

            # Regresión auxiliar: e² sobre Z
            ZtZ = Z.T @ Z
            ZtZ_inv = np.linalg.inv(ZtZ)
            beta_aux = ZtZ_inv @ (Z.T @ resid_sq.reshape(-1, 1))
            fitted_aux = Z @ beta_aux

            # Calcular R² de la regresión auxiliar
            tss_aux = np.sum((resid_sq - np.mean(resid_sq)) ** 2)
            rss_aux = np.sum((resid_sq - fitted_aux.flatten()) ** 2)
            r2_aux = 1 - rss_aux / tss_aux if tss_aux > 0 else 0

            # Estadístico White = n * R²
            white_stat = n * r2_aux

            # grados de libertad = número de regresores auxiliares (sin intercepto)
            df = Z.shape[1] - 1

            p_value = 1 - stats.chi2.cdf(white_stat, df)

            conclusion = (
                "No hay evidencia de heterocedasticidad (modelo eficiente)"
                if p_value > self.alpha else
                "Existe heterocedasticidad (modelo NO eficiente)"
            )

            return {
                'statistic': float(white_stat),
                'p_value': float(p_value),
                'df': int(df),
                'efficient': p_value > self.alpha,
                'conclusion': conclusion
            }
        except Exception as e:
            return {
                'statistic': None,
                'p_value': None,
                'df': 0,
                'efficient': False,
                'conclusion': f'Error en prueba White: {str(e)}'
            }
    
    def get_diagnostics(self):
        """Obtiene todos los diagnósticos del modelo"""
        return {
            'jarque_bera': self.jarque_bera_test(),
            'breusch_pagan': self.breusch_pagan_test(),
            'white': self.white_test(),
            'durbin_watson': self.durbin_watson_test(),
            'vif': self.vif_test()
        }
    
    def plot_diagnostics(self):
        """Genera gráficos de diagnóstico"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Diagnóstico del Modelo de Regresión', fontsize=16, fontweight='bold')
        
        resid = self.residuals.flatten()
        fitted = self.fitted_values.flatten()
        
        # 1. Histograma de residuos con curva normal
        ax1 = axes[0, 0]
        ax1.hist(resid, bins=20, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Curva normal teórica
        mu, sigma = np.mean(resid), np.std(resid)
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        ax1.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2, label='Normal teórica')
        ax1.set_xlabel('Residuos')
        ax1.set_ylabel('Densidad')
        ax1.set_title('Histograma de Residuos vs Normal')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Q-Q Plot
        ax2 = axes[0, 1]
        stats.probplot(resid, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot (Normalidad)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Residuos vs Valores Ajustados
        ax3 = axes[1, 0]
        ax3.scatter(fitted, resid, alpha=0.6, s=50, edgecolors='k', linewidths=0.5)
        ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax3.set_xlabel('Valores Ajustados')
        ax3.set_ylabel('Residuos')
        ax3.set_title('Residuos vs Valores Ajustados (Homocedasticidad)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Residuos vs Índice (Autocorrelación)
        ax4 = axes[1, 1]
        ax4.plot(range(len(resid)), resid, 'o-', alpha=0.6, markersize=5)
        ax4.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax4.set_xlabel('Índice de Observación')
        ax4.set_ylabel('Residuos')
        ax4.set_title('Residuos vs Índice (Autocorrelación)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convertir a base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def get_results_table(self):
        """Tabla de coeficientes"""
        results = []
        for i in range(self.k):
            significativo = 'Sí' if self.p_values[i, 0] < self.alpha else 'No'
            results.append({
                'Variable': self.var_names[i],
                'Beta': float(self.beta[i, 0]),
                'SE': float(self.se[i, 0]),
                't': float(self.t_stats[i, 0]),
                'p_value': float(self.p_values[i, 0]),
                'IC95_low': float(self.ci_lower[i, 0]),
                'IC95_high': float(self.ci_upper[i, 0]),
                'Significativo': significativo
            })
        return results
    
    def get_summary(self, intercept_used):
        """Resumen del modelo"""
        return {
            'n_observaciones': int(self.n),
            'k_parametros': int(self.k),
            'grados_libertad': int(self.n - self.k),
            'RSS': float(self.rss),
            'sigma2': float(self.sigma2),
            'R2': float(self.r2),
            'R2_ajustado': float(self.r2_adj),
            'intercepto_usado': intercept_used,
            'filas_eliminadas': int(self.rows_dropped),
            'nivel_significancia': float(self.alpha)
        }


def load_and_validate_data(file_path, y_var, x_vars):
    """Carga y valida datos"""
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Formato no soportado")
        
        # Verificar columnas
        all_vars = [y_var] + x_vars
        missing_cols = [col for col in all_vars if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columnas no encontradas: {missing_cols}")
        
        # Limpiar datos
        df_subset = df[all_vars].copy()
        initial_rows = len(df_subset)
        df_clean = df_subset.dropna()
        rows_dropped = initial_rows - len(df_clean)
        
        # Validar numéricos
        for col in all_vars:
            if not np.issubdtype(df_clean[col].dtype, np.number):
                try:
                    df_clean[col] = pd.to_numeric(df_clean[col])
                except:
                    raise ValueError(f"La columna '{col}' contiene valores no numéricos")
        
        # Validaciones
        n = len(df_clean)
        p = len(x_vars)
        
        if n == 0:
            raise ValueError("No quedan observaciones después de eliminar NA")
        if n > 5000:
            raise ValueError(f"Demasiadas observaciones: {n} > 5000")
        if p > 10:
            raise ValueError(f"Demasiadas variables independientes: {p} > 10")
        if p < 1:
            raise ValueError("Debe especificar al menos 1 variable independiente")
        
        return df_clean, rows_dropped
        
    except Exception as e:
        raise Exception(f"Error al procesar datos: {str(e)}")


@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Procesa archivo y ejecuta regresión"""
    try:
        # Recibir archivo
        if 'file' not in request.files:
            return jsonify({'error': 'No se encontró archivo'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Archivo vacío'}), 400
        
        # Guardar archivo
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Obtener parámetros
        y_var = request.form.get('y_var')
        x_vars = [x.strip() for x in request.form.get('x_vars').split(',')]
        intercept = request.form.get('intercept', 'true').lower() == 'true'
        alpha = float(request.form.get('alpha', 0.05))
        
        # Cargar y validar datos
        df_clean, rows_dropped = load_and_validate_data(filepath, y_var, x_vars)
        
        # Preparar datos
        X = df_clean[x_vars].values
        y = df_clean[y_var].values
        
        # Ajustar modelo
        model = OLSValidation(alpha=alpha)
        model.rows_dropped = rows_dropped
        model.fit(X, y, intercept=intercept, var_names=x_vars)
        
        # Obtener resultados
        results_table = model.get_results_table()
        summary = model.get_summary(intercept)
        diagnostics = model.get_diagnostics()
        plot_base64 = model.plot_diagnostics()
        
        results = {
            'coefficients': results_table,
            'summary': summary,
            'diagnostics': diagnostics,
            'plot': plot_base64
        }

        file_id = save_results_to_file(convert_structure(results))
        
        session['file_id'] = file_id

        return jsonify(convert_structure({
            'success': True,
            'file_id': file_id,
            'coefficients': results_table,
            'summary': summary,
            'diagnostics': diagnostics,
            'plot': plot_base64
        }))
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/download/<file_id>')
def download_file(file_id):
    """Descarga resultados en Excel"""
    try:
        filepath = os.path.join("tmp_results", f"{file_id}.json")
        if not os.path.exists(filepath):
            return "No hay resultados disponibles", 404
    
        with open(filepath, "r", encoding="utf-8") as f:
            results = json.load(f)
        #results = session.get('results')
        if results is None:
            return "No hay resultados disponibles", 404

        
        # Crear Excel
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Coeficientes
            df_coef = pd.DataFrame(results['coefficients'])
            df_coef.to_excel(writer, sheet_name='Coeficientes', index=False)
            
            # Resumen
            df_summary = pd.DataFrame([results['summary']])
            df_summary.to_excel(writer, sheet_name='Resumen', index=False)
            
            # Diagnósticos
            diag_data = []
            for test_name, test_results in results['diagnostics'].items():
                if test_name == 'vif':
                    for vif_item in test_results.get('vif_values', []):
                        diag_data.append({
                            'Prueba': 'VIF',
                            'Variable': vif_item['variable'],
                            'Valor': vif_item['vif']
                        })
                else:
                    diag_data.append({
                        'Prueba': test_name,
                        'Estadístico': test_results.get('statistic', 'N/A'),
                        'P-value': test_results.get('p_value', 'N/A'),
                        'Conclusión': test_results['conclusion']
                    })
            
            df_diag = pd.DataFrame(diag_data)
            df_diag.to_excel(writer, sheet_name='Diagnósticos', index=False)
        
        output.seek(0)
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name='resultados_ols_completo.xlsx'
        )
        
    except Exception as e:
        return f"Error al generar archivo: {str(e)}", 500

def to_serializable(obj):
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def convert_structure(data):
    if isinstance(data, dict):
        return {k: convert_structure(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_structure(i) for i in data]
    else:
        return to_serializable(data)
    
def save_results_to_file(results):
    file_id = str(uuid.uuid4())
    filepath = os.path.join("tmp_results", f"{file_id}.json")

    # Crear carpeta por si algo falla
    os.makedirs("tmp_results", exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f)

    return file_id

'''@app.route('/download/<file_id>')
def download_results(file_id):
    filepath = os.path.join("tmp_results", f"{file_id}.json")
    if not os.path.exists(filepath):
        return "No hay resultados disponibles", 404
    
    with open(filepath, "r", encoding="utf-8") as f:
        results = json.load(f)'''


if __name__ == '__main__':
    app.run(debug=True, port=5000)