import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Configuración de la página
st.set_page_config(
    page_title="Regresión Lineal Simple",
    page_icon="📈",
    layout="wide"
)

# Título y descripción
st.title("📈 Aplicación de Regresión Lineal Simple")
st.markdown("""
Esta aplicación permite realizar análisis de regresión lineal simple para predecir 
la relación entre dos variables cuantitativas.
""")

# Sidebar para configuración
st.sidebar.header("Configuración")

# Sección para cargar datos
st.header("1. Cargar Datos")

# Opción para usar datos de ejemplo o cargar archivo
data_option = st.radio(
    "Seleccione la fuente de datos:",
    ["Usar datos de ejemplo", "Subir archivo CSV"]
)

df = None

if data_option == "Usar datos de ejemplo":
    # Crear datos de ejemplo basados en la imagen proporcionada
    data = {
        'X': [0, 1, 2, 3, 4],
        'Y': [1, 2, 3, 4, 5]
    }
    df = pd.DataFrame(data)
    st.success("Datos de ejemplo cargados correctamente")
    
else:
    uploaded_file = st.file_uploader(
        "Suba su archivo CSV", 
        type=['csv'],
        help="El archivo debe contener al menos dos columnas numéricas"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("Archivo cargado correctamente")
        except Exception as e:
            st.error(f"Error al cargar el archivo: {e}")

# Mostrar datos si están disponibles
if df is not None:
    st.subheader("Vista previa de los datos")
    st.dataframe(df.head())
    
    st.write(f"**Forma del dataset:** {df.shape}")
    
    # Selección de variables
    st.header("2. Selección de Variables")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Variable independiente (X)
        independent_var = st.selectbox(
            "Seleccione la variable independiente (X):",
            df.select_dtypes(include=[np.number]).columns
        )
    
    with col2:
        # Variable dependiente (Y)
        dependent_var = st.selectbox(
            "Seleccione la variable dependiente (Y):",
            df.select_dtypes(include=[np.number]).columns
        )
    
    # Validar que las variables sean diferentes
    if independent_var == dependent_var:
        st.warning("Por favor seleccione variables diferentes para X e Y")
    else:
        # Preparar datos para el modelo
        X = df[[independent_var]].values
        y = df[dependent_var].values
        
        # Entrenar modelo de regresión lineal
        model = LinearRegression()
        model.fit(X, y)
        
        # Predicciones
        y_pred = model.predict(X)
        
        # Métricas del modelo
        r2 = r2_score(y, y_pred)
        slope = model.coef_[0]
        intercept = model.intercept_
        
        # Mostrar resultados
        st.header("3. Resultados del Modelo")
        
        # Métricas en columnas
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Coeficiente de Determinación (R²)", f"{r2:.4f}")
        
        with col2:
            st.metric("Pendiente (β₁)", f"{slope:.4f}")
        
        with col3:
            st.metric("Intercepto (β₀)", f"{intercept:.4f}")
        
        # Ecuación del modelo
        st.subheader("Ecuación del Modelo")
        st.latex(f"Y = {slope:.2f}X + {intercept:.2f}")
        
        # Interpretación de R²
        st.subheader("Interpretación del Coeficiente de Determinación")
        st.info(f"""
        El valor de R² = {r2:.4f} indica que el **{r2*100:.2f}%** de la variabilidad 
        en la variable **{dependent_var}** es explicada por la variable **{independent_var}** 
        a través del modelo de regresión lineal.
        """)
        
        # Sección de predicciones
        st.header("4. Realizar Predicciones")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Predicción Individual")
            input_value = st.number_input(
                f"Ingrese el valor de {independent_var} para predecir:",
                value=float(X.mean()) if len(X) > 0 else 0.0
            )
            
            if st.button("Calcular Predicción"):
                prediction = model.predict([[input_value]])[0]
                st.success(f"Predicción para {independent_var} = {input_value}:")
                st.metric(
                    f"Valor predicho de {dependent_var}",
                    f"{prediction:.2f}"
                )
        
        with col2:
            st.subheader("Predicción Múltiple")
            st.markdown("Suba un archivo CSV con nuevos valores para predecir:")
            
            new_file = st.file_uploader(
                "Archivo con nuevos datos",
                type=['csv'],
                key="prediction_file"
            )
            
            if new_file is not None:
                try:
                    new_df = pd.read_csv(new_file)
                    if independent_var in new_df.columns:
                        new_X = new_df[[independent_var]].values
                        predictions = model.predict(new_X)
                        
                        result_df = new_df.copy()
                        result_df[f'Predicción_{dependent_var}'] = predictions
                        
                        st.success("Predicciones realizadas:")
                        st.dataframe(result_df)
                        
                        # Descargar resultados
                        csv = result_df.to_csv(index=False)
                        st.download_button(
                            label="Descargar predicciones (CSV)",
                            data=csv,
                            file_name="predicciones.csv",
                            mime="text/csv"
                        )
                    else:
                        st.error(f"El archivo debe contener la columna '{independent_var}'")
                except Exception as e:
                    st.error(f"Error al procesar el archivo: {e}")
        
        # Visualización
        st.header("5. Visualización del Modelo")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Gráfico de dispersión con línea de regresión
        scatter = ax.scatter(X, y, alpha=0.7, label='Datos reales', color='blue')
        
        # Línea de regresión
        x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_range = model.predict(x_range)
        ax.plot(x_range, y_range, color='red', linewidth=2, label='Línea de regresión')
        
        # Predicciones
        ax.scatter(X, y_pred, alpha=0.7, color='green', marker='x', label='Predicciones')
        
        ax.set_xlabel(independent_var)
        ax.set_ylabel(dependent_var)
        ax.set_title(f'Regresión Lineal: {dependent_var} vs {independent_var}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)

# Información educativa
with st.sidebar:
    st.header("📚 Información sobre Regresión Lineal")
    
    st.markdown("""
    **Regresión Lineal Simple**
    
    Modela la relación entre:
    - Variable independiente (X)
    - Variable dependiente (Y)
    
    **Ecuación:**
    \[ Y = \beta_0 + \beta_1X + \epsilon \]
    
    Donde:
    - $\beta_0$: Intercepto
    - $\beta_1$: Pendiente
    - $\epsilon$: Error
    """)
    
    st.markdown("""
    **Coeficiente R²**
    
    Mide qué tan bien el modelo explica la variabilidad de los datos:
    - R² = 1: Ajuste perfecto
    - R² = 0: Sin relación lineal
    - Valores cercanos a 1 indican mejor ajuste
    """)

# Pie de página
st.markdown("---")
st.markdown(
    "Aplicación desarrollada para análisis de regresión lineal simple"
)