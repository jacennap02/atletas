import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

# Importaciones de TensorFlow/Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model

# Importaciones de sklearn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report, roc_curve, auc)
from sklearn.impute import SimpleImputer

# Configuración de la página
st.set_page_config(
    page_title="MLP Model Training App",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🧠 Aplicación de Entrenamiento MLP - Clasificación de Atletas")
st.markdown("---")

# Funciones auxiliares
@st.cache_data
def load_and_preprocess_data():
    """Carga y preprocesa los datos"""
    try:
        # Intentar cargar desde diferentes ubicaciones posibles
        try:
            df = pd.read_csv("D:/Usuarios/jacennap02/Desktop/bombo/app/atletas.csv")
        except:
            st.error("No se pudo encontrar el archivo 'atletas.csv'. Por favor, asegúrate de que esté en el directorio de la aplicación.")
            return None, None, None, None, None
        
        # Preprocesamiento (similar al script original)
        # Imputar valores faltantes numéricas con la mediana
        imputer = SimpleImputer(strategy='median')
        df_numeric = df.select_dtypes(include=[np.number])
        df[df_numeric.columns] = imputer.fit_transform(df_numeric)
        
        # Para valores categóricos, imputar con el valor más frecuente
        cat_columns = df.select_dtypes(include=['object']).columns
        for col in cat_columns:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0])
        
        # Tratamiento de outliers
        cols_to_cap = ['Peso', 'Valor Máximo de O2', '% Fibras Musculares Lentas', 
                      '% Fibras Musculares Rápidas', 'Edad']
        for col in cols_to_cap:
            if col in df.columns:
                upper_limit = df[col].quantile(0.99)
                lower_limit = df[col].quantile(0.01)
                df[col] = df[col].clip(lower=lower_limit, upper=upper_limit)
        
        # Codificación de variables categóricas
        le = LabelEncoder()
        if 'raza' in df.columns:
            df['Raza_encoded'] = le.fit_transform(df['raza'])
        
        # Preparar X e y
        X = df.drop(['sexo', 'raza'] if 'raza' in df.columns else ['sexo'], axis=1)
        y = df['sexo']
        
        # División de datos
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Escalado
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, df
        
    except Exception as e:
        st.error(f"Error al cargar los datos: {str(e)}")
        return None, None, None, None, None

def create_mlp_model(input_dim, hidden_layers, activation_hidden, activation_output, 
                    optimizer_name, learning_rate, loss_function, metrics, dropout_rate):
    """Crea el modelo MLP con los parámetros especificados"""
    
    model = Sequential()
    
    # Primera capa oculta
    model.add(Dense(hidden_layers[0], 
                   activation=activation_hidden, 
                   input_dim=input_dim, 
                   name=f'hidden_layer_1'))
    
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate, name='dropout_1'))
    
    # Capas ocultas adicionales
    for i, units in enumerate(hidden_layers[1:], 2):
        model.add(Dense(units, 
                       activation=activation_hidden, 
                       name=f'hidden_layer_{i}'))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate, name=f'dropout_{i}'))
    
    # Capa de salida
    if activation_output == 'softmax':
        # Para clasificación multiclase
        output_units = len(np.unique(st.session_state.y_train)) if 'y_train' in st.session_state else 2
        model.add(Dense(output_units, activation=activation_output, name='output_layer'))
    else:
        # Para clasificación binaria
        model.add(Dense(1, activation=activation_output, name='output_layer'))
    
    # Configurar optimizador
    if optimizer_name == 'Adam':
        optimizer = optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'SGD':
        optimizer = optimizers.SGD(learning_rate=learning_rate)
    elif optimizer_name == 'RMSprop':
        optimizer = optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer_name == 'Adagrad':
        optimizer = optimizers.Adagrad(learning_rate=learning_rate)
    else:
        optimizer = optimizers.Adam(learning_rate=learning_rate)
    
    # Compilar modelo
    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=metrics
    )
    
    return model

def plot_training_history(history):
    """Visualiza el historial de entrenamiento"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Pérdida
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_title('Pérdida del Modelo')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Pérdida')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Precisión
    metric_key = 'accuracy' if 'accuracy' in history.history else list(history.history.keys())[1]
    axes[1].plot(history.history[metric_key], label=f'Training {metric_key.title()}', linewidth=2)
    axes[1].plot(history.history[f'val_{metric_key}'], label=f'Validation {metric_key.title()}', linewidth=2)
    axes[1].set_title(f'{metric_key.title()} del Modelo')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel(metric_key.title())
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_confusion_matrix(y_true, y_pred, title="Matriz de Confusión"):
    """Crea matriz de confusión"""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_title(title)
    ax.set_ylabel('Valor Real')
    ax.set_xlabel('Predicción')
    return fig

def plot_roc_curve(y_true, y_prob, title="Curva ROC"):
    """Crea curva ROC"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Tasa de Falsos Positivos')
    ax.set_ylabel('Tasa de Verdaderos Positivos')
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    return fig

def visualize_network_architecture(model):
    """Visualiza la arquitectura de la red neuronal"""
    try:
        # Crear visualización de la arquitectura
        plot_model(model, show_shapes=True, show_layer_names=True, rankdir='TB')
        st.success("Arquitectura del modelo generada exitosamente")
    except:
        # Alternativa de texto si no se puede generar la imagen
        st.subheader("Arquitectura del Modelo:")
        model.summary(print_fn=lambda x: st.text(x))

# Sidebar para configuración del modelo
st.sidebar.header("⚙️ Configuración del Modelo")

# Cargar datos al inicio
if 'data_loaded' not in st.session_state:
    with st.spinner("Cargando y preprocesando datos..."):
        X_train, X_test, y_train, y_test, df = load_and_preprocess_data()
        if X_train is not None:
            st.session_state.update({
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'df': df,
                'data_loaded': True,
                'input_dim': X_train.shape[1]
            })
        else:
            st.stop()

# Configuración de la arquitectura
st.sidebar.subheader("🏗️ Arquitecura de la Red")

# Capas ocultas
num_hidden_layers = st.sidebar.slider("Número de capas ocultas", 1, 5, 2)
hidden_layers = []
for i in range(num_hidden_layers):
    units = st.sidebar.slider(f"Neuronas en capa oculta {i+1}", 4, 128, 16 if i == 0 else 8)
    hidden_layers.append(units)

# Funciones de activación
st.sidebar.subheader("🔧 Funciones de Activación")
activation_hidden = st.sidebar.selectbox(
    "Activación capas ocultas (Regresión)",
    ['relu', 'tanh', 'sigmoid', 'elu', 'leaky_relu'],
    index=0
)

activation_output = st.sidebar.selectbox(
    "Activación capa salida (Clasificación)",
    ['sigmoid', 'softmax', 'tanh'],
    index=0
)

# Configuración del optimizador
st.sidebar.subheader("🚀 Optimizador")
optimizer_name = st.sidebar.selectbox(
    "Tipo de optimizador",
    ['Adam', 'SGD', 'RMSprop', 'Adagrad'],
    index=0
)

learning_rate = st.sidebar.slider(
    "Learning Rate",
    0.0001, 0.1, 0.001,
    format="%.4f"
)

# Función de pérdida
st.sidebar.subheader("📉 Función de Pérdida")
loss_function = st.sidebar.selectbox(
    "Loss Function",
    ['binary_crossentropy', 'categorical_crossentropy', 'sparse_categorical_crossentropy', 'mse', 'mae'],
    index=0
)

# Métricas
st.sidebar.subheader("📊 Métricas")
available_metrics = ['accuracy', 'precision', 'recall', 'auc']
selected_metrics = st.sidebar.multiselect(
    "Selecciona métricas",
    available_metrics,
    default=['accuracy']
)

# Configuración de entrenamiento
st.sidebar.subheader("🏃‍♂️ Entrenamiento")
epochs = st.sidebar.slider("Número de épocas", 10, 200, 50)
batch_size = st.sidebar.slider("Batch size", 16, 128, 32)
dropout_rate = st.sidebar.slider("Dropout rate", 0.0, 0.5, 0.2)

# Early stopping
use_early_stopping = st.sidebar.checkbox("Usar Early Stopping", True)
if use_early_stopping:
    patience = st.sidebar.slider("Patience", 5, 20, 10)

# Botón para entrenar
train_button = st.sidebar.button("🚀 Entrenar Modelo", type="primary")

# Contenido principal
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📊 Información del Dataset")
    if 'df' in st.session_state:
        st.write(f"**Dimensiones:** {st.session_state.df.shape}")
        st.write(f"**Características de entrada:** {st.session_state.input_dim}")
        
        # Mostrar primeras filas
        with st.expander("Ver primeras filas del dataset"):
            st.dataframe(st.session_state.df.head())
        
        # Balance de clases
        class_counts = st.session_state.df['sexo'].value_counts()
        st.write("**Balance de clases:**")
        st.bar_chart(class_counts)

with col2:
    st.header("🏗️ Configuración Actual")
    config_text = f"""
    **Arquitectura:**
    - Entrada: {st.session_state.input_dim if 'input_dim' in st.session_state else 'N/A'}
    - Capas ocultas: {' → '.join(map(str, hidden_layers))}
    - Salida: 1 (clasificación binaria)
    
    **Activaciones:**
    - Ocultas: {activation_hidden}
    - Salida: {activation_output}
    
    **Optimización:**
    - Optimizador: {optimizer_name}
    - Learning Rate: {learning_rate}
    - Loss: {loss_function}
    - Métricas: {', '.join(selected_metrics)}
    
    **Entrenamiento:**
    - Épocas: {epochs}
    - Batch Size: {batch_size}
    - Dropout: {dropout_rate}
    """
    st.text(config_text)

# Entrenamiento del modelo
if train_button and 'data_loaded' in st.session_state:
    st.header("🚀 Entrenamiento del Modelo")
    
    # Configurar semillas para reproducibilidad
    tf.random.set_seed(42)
    np.random.seed(42)
    
    with st.spinner("Entrenando modelo..."):
        try:
            # Crear modelo
            model = create_mlp_model(
                input_dim=st.session_state.input_dim,
                hidden_layers=hidden_layers,
                activation_hidden=activation_hidden,
                activation_output=activation_output,
                optimizer_name=optimizer_name,
                learning_rate=learning_rate,
                loss_function=loss_function,
                metrics=selected_metrics,
                dropout_rate=dropout_rate
            )
            
            # Mostrar arquitectura
            st.subheader("🏗️ Arquitectura del Modelo")
            model.summary(print_fn=lambda x: st.text(x))
            
            # Configurar callbacks
            callbacks = []
            if use_early_stopping:
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=patience,
                    restore_best_weights=True,
                    verbose=1
                )
                callbacks.append(early_stopping)
            
            # Entrenar modelo
            history = model.fit(
                st.session_state.X_train, st.session_state.y_train,
                validation_data=(st.session_state.X_test, st.session_state.y_test),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0
            )
            
            st.success("¡Modelo entrenado exitosamente!")
            
            # Guardar modelo en session state
            st.session_state.model = model
            st.session_state.history = history
            
        except Exception as e:
            st.error(f"Error durante el entrenamiento: {str(e)}")

# Mostrar resultados si el modelo ha sido entrenado
if 'model' in st.session_state and 'history' in st.session_state:
    st.header("📈 Resultados del Entrenamiento")
    
    # Gráficos de entrenamiento
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Historial de Entrenamiento")
        fig_history = plot_training_history(st.session_state.history)
        st.pyplot(fig_history)
    
    with col2:
        st.subheader("Arquitectura Visual")
        visualize_network_architecture(st.session_state.model)
    
    # Predicciones y evaluación
    st.header("🎯 Evaluación del Modelo")
    
    try:
        # Realizar predicciones
        y_pred_prob = st.session_state.model.predict(st.session_state.X_test).flatten()
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        # Calcular métricas
        accuracy = accuracy_score(st.session_state.y_test, y_pred)
        precision = precision_score(st.session_state.y_test, y_pred)
        recall = recall_score(st.session_state.y_test, y_pred)
        f1 = f1_score(st.session_state.y_test, y_pred)
        auc_score = roc_auc_score(st.session_state.y_test, y_pred_prob)
        
        # Mostrar métricas
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Accuracy", f"{accuracy:.4f}")
        with col2:
            st.metric("Precision", f"{precision:.4f}")
        with col3:
            st.metric("Recall", f"{recall:.4f}")
        with col4:
            st.metric("F1-Score", f"{f1:.4f}")
        with col5:
            st.metric("AUC", f"{auc_score:.4f}")
        
        # Visualizaciones
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Matriz de Confusión")
            fig_cm = plot_confusion_matrix(st.session_state.y_test, y_pred)
            st.pyplot(fig_cm)
        
        with col2:
            st.subheader("Curva ROC")
            fig_roc = plot_roc_curve(st.session_state.y_test, y_pred_prob)
            st.pyplot(fig_roc)
        
        # Reporte de clasificación
        with st.expander("Reporte Detallado de Clasificación"):
            report = classification_report(st.session_state.y_test, y_pred)
            st.text(report)
            
    except Exception as e:
        st.error(f"Error en la evaluación: {str(e)}")

# Footer
st.markdown("---")
st.markdown("💡 **Consejos:**")
st.markdown("""
- **ReLU** es generalmente buena para capas ocultas en problemas de regresión
- **Sigmoid** es adecuada para clasificación binaria en la capa de salida
- **Adam** suele ser un buen optimizador para empezar
- **binary_crossentropy** es ideal para clasificación binaria
- Usa **Early Stopping** para evitar sobreajuste
""")