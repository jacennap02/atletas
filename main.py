import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score, recall_score
from sklearn.impute import SimpleImputer
import pickle

# Importaciones para MLP
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow.keras import layers, models, optimizers

# Ejemplo sencillo de modelo
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(100,)),
    layers.Dense(10, activation='softmax')
])

# Compilar el modelo
model.compile(
    optimizer=optimizers.Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Luego puedes entrenar con model.fit(), evaluar, etc.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

# Configuración para visualización
plt.style.use('ggplot')
sns.set(style="whitegrid")

# Configurar TensorFlow para mejor rendimiento
tf.config.experimental.enable_op_determinism()

# 1. Cargar los datos
print("1. CARGANDO DATOS")
print("-----------------")
# Modify the path to your actual data file location
# Using raw string to handle Windows backslashes
df = pd.read_csv(r"D:/Usuarios/jacennap02/Desktop/bombo/app/atletas.csv")
print(f"Dimensiones del dataset: {df.shape}")
print(f"Primeras filas del dataset:\n{df.head()}")
print("\n")

# 2. Exploración inicial y detección de valores faltantes
print("2. EXPLORACIÓN INICIAL Y VALORES FALTANTES")
print("-----------------------------------------")
print(f"Información del dataset:\n{df.info()}")
print(f"\nValores faltantes por columna:\n{df.isnull().sum()}")
print(f"\nEstadísticas descriptivas:\n{df.describe()}")

# Visualización de valores faltantes
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title('Mapa de valores faltantes')
plt.savefig('valores_faltantes.png')
plt.close()

# 3. Preprocesamiento: imputación de valores faltantes
print("\n3. PREPROCESAMIENTO: IMPUTACIÓN DE VALORES FALTANTES")
print("-------------------------------------------------")

# Imputar valores faltantes numéricas con la mediana
imputer = SimpleImputer(strategy='median')
df_numeric = df.select_dtypes(include=[np.number])
df[df_numeric.columns] = imputer.fit_transform(df_numeric)

# Para valores categóricos no numéricos, imputar con el valor más frecuente
cat_columns = df.select_dtypes(include=['object']).columns
for col in cat_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

print(f"Verificación después de imputación - valores faltantes:\n{df.isnull().sum()}")

# 4. Detección y tratamiento de outliers
print("\n4. DETECCIÓN Y TRATAMIENTO DE OUTLIERS")
print("-------------------------------------")

# Visualizar outliers con boxplots para algunas variables numéricas
plt.figure(figsize=(15, 10))
numeric_cols = df.select_dtypes(include=[np.number]).columns[:6]  # Primeras 6 columnas numéricas
for i, col in enumerate(numeric_cols):
    plt.subplot(2, 3, i+1)
    sns.boxplot(y=df[col])
    plt.title(f'Boxplot de {col}')
plt.tight_layout()
plt.savefig('outliers.png')
plt.close()

# Identificar y reportar outliers (método de IQR)
def detect_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    return outliers.shape[0]

print("Número de outliers detectados por variable:")
for col in df.select_dtypes(include=[np.number]).columns:
    num_outliers = detect_outliers(df, col)
    print(f"{col}: {num_outliers} outliers")

# Tratamiento de outliers - Para variables extremas (mayor a 150)
cols_to_cap = ['Peso', 'Valor Máximo de O2', '% Fibras Musculares Lentas', '% Fibras Musculares Rápidas', 'Edad']
for col in cols_to_cap:
    if col in df.columns:
        # Identificar valores extremos y capearlos
        upper_limit = df[col].quantile(0.99)
        lower_limit = df[col].quantile(0.01)
        df[col] = df[col].clip(lower=lower_limit, upper=upper_limit)

print("\nEstadísticas después de tratar outliers:")
print(df[cols_to_cap].describe())

# 5. Codificación de variables categóricas
print("\n5. CODIFICACIÓN DE VARIABLES CATEGÓRICAS")
print("--------------------------------------")

# Codificación de la variable 'Raza'
le = LabelEncoder()
df['Raza_encoded'] = le.fit_transform(df['raza'])
print(f"Mapeo de codificación para Raza:\n{pd.DataFrame({'Original': le.classes_, 'Encoded': range(len(le.classes_))})}")

# 6. Análisis de correlación y multicolinealidad
print("\n6. ANÁLISIS DE CORRELACIÓN Y MULTICOLINEALIDAD")
print("---------------------------------------------")

# Matriz de correlación
corr_matrix = df.select_dtypes(include=[np.number]).corr()
print("Matriz de correlación:")
print(corr_matrix['sexo'].sort_values(ascending=False))  # Correlaciones con la variable objetivo

# Visualización del mapa de calor de correlaciones
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Matriz de Correlación - Análisis de Multicolinealidad')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('correlacion.png')
plt.close()

# 7. Balance de clases
print("\n7. BALANCE DE CLASES")
print("------------------")
class_counts = df['sexo'].value_counts()
print(f"Distribución de clases:\n{class_counts}")
print(f"Proporción: {class_counts[0]/class_counts[1]:.2f}")

# Visualización del balance de clases
plt.figure(figsize=(8, 6))
sns.countplot(x='sexo', data=df)
plt.title('Balance de Clases')
plt.savefig('balance_clases.png')
plt.close()

# 8. Preparación de datos para modelado
print("\n8. PREPARACIÓN DE DATOS PARA MODELADO")
print("-----------------------------------")

# Seleccionar características y variable objetivo
X = df.drop(['sexo', 'raza'], axis=1)  # Excluimos 'Raza' original y usaremos 'Raza_encoded'
y = df['sexo']

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Escalar características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Dimensiones de conjunto de entrenamiento: {X_train.shape}")
print(f"Dimensiones de conjunto de prueba: {X_test.shape}")
print(f"Número de características (input_dim): {X_train_scaled.shape[1]}")

# 9. Entrenamiento y evaluación del modelo: Regresión Logística
print("\n9. MODELO: REGRESIÓN LOGÍSTICA")
print("-----------------------------")

# Modelo base
lr = LogisticRegression(random_state=42)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]

# Métricas de evaluación
print("Resultados de Regresión Logística:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_lr):.4f}")
print(f"Reporte de clasificación:\n{classification_report(y_test, y_pred_lr)}")

# Matriz de confusión
cm_lr = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Matriz de Confusión - Regresión Logística')
plt.ylabel('Actual')
plt.xlabel('Predicción')
plt.savefig('confusion_matrix_lr.png')
plt.close()

# Curva ROC
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)

plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, lw=2, label=f'ROC curve (area = {roc_auc_lr:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC - Regresión Logística')
plt.legend(loc="lower right")
plt.savefig('roc_curve_lr.png')
plt.close()

# 10. Modelo: Árbol de Decisión
print("\n10. MODELO: ÁRBOL DE DECISIÓN")
print("---------------------------")

# Modelo base
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_scaled, y_train)
y_pred_dt = dt.predict(X_test_scaled)
y_prob_dt = dt.predict_proba(X_test_scaled)[:, 1]

# Métricas de evaluación
print("Resultados del Árbol de Decisión:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_dt):.4f}")
print(f"Reporte de clasificación:\n{classification_report(y_test, y_pred_dt)}")

# Matriz de confusión
cm_dt = confusion_matrix(y_test, y_pred_dt)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Matriz de Confusión - Árbol de Decisión')
plt.ylabel('Actual')
plt.xlabel('Predicción')
plt.savefig('confusion_matrix_dt.png')
plt.close()

# Curva ROC
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_dt)
roc_auc_dt = auc(fpr_dt, tpr_dt)

plt.figure(figsize=(8, 6))
plt.plot(fpr_dt, tpr_dt, lw=2, label=f'ROC curve (area = {roc_auc_dt:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC - Árbol de Decisión')
plt.legend(loc="lower right")
plt.savefig('roc_curve_dt.png')
plt.close()

# 11. MODELO: MLP (Multi-Layer Perceptron) - NUEVO
print("\n11. MODELO: MLP (MULTI-LAYER PERCEPTRON)")
print("--------------------------------------")

# Configurar semilla para reproducibilidad
tf.random.set_seed(42)
np.random.seed(42)

# Obtener el número de características de entrada
input_dim = X_train_scaled.shape[1]

# Crear el modelo MLP Sequential con arquitectura x,16,8,1
mlp_model = Sequential([
    Dense(16, activation='relu', input_dim=input_dim, name='hidden_layer_1'),
    Dropout(0.2, name='dropout_1'),
    Dense(8, activation='relu', name='hidden_layer_2'),
    Dropout(0.2, name='dropout_2'),
    Dense(1, activation='sigmoid', name='output_layer')
])

# Compilar el modelo
mlp_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Mostrar la arquitectura del modelo
print("Arquitectura del modelo MLP:")
mlp_model.summary()

# Configurar early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# Entrenar el modelo
print("\nEntrenando el modelo MLP...")
history = mlp_model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# Realizar predicciones
y_pred_mlp_prob = mlp_model.predict(X_test_scaled).flatten()
y_pred_mlp = (y_pred_mlp_prob > 0.5).astype(int)

# Métricas de evaluación
print("\nResultados del MLP:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_mlp):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_mlp):.4f}")
print(f"Reporte de clasificación:\n{classification_report(y_test, y_pred_mlp)}")

# Matriz de confusión
cm_mlp = confusion_matrix(y_test, y_pred_mlp)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_mlp, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Matriz de Confusión - MLP')
plt.ylabel('Actual')
plt.xlabel('Predicción')
plt.savefig('confusion_matrix_mlp.png')
plt.close()

# Curva ROC para MLP
fpr_mlp, tpr_mlp, _ = roc_curve(y_test, y_pred_mlp_prob)
roc_auc_mlp = auc(fpr_mlp, tpr_mlp)

plt.figure(figsize=(8, 6))
plt.plot(fpr_mlp, tpr_mlp, lw=2, label=f'ROC curve (area = {roc_auc_mlp:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC - MLP')
plt.legend(loc="lower right")
plt.savefig('roc_curve_mlp.png')
plt.close()

# Visualizar el entrenamiento del MLP
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Pérdida del Modelo MLP')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Precisión del Modelo MLP')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(fpr_mlp, tpr_mlp, lw=2, label=f'MLP ROC (AUC = {roc_auc_mlp:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC - MLP')
plt.legend()

plt.tight_layout()
plt.savefig('mlp_training_metrics.png')
plt.close()

# 12. Comparación de curvas ROC (incluyendo MLP)
plt.figure(figsize=(10, 8))
plt.plot(fpr_lr, tpr_lr, lw=2, label=f'Regresión Logística (AUC = {roc_auc_lr:.2f})')
plt.plot(fpr_dt, tpr_dt, lw=2, label=f'Árbol de Decisión (AUC = {roc_auc_dt:.2f})')
plt.plot(fpr_mlp, tpr_mlp, lw=2, label=f'MLP (AUC = {roc_auc_mlp:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Comparación de Curvas ROC (Todos los Modelos)')
plt.legend(loc="lower right")
plt.savefig('roc_comparison_all_models.png')
plt.close()

# 13. Tuning de hiperparámetros (mantener solo para modelos sklearn)
print("\n12. TUNING DE HIPERPARÁMETROS")
print("---------------------------")

# Tuning para Regresión Logística
param_grid_lr = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga'],
    'penalty': ['l1', 'l2']
}

grid_lr = GridSearchCV(LogisticRegression(random_state=42), param_grid_lr, cv=5, scoring='accuracy')
grid_lr.fit(X_train_scaled, y_train)

print("Resultados de GridSearchCV para Regresión Logística:")
print(f"Mejores parámetros: {grid_lr.best_params_}")
print(f"Mejor score: {grid_lr.best_score_:.4f}")

# Evaluación del modelo optimizado
best_lr = grid_lr.best_estimator_
y_pred_best_lr = best_lr.predict(X_test_scaled)
y_prob_best_lr = best_lr.predict_proba(X_test_scaled)[:, 1]

print("\nResultados del modelo optimizado de Regresión Logística:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_best_lr):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_best_lr):.4f}")
print(f"Reporte de clasificación:\n{classification_report(y_test, y_pred_best_lr)}")

# Tuning para Árbol de Decisión
param_grid_dt = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_dt = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_dt, cv=5, scoring='accuracy')
grid_dt.fit(X_train_scaled, y_train)

print("\nResultados de GridSearchCV para Árbol de Decisión:")
print(f"Mejores parámetros: {grid_dt.best_params_}")
print(f"Mejor score: {grid_dt.best_score_:.4f}")

# Evaluación del modelo optimizado
best_dt = grid_dt.best_estimator_
y_pred_best_dt = best_dt.predict(X_test_scaled)
y_prob_best_dt = best_dt.predict_proba(X_test_scaled)[:, 1]

print("\nResultados del modelo optimizado de Árbol de Decisión:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_best_dt):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_best_dt):.4f}")
print(f"Reporte de clasificación:\n{classification_report(y_test, y_pred_best_dt)}")

# 14. Comparación final de modelos optimizados (incluyendo MLP)
print("\n13. COMPARACIÓN FINAL DE TODOS LOS MODELOS")
print("----------------------------------------")

# Calcular curvas ROC para modelos optimizados
fpr_best_lr, tpr_best_lr, _ = roc_curve(y_test, y_prob_best_lr)
roc_auc_best_lr = auc(fpr_best_lr, tpr_best_lr)

fpr_best_dt, tpr_best_dt, _ = roc_curve(y_test, y_prob_best_dt)
roc_auc_best_dt = auc(fpr_best_dt, tpr_best_dt)

plt.figure(figsize=(12, 8))
plt.plot(fpr_best_lr, tpr_best_lr, lw=2, label=f'Reg. Logística Optimizada (AUC = {roc_auc_best_lr:.2f})')
plt.plot(fpr_best_dt, tpr_best_dt, lw=2, label=f'Árbol Decisión Optimizado (AUC = {roc_auc_best_dt:.2f})')
plt.plot(fpr_mlp, tpr_mlp, lw=2, label=f'MLP (AUC = {roc_auc_mlp:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Comparación Final de Curvas ROC - Todos los Modelos')
plt.legend(loc="lower right")
plt.savefig('roc_comparison_final_all.png')
plt.close()

# Resumen comparativo
models_comparison = pd.DataFrame({
    'Modelo': ['Regresión Logística Base', 'Árbol de Decisión Base', 'MLP',
               'Regresión Logística Optimizada', 'Árbol de Decisión Optimizado'],
    'Accuracy': [accuracy_score(y_test, y_pred_lr), accuracy_score(y_test, y_pred_dt), 
                 accuracy_score(y_test, y_pred_mlp), accuracy_score(y_test, y_pred_best_lr), 
                 accuracy_score(y_test, y_pred_best_dt)],
    'Recall': [recall_score(y_test, y_pred_lr), recall_score(y_test, y_pred_dt),
               recall_score(y_test, y_pred_mlp), recall_score(y_test, y_pred_best_lr), 
               recall_score(y_test, y_pred_best_dt)],
    'AUC': [roc_auc_lr, roc_auc_dt, roc_auc_mlp, roc_auc_best_lr, roc_auc_best_dt]
})

print("\nComparación final de modelos:")
print(models_comparison)

# 15. Guardar TODOS los modelos en archivos pkl
print("\n14. GUARDANDO MODELOS EN ARCHIVOS PKL")
print("-----------------------------------")

# Guardar modelos sklearn
with open('logistic_regression_model.pkl', 'wb') as file:
    pickle.dump(best_lr, file)

with open('decision_tree_model.pkl', 'wb') as file:
    pickle.dump(best_dt, file)

# Guardar scaler para futuros usos
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

# Guardar modelo MLP (TensorFlow/Keras)
mlp_model.save('mlp_model.h5')
print("Modelo MLP guardado como 'mlp_model.h5'")

# Guardar también información del modelo MLP en pickle para metadata
mlp_info = {
    'architecture': 'Sequential',
    'layers': [
        {'type': 'Dense', 'units': 16, 'activation': 'relu', 'input_dim': input_dim},
        {'type': 'Dropout', 'rate': 0.2},
        {'type': 'Dense', 'units': 8, 'activation': 'relu'},
        {'type': 'Dropout', 'rate': 0.2},
        {'type': 'Dense', 'units': 1, 'activation': 'sigmoid'}
    ],
    'input_dim': input_dim,
    'optimizer': 'Adam',
    'learning_rate': 0.001,
    'loss': 'binary_crossentropy',
    'metrics': ['accuracy'],
    'best_accuracy': accuracy_score(y_test, y_pred_mlp),
    'best_auc': roc_auc_mlp
}

with open('mlp_model_info.pkl', 'wb') as file:
    pickle.dump(mlp_info, file)

# Guardar LabelEncoder para variables categóricas
with open('label_encoder.pkl', 'wb') as file:
    pickle.dump(le, file)

# Guardar información adicional para uso futuro
model_metadata = {
    'feature_names': list(X.columns),
    'target_name': 'Sexo',
    'input_dimension': input_dim,
    'categorical_columns': ['Raza'],
    'numerical_columns': list(df.select_dtypes(include=[np.number]).columns),
    'best_model': 'MLP' if roc_auc_mlp == max(roc_auc_lr, roc_auc_dt, roc_auc_mlp) else 'Logistic_Regression',
    'preprocessing_steps': ['imputation', 'outlier_treatment', 'encoding', 'scaling']
}

with open('model_metadata.pkl', 'wb') as file:
    pickle.dump(model_metadata, file)

print("Modelos guardados exitosamente:")
print("- logistic_regression_model.pkl")
print("- decision_tree_model.pkl") 
print("- mlp_model.h5 (modelo Keras)")
print("- mlp_model_info.pkl (información del MLP)")
print("- scaler.pkl")
print("- label_encoder.pkl")
print("- model_metadata.pkl")

# Conclusiones
print("\n15. CONCLUSIONES")
print("--------------")
print(f"""
RESUMEN DE HALLAZGOS:

1. Preprocesamiento:
   - Se identificaron y trataron valores faltantes en varias columnas.
   - Se detectaron outliers en variables como Peso, Valor Máximo de O2 y algunas características fisiológicas.
   - Se realizó codificación de variables categóricas como 'Raza'.

2. Análisis exploratorio:
   - El balance de clases muestra una distribución relativamente equilibrada entre categorías de sexo.
   - Se identificaron correlaciones interesantes entre variables biométricas y la variable objetivo.

3. Modelado (Comparación de Rendimiento):
   - Regresión Logística: AUC = {roc_auc_best_lr:.3f}
   - Árbol de Decisión: AUC = {roc_auc_best_dt:.3f}
   - MLP (Arquitectura {input_dim},16,8,1): AUC = {roc_auc_mlp:.3f}
   
4. Modelo MLP - Detalles:
   - Arquitectura: Sequential con {input_dim} entradas, 16 neuronas (capa oculta 1), 8 neuronas (capa oculta 2), 1 salida
   - Funciones de activación: ReLU en capas ocultas, Sigmoid en salida
   - Regularización: Dropout (0.2) en cada capa oculta
   - Optimizador: Adam con learning rate 0.001
   - Accuracy final: {accuracy_score(y_test, y_pred_mlp):.4f}""")