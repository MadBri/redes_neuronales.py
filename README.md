# redes_neuronales.py

```python
import tensorflow as tf from tensorflow.keras.datasets import mnist from tensorflow.keras.models import Sequential from tensorflow.keras.layers import Dense, Flatten, Dropout from tensorflow.keras.callbacks import EarlyStopping from tensorflow.keras.utils import to_categorical import matplotlib.pyplot as plt

Cargar el dataset MNIST (dígitos manuscritos del 0 al 9)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

Normalizar los datos (escala 0-1)

x_train = x_train / 255.0 x_test = x_test / 255.0

One-hot encoding de las etiquetas

y_train = to_categorical(y_train, 10) y_test = to_categorical(y_test, 10)

Crear el modelo secuencial con capas adicionales

model = Sequential([ Flatten(input_shape=(28, 28)),      # Capa de entrada Dense(256, activation='relu'),     # Capa oculta 1 Dropout(0.3),                      # Regularización para evitar sobreajuste Dense(128, activation='relu'),     # Capa oculta 2 Dense(10, activation='softmax')    # Capa de salida ])

Compilar el modelo

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

Early stopping para evitar sobreentrenamiento

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

Entrenar el modelo

history = model.fit( x_train, y_train, epochs=15, batch_size=32, validation_split=0.1, callbacks=[early_stop], verbose=2 )

Evaluar el modelo en el conjunto de prueba

loss, accuracy = model.evaluate(x_test, y_test) print(f"\nPérdida en test: {loss:.4f}") print(f"Precisión en test: {accuracy:.4f}")

Graficar la precisión y la pérdida

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1) plt.plot(history.history['accuracy'], label='Entrenamiento') plt.plot(history.history['val_accuracy'], label='Validación') plt.title('Precisión por época') plt.xlabel('Epoch') plt.ylabel('Precisión') plt.legend()

plt.subplot(1, 2, 2) plt.plot(history.history['loss'], label='Entrenamiento') plt.plot(history.history['val_loss'], label='Validación') plt.title('Pérdida por época') plt.xlabel('Epoch') plt.ylabel('Pérdida') plt.legend()

plt.tight_layout() plt.show()
```
En este proyecto se implementó una red neuronal alimentada por el dataset MNIST. Este contiene imágenes en escala de grises de dígitos manuscritos del 0 al 9. Utilizamos Keras con TensorFlow para construir y entrenar el modelo.

Arquitectura utilizada: 

Entrada: imágenes de 28x28 píxeles, aplanadas en vectores de 784 valores.

Capas ocultas:

Una capa densa de 256 neuronas con activación ReLU.

Una segunda capa densa de 128 neuronas con activación ReLU.

Se utilizó Dropout para reducir el sobreajuste durante el entrenamiento.


Salida: capa softmax con 10 neuronas, una por cada dígito (0–9).


Características del entrenamiento:

Se aplicó normalización de imágenes para mejorar el aprendizaje.

Se utilizó EarlyStopping, que detiene el entrenamiento automáticamente si el modelo deja de mejorar.

El desempeño se evaluó en un conjunto de prueba, obteniendo una alta precisión.

Se generaron gráficas de pérdida y precisión por época, facilitando el análisis del entrenamiento.

