import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Cargar el modelo generador
generator = load_model('SANTIAGO_bagan_gp.h5')  # Asegúrate de usar el nombre correcto del archivo .h5

# Definir el número de clases y la dimensión latente
n_classes = 3  # Cambia esto según tu número de clases
latent_dim = 128  # Cambia esto según tu configuración de la dimensión latente

# Generar imágenes
def generate_images(generator, n_samples):
    # Crear vectores latentes aleatorios
    random_latent_vectors = np.random.normal(size=(n_samples, latent_dim))
    # Crear etiquetas aleatorias para las imágenes generadas
    random_labels = np.random.randint(0, n_classes, n_samples)
    # Generar imágenes
    generated_images = generator.predict([random_latent_vectors, random_labels])
    # Escalar las imágenes de vuelta al rango [0, 1]
    generated_images = (generated_images * 0.5) + 0.5
    return generated_images, random_labels

# Número de imágenes a generar
n_samples = 9

# Generar y mostrar imágenes
generated_images, labels = generate_images(generator, n_samples)

# Mostrar las imágenes generadas
plt.figure(figsize=(10, 10))
for i in range(n_samples):
    plt.subplot(5, 5, i + 1)
    plt.imshow(generated_images[i].reshape(64, 64, 3))
    plt.title(f"Class: {labels[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
