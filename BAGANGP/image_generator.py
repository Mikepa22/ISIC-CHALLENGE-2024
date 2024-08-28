import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os
from PIL import Image 

# Cargar el modelo generador
generator = load_model('bagan_gp_skin7_epoch49.h5')

# Definir el número de clases y la dimensión latente
n_classes = 2  # Cambia esto según tu número de clases
latent_dim = 128  # Cambia esto según tu configuración de la dimensión latente

# GRAFICAR IMAGENES GENERADAS
# Cargar las imágenes y etiquetas del dataset
# train_data = np.load(r'C:\Users\Miguel Palomino\Repository\dataset.npz')
# images = train_data['x']
# labels = train_data['y']

# # Extraer las imágenes reales de la clase 1
# real_images_class_1 = images[labels == 1][:5]  # Selecciona las primeras 5 imágenes de la clase 1

# # Función para generar imágenes de una clase específica
# def generate_images_by_class(generator, n_samples, class_label):
#     # Crear vectores latentes aleatorios
#     random_latent_vectors = np.random.normal(size=(n_samples, latent_dim))
#     # Crear etiquetas para la clase específica
#     labels = np.full((n_samples,), class_label)
#     # Generar imágenes
#     generated_images = generator.predict([random_latent_vectors, labels])
#     # Escalar las imágenes de vuelta al rango [0, 1]
#     generated_images = (generated_images * 0.5) + 0.5
#     return generated_images

# # Generar 5 imágenes de la clase 1
# n_samples = 5
# generated_images_class_1 = generate_images_by_class(generator, n_samples, class_label=1)

# # Crear la figura para mostrar las imágenes
# plt.figure(figsize=(10, 4))

# # Mostrar las imágenes reales en la primera fila
# for i in range(n_samples):
#     plt.subplot(2, 5, i + 1)
#     plt.imshow(real_images_class_1[i])
#     plt.title("Real")
#     plt.axis('off')

# # Mostrar las imágenes generadas en la segunda fila
# for i in range(n_samples):
#     plt.subplot(2, 5, i + 6)
#     plt.imshow(generated_images_class_1[i])
#     plt.title("Generated")
#     plt.axis('off')

# plt.tight_layout()
# plt.show()  

### CREAR DATASET
# Función para generar imágenes de una clase específica
def generate_images_by_class(generator, n_samples, class_label):
    random_latent_vectors = np.random.normal(size=(n_samples, latent_dim))
    labels = np.full((n_samples,), class_label)
    generated_images = generator.predict([random_latent_vectors, labels])
    generated_images = (generated_images * 0.5) + 0.5  # Escalar de vuelta al rango [0, 1]
    return generated_images

# Generar 307 imágenes de la clase 1
n_samples = 307
generated_images_class_1 = generate_images_by_class(generator, n_samples, class_label=1)

# Crear carpeta para guardar las imágenes generadas
output_dir = 'dataset'
os.makedirs(output_dir, exist_ok=True)

# Archivo de texto para almacenar los nombres de las imágenes y sus clases
txt_filename = 'image_labels.txt'
with open(txt_filename, 'w') as file:
    for i in range(n_samples):
        # Asignar nombre a la imagen
        img_name = f"{i+1}.jpg"
        img_path = os.path.join(output_dir, img_name)
        
        # Convertir la imagen a formato PIL y guardarla en formato JPG
        img = Image.fromarray((generated_images_class_1[i] * 255).astype(np.uint8))
        imgr = img.resize((128, 128), Image.LANCZOS)
        imgr.save(img_path)
        
        # Escribir el nombre de la imagen y su clase en el archivo de texto
        file.write(f"{img_name} 1\n")

print(f"307 imágenes generadas y guardadas en la carpeta '{output_dir}'.")
print(f"Archivo de texto con etiquetas guardado como '{txt_filename}'.")

