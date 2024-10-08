# %% --------------------------------------- Load Packages -------------------------------------------------------------
import numpy as np
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import shuffle
from scipy.linalg import sqrtm
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model
from skimage.transform import resize
from sklearn.model_selection import train_test_split

# %% --------------------------------------- Define FID ----------------------------------------------------------------
# Reference: https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/
# calculate frechet inception distance
def calculate_fid(model, images1, images2):
    # calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

# prepare the inception v3 model
model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))

# %% --------------------------------------- Calculate FID for Generator -----------------------------------------------
# scale an array of images to a new size
# Note: skimage will automatically change image range into [0, 1] after resizing
def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)*255
        # store
        images_list.append(new_image)
    return asarray(images_list)


# load generator
gen_path = 'bagan_gp_skin7_epoch49.h5'
generator = load_model(gen_path)

# Carga tu dataset
train_data = np.load(r'C:\Users\Miguel Palomino\Repository\dataset.npz')
images = train_data['x']
labels = train_data['y']

# División del dataset en entrenamiento y validación (test)
x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.25, shuffle=True, random_state=38)

# Asigna real_imgs y real_label a las imágenes y etiquetas de validación
real_imgs = x_val
real_label = y_val

# Calcula el número de imágenes disponibles por clase en el conjunto de validación
class_counts = {c: np.sum(real_label == c) for c in np.unique(real_label)}

# Puedes seleccionar el sample_size basado en el mínimo número de imágenes disponibles por clase
sample_size = min(class_counts.values())

print(f"Sample size set to {sample_size} based on available images per class")


# calculate FID for each class
n_classes = len(np.unique(real_label))
#sample_size = 1000
for c in range(n_classes):
    ########### get generated samples by class ###########
    label = np.ones(sample_size) * c
    noise = np.random.normal(0, 1, (sample_size, generator.input_shape[0][1]))
    print('Latent dimension:', generator.input_shape[0][1])
    gen_samples = generator.predict([noise, label])
    gen_samples = gen_samples*0.5 + 0.5

    ########### load real samples from training set ###########
    # gen_samples = np.load('x_train.npy')
    # gen_samples = np.load('y_train.npy')
    # gen_samples = gen_samples[gen_label == c]
    # shuffle(gen_samples)
    # gen_samples = gen_samples[:1000]

    ########### get real samples by class ###########
    real_samples = real_imgs[real_label == c]
    # shuffle(real_imgs)  # shuffle it or not
    # real_samples = real_samples[:1000]  # less calculation
    real_samples = real_samples.astype('float32') / 255.

    # resize images
    gen_samples = scale_images(gen_samples, (299,299,3))
    real_samples = scale_images(real_samples, (299,299,3))
    print('Scaled', gen_samples.shape, real_samples.shape)
    print('Scaled range for generated', np.min(gen_samples[0]), np.max(gen_samples[0]))
    print('Scaled range for real', np.min(real_samples[0]), np.max(real_samples[0]))

    # preprocess images
    gen_samples = preprocess_input(gen_samples)
    real_samples = preprocess_input(real_samples)
    print('Scaled range for generated', np.min(gen_samples[0]), np.max(gen_samples[0]))
    print('Scaled range for real', np.min(real_samples[0]), np.max(real_samples[0]))

    # calculate fid
    fid = calculate_fid(model, gen_samples, real_samples)
    print('>>FID(%d): %.3f' % (c, fid))
    print('-'*50)