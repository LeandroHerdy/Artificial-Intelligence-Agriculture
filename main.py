import sklearn
import numpy as np
import tensorflow as tf
from pathlib import Path

from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip
from tensorflow.keras.layers.experimental.preprocessing import RandomRotation
from tensorflow.keras.layers.experimental.preprocessing import RandomZoom
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision
from tensorflow.keras.metrics import Recall

# Carregando os Dados

diretorio_atual = Path.cwd()
print(diretorio_atual)


caminho_dados_treino = Path("fruits-360/Training")

caminho_dados_teste = Path("fruits-360/Test")
imagens_treino = list(caminho_dados_treino.glob("*/*"))


print(imagens_treino[925:936])
imagens_treino = list(map(lambda x: str(x), imagens_treino))
print(imagens_treino[925:936])
print(len(imagens_treino))

#Pré-Processamento dos Dados

def extrair_label(caminhho_imagem):
    return caminhho_imagem.split("/")[-2]


imagens_treino_labels = list(map(lambda x: extrair_label(x), imagens_treino))
print(imagens_treino_labels[840:846])

# LabelEncoder() tranforma a string em valor numerico.
encoder = LabelEncoder()
imagens_treino_labels = encoder.fit_transform(imagens_treino_labels)
print(imagens_treino_labels[840:846])

# Aplica uma rot-encoding no labels
imagens_treino_labels = tf.keras.utils.to_categorical(imagens_treino_labels)
imagens_treino_labels[840:846]

x_treino, x_valid, y_treino, y_valid = train_test_split(imagens_treino, imagens_treino_labels)


img_size = 224
resize = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.Resizing(img_size, img_size)])

data_augmentation = tf.keras.Sequential([RandomFlip("horizontal"),
                                         RandomRotation(0.2),
                                         RandomZoom(height_factor=(-0.3, -0.2))])

# Preparando os Dados

batch_size = 32
autotune = tf.data.experimental.AUTOTUNE


def carrega_transforma(image, label):
    image = tf.io.read_file(image)
    image = tf.io.decode_jpeg(image, channels=3)
    return image, label


def prepara_dataset(path, labels, train = True):
    image_paths = tf.convert_to_tensor(path)
    labels = tf.convert_to_tensor(labels)
    image_dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    label_dataset = tf.data.Dataset.from_tensor_slices(labels)
    dataset = tf.data.Dataset.zip((image_dataset, label_dataset))
    dataset = dataset.map(lambda image, label: carrega_transforma(image, label))
    dataset = dataset.map(lambda image, label: (resize(image), label), num_parallel_calls=autotune)
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size)
    if train:
        dataset = dataset.map(lambda image, label: (data_augmentation(image), label), num_parallel_calls=autotune)
    dataset = dataset.repeat()
    return dataset


dataset_treino = prepara_dataset(x_treino, y_treino)

image, label = next(iter(dataset_treino))
print(image.shape)
print(label.shape)

print(encoder.inverse_transform(np.argmax(label, axis=1))[0])
plt.imshow((image[0].numpy()/255).reshape(224, 224, 3))

dataset_valid = prepara_dataset(x_valid, y_valid, train=False)
imagem, label = next(iter(dataset_valid))
print(imagem.shape)
print(label.shape)

# Contrução do Modelo

modelo_pre = EfficientNetB3(input_shape=(224, 224, 3), include_top=False)

modelo = tf.keras.Sequential([modelo_pre, tf.keras.layers.GlobalAveragePooling2D(),
                              tf.keras.layers.Dense(131, activation='softmax')])

print(modelo.summary())

lr = 0.001
beta1 = 0.9
beta2 = 0.999
ep = 1e-07

modelo.complile(optimizer=Adam(learning_rate=lr,
                                 beta_1=beta1,
                                 beta_2=beta2,
                                 epsilon=ep),
                loss='categorical_crossentropy',
                metrics=['accuracy', Precision(name='precision'), Recall(name='recall')])

history = modelo.fit(dataset_treino,
                      steps_por_epoch=len(x_treino)//batch_size,
                      epochs=1,
                      validation_steps=len(y_treino)//batch_size)

modelo.layers[0].trainable = False

checkpoint = tf.keras.callbacks.ModelCheckpoint('modelo/melhor_modelo.h5',
                                                verbose=1,
                                                save_best=True,
                                                save_weights_only=True)

early_stop = tf.keras.callbacks.EarlyStopping(patiense=4)

print(modelo.sumary())

history = modelo.fit(dataset_treino,
                     steps_por_epoch=len(x_treino)//batch_size,
                     epochs=6,
                     validation_data=dataset_valid,
                     validation_steps=len(y_treino)//batch_size,
                     callback=[checkpoint, early_stop])

# Avaliação do Modelo

modelo.layers[0].trainable = True
modelo.load_weights("modelo/melhor_modelo.h5")

# Carregando e preparando os dados de teste

caminho_imagens_teste = list(caminho_dados_teste.glob('*/*'))
imagens_teste = list(map(lambda x: str(x), caminho_imagens_teste))
imagens_teste_labels = list(map(lambda x: extrair_label(x), imagens_teste))
imagens_teste_labels = encoder.fit_transform(imagens_teste_labels)
imagens_teste_labels = tf.keras.utils.to_categorical(imagens_teste_labels)
test_image_paths = tf.convert_to_tensor(imagens_teste)
test_image_labels = tf.convert_to_tensor(imagens_teste_labels)

def decode_imagens(image, label):
    image = tf.io.read_fale(image)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224], method='bilinear')
    return image, label

dataset_teste = (tf.data.Dataset
                  .from_tensor_slices(imagens_teste, imagens_teste_labels)
                  .map(decode_imagens)
                  .batch(batch_size))

imagem, label = next(iter(dataset_teste))
print(imagem.shape)
print(label.shape)

print(encoder.inverse_transform(np.argmax(label, axis=1))[0])
plt.imshow(imagem[0].numpy()/255).reshape(224, 224, 3)

# Avaliar o modelo

loss, acc, prec, rec = modelo.evaluate(dataset_teste)

print("Acurácia: ", acc)
print("Precision: ", prec)
print("Recall: ", rec)

# Previsão com o Modelo Treinado


def carregar_nova_imagem(image_path):
    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224], method="bilinear")
    plt.imshow(image.numpy()/255)
    image = tf.expand_dims(image, 0)
    return image


def faz_previsao(image_path, model, enc):
    image = carregar_nova_imagem(image_path)
    prediction = model.predict(image)
    pred = np.argmax(prediction, axis=1)
    return enc.inverse_transform(pred[0])


print(faz_previsao("imagens/imagem1.jpg", modelo, encoder))


print(faz_previsao("imagens/imagem2.jpg", modelo, encoder))
