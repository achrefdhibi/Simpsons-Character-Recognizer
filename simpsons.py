import os
import caer
import canaro
import numpy as np
import cv2 as cv
import gc
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.models import load_model

# Paramètres
IMG_SIZE = (80, 80)
channels = 1
char_path = r'../input/the-simpsons-characters-dataset/simpsons_dataset'

# Création d'un dictionnaire de personnages
char_dict = {}
for char in os.listdir(char_path):
    char_dict[char] = len(os.listdir(os.path.join(char_path, char)))

# Trier le dictionnaire par ordre décroissant
char_dict = caer.sort_dict(char_dict, descending=True)

# Sélectionner les 10 premiers personnages
characters = []
count = 0
for i in char_dict:
    characters.append(i[0])
    count += 1
    if count >= 10:
        break

# Création des données d'entraînement
train = caer.preprocess_from_dir(char_path, characters, channels=channels, IMG_SIZE=IMG_SIZE, isShuffle=True)

# Séparation des données et des étiquettes
featureSet, labels = caer.sep_train(train, IMG_SIZE=IMG_SIZE)

# Normalisation des données
featureSet = caer.normalize(featureSet)

# Conversion des étiquettes en vecteurs binaires
labels = to_categorical(labels, len(characters))

# Création des ensembles d'entraînement et de validation
x_train, x_val, y_train, y_val = caer.train_test_split(featureSet, labels, val_ratio=.2)

# Suppression des variables pour libérer de la mémoire
del train
del featureSet
del labels
gc.collect()

# Définition des paramètres
BATCH_SIZE = 32
EPOCHS = 10

# Création du générateur de données
datagen = canaro.generators.imageDataGenerator()
train_gen = datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)

# Fonction pour charger un modèle pré-entraîné
def load_pretrained_model(model_path):
    return load_model(model_path)

# Fonction pour préparer une image
def prepare(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = cv.resize(image, IMG_SIZE)
    image = caer.reshape(image, IMG_SIZE, 1)
    return image

# Fonction pour prédire le personnage dans une image donnée
def predict_character(model, image):
    img = prepare(image)
    img = np.expand_dims(img, axis=0)
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions[0])
    predicted_character = characters[predicted_class]
    return predicted_character

# Entraînement du modèle
model = canaro.models.createSimpsonsModel(IMG_SIZE=IMG_SIZE, channels=channels, output_dim=len(characters), 
loss='binary_crossentropy', decay=1e-7, learning_rate=0.001, momentum=0.9,
nesterov=True)

# Entraînement du modèle
callbacks_list = [LearningRateScheduler(canaro.lr_schedule)]
training = model.fit(train_gen,
steps_per_epoch=len(x_train)//BATCH_SIZE,
epochs=EPOCHS,
validation_data=(x_val,y_val),
validation_steps=len(y_val)//BATCH_SIZE,
callbacks = callbacks_list)

# Chargement du modèle pré-entraîné (à remplacer par le chemin réel)
pretrained_model_path = 'C:C:\Users\Achref\Desktop\projectIA\Modeles\model.h5'

pretrained_model = load_pretrained_model(pretrained_model_path)

# Fonction pour la prédiction en temps réel à partir de la caméra
def real_time_prediction(model):
    cap = cv.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("impossible pour capturer la vidéo.")
            break

        predicted_character = predict_character(model, frame)

        cv.putText(frame, "Personnage : " + predicted_character, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.imshow("reconnaissance de personnages des simpson", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

# Appel de la fonction pour la prédiction en temps réel
real_time_prediction(pretrained_model)
