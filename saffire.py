import streamlit as st
import zipfile
import os
import shutil
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Déclaration globale du modèle
model = None
model_path = "saved_model.h5"
class_names = []

# Configuration de la page Streamlit
st.set_page_config(page_title="SAFFIRE - Fire and smoke Detection System", layout="wide")
st.title("SAFFIRE Detection System")
# Ajouter une image de fond
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("C:.\");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Sidebar Mode Selection
st.sidebar.header("Configuration")
mode = st.sidebar.radio("Select Mode:", ["Automatic", "Manual"])

# Chargement des données
st.markdown("# Chargement des Données")
train_data = st.file_uploader("Importer les données d'entraînement (ZIP)", type=["zip"])
train_dir = "temp_train_dir"

def extract_zip(zip_file, extract_to):
    if os.path.exists(extract_to):
        shutil.rmtree(extract_to)
    os.makedirs(extract_to)
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        for member in zip_ref.namelist():
            # Ignorer les fichiers systèmes et dossiers vides
            if '__MACOSX' in member or member.endswith('/'):
                continue
            # Supprimer le dossier parent
            parts = member.split('/')
            extracted_path = os.path.join(extract_to, *parts[1:])
            os.makedirs(os.path.dirname(extracted_path), exist_ok=True)
            with zip_ref.open(member) as source, open(extracted_path, "wb") as target:
                shutil.copyfileobj(source, target)

# Configuration des paramètres en fonction du mode sélectionné
if mode == "Manual":
    optimizer_choice = st.sidebar.selectbox("Choisissez l'optimiseur:", ("Adam", "SGD", "RMSprop"))
    learning_rate = st.sidebar.number_input("Taux d'apprentissage:", min_value=0.0001, max_value=1.0, value=0.001, step=0.0001, format="%f")
    epochs = st.sidebar.number_input("Nombre d'epochs:", min_value=1, max_value=100, value=30, step=1)
    batch_size = st.sidebar.number_input("Taille des batchs:", min_value=1, max_value=128, value=32, step=1)
    num_conv_layers = st.sidebar.slider("Nombre de couches convolutives:", min_value=1, max_value=5, value=3)
    filters_per_layer = [st.sidebar.number_input(f"Filtres pour la couche {i+1}:", min_value=8, max_value=512, value=16 * (2**i), step=8) for i in range(num_conv_layers)]
    dense_units = st.sidebar.number_input("Neurones dans la couche Dense:", min_value=8, max_value=512, value=64, step=8)
    dropout_rate = st.sidebar.slider("Taux de Dropout:", min_value=0.0, max_value=0.9, value=0.5, step=0.05)
else:
    optimizer_choice = "Adam"
    learning_rate = 0.001
    epochs = 20
    batch_size = 32
    num_conv_layers = 3
    filters_per_layer = [32, 64, 128]
    dense_units = 128
    dropout_rate = 0.3

# Générateur de données
if st.button("Démarrer l'entraînement"):
    if train_data is not None:
        extract_zip(train_data, train_dir)
        datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
        train_generator = datagen.flow_from_directory(train_dir, target_size=(299, 299), batch_size=batch_size, class_mode='categorical', subset='training')
        val_generator = datagen.flow_from_directory(train_dir, target_size=(299, 299), batch_size=batch_size, class_mode='categorical', subset='validation')
        num_classes = len(train_generator.class_indices)
        class_names = list(train_generator.class_indices.keys())

        # Correction de l'initialisation de l'optimiseur
        if optimizer_choice == "Adam":
            optimizer = Adam(learning_rate=learning_rate)
        elif optimizer_choice == "SGD":
            optimizer = SGD(learning_rate=learning_rate)
        elif optimizer_choice == "RMSprop":
            optimizer = RMSprop(learning_rate=learning_rate)

        model = Sequential()
        for filters in filters_per_layer:
            model.add(Conv2D(filters, (3, 3), activation='relu', kernel_regularizer=l2(0.0001), padding='same'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(GlobalAveragePooling2D())
        model.add(Dense(dense_units, activation='relu', kernel_regularizer=l2(0.0001)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(train_generator, epochs=epochs, validation_data=val_generator)
        model.save(model_path)
        st.success(" Entraînement terminé et modèle sauvegardé !")



        # Affichage des courbes de convergence
        st.markdown("# Courbes de Convergence")
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        ax[0].plot(history.history['loss'], label='Perte Entraînement')
        ax[0].plot(history.history['val_loss'], label='Perte Validation')
        ax[0].set_title('Courbe de la Perte')
        ax[0].legend()

        ax[1].plot(history.history['accuracy'], label='Précision Entraînement')
        ax[1].plot(history.history['val_accuracy'], label='Précision Validation')
        ax[1].set_title('Courbe de Précision')
        ax[1].legend()
        st.pyplot(fig)

        # Matrice de confusion
        st.markdown("# Matrice de Confusion")
        if val_generator.samples > 0:
            # Générer les prédictions
            val_preds = model.predict(val_generator, verbose=0)
            y_pred = np.argmax(val_preds, axis=1)
            y_true = val_generator.classes

            # Vérification des classes
            if not class_names:
                class_names = list(val_generator.class_indices.keys())

            # Création de la matrice de confusion
            cm = confusion_matrix(y_true, y_pred)
            st.write("**Vraies classes :**", y_true[:10])
            st.write("**Classes prédites :**", y_pred[:10])
            fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
            disp.plot(ax=ax_cm, cmap='Blues', colorbar=True)
            ax_cm.set_title("Matrice de Confusion")
            st.pyplot(fig_cm)
        else:
            st.warning("Aucune donnée de validation disponible pour générer la matrice de confusion.")


        


# Prédiction d'une image
st.markdown("#Prédiction sur une Image")
image_file = st.file_uploader("Choisissez une image pour prédiction", type=["jpg", "png"])
if image_file and st.button("Prédire"):
    if model is None and os.path.exists(model_path):
        model = load_model(model_path)
    if model is not None:
        img = load_img(image_file, target_size=(299, 299))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        prediction = model.predict(img_array)[0]
        predicted_class = np.argmax(prediction)
        predicted_label = class_names[predicted_class] if class_names else str(predicted_class)
        confidence = round(prediction[predicted_class] * 100, 2)
        st.image(image_file, caption="🖼️ Image à prédire")
        st.write(f"**Classe prédite : {predicted_label}** avec une confiance de **{confidence}%**")
    else:
        st.error("❗ Aucun modèle chargé. Veuillez entraîner ou charger un modèle.")

#ok





























