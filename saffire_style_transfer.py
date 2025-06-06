import streamlit as st
import base64
import zipfile
import os
import shutil
import pandas as pd
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from PIL import Image
import io

# Déclaration globale du modèle
model = None
model_path = "saved_model.h5"
class_names = []

# Configuration de la page Streamlit
st.set_page_config(page_title="SAFFIRE - Fire and smoke Detection System", layout="wide")

# Titre principal
st.markdown(
    """
    <h1 style="text-align: center; color: black;">SAFFIRE Detection System</h1>
    """,
    unsafe_allow_html=True
)

# Sous-titre
st.markdown(
    """
    <h2 style="text-align: center; color: black;">Powered by AI and TensorFlow</h2>
    """,
    unsafe_allow_html=True
)

# Texte centré
st.markdown(
    """
    <p style="text-align: center; color: black;">Détection avancée de fumée et de feu pour une sécurité optimale</p>
    """,
    unsafe_allow_html=True
)

# Configuration de l'image de fond
background_image_path = "background.jpg"

# Vérifier si l'image de fond existe
if os.path.exists(background_image_path):
    with open(background_image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{encoded_string});
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Sidebar Mode Selection
st.sidebar.header("Configuration")

# Nouveau sélecteur de mode principal
main_mode = st.sidebar.radio("Sélectionnez le module:", ["Classification", "Transfert de Style"])

if main_mode == "Classification":
    mode = st.sidebar.radio("Select Mode:", ["Automatic", "Manual"])
    
    # Ajouter un logo dans la sidebar
    if os.path.exists("logo.jpg"):
        st.sidebar.image("logo.jpg", width=150, caption="SAFFIRE")

    # Chargement des données
    st.markdown("## Chargement des Données")
    train_data = st.file_uploader("Importer les données d'entraînement (ZIP)", type=["zip"])
    train_dir = "temp_train_dir"

    def extract_zip(zip_file, extract_to):
        if os.path.exists(extract_to):
            shutil.rmtree(extract_to)
        os.makedirs(extract_to)
        
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            # Extraire tout d'abord dans un dossier temporaire
            temp_extract = extract_to + "_temp"
            zip_ref.extractall(temp_extract)
            
            # Identifier la structure du dataset
            items = os.listdir(temp_extract)
            
            # Cas 1: Structure directe (dataset.zip/classe1/, classe2/, ...)
            if all(os.path.isdir(os.path.join(temp_extract, item)) for item in items if not item.startswith('.')):
                # Structure correcte, copier directement
                for item in items:
                    if not item.startswith('.') and not item == '__MACOSX':
                        source_path = os.path.join(temp_extract, item)
                        dest_path = os.path.join(extract_to, item)
                        if os.path.isdir(source_path):
                            shutil.copytree(source_path, dest_path)
                        else:
                            shutil.copy2(source_path, dest_path)
            
            # Cas 2: Dossier parent en trop (dataset.zip/parent_folder/classe1/, classe2/, ...)
            else:
                # Chercher le premier dossier qui contient des sous-dossiers
                parent_folder = None
                for item in items:
                    item_path = os.path.join(temp_extract, item)
                    if os.path.isdir(item_path) and not item.startswith('.') and item != '__MACOSX':
                        # Vérifier si ce dossier contient des sous-dossiers (classes)
                        sub_items = os.listdir(item_path)
                        if any(os.path.isdir(os.path.join(item_path, sub)) for sub in sub_items if not sub.startswith('.')):
                            parent_folder = item_path
                            break
                
                if parent_folder:
                    # Copier le contenu du dossier parent vers extract_to
                    for item in os.listdir(parent_folder):
                        if not item.startswith('.') and not item == '__MACOSX':
                            source_path = os.path.join(parent_folder, item)
                            dest_path = os.path.join(extract_to, item)
                            if os.path.isdir(source_path):
                                shutil.copytree(source_path, dest_path)
                            else:
                                shutil.copy2(source_path, dest_path)
                else:
                    # Structure non reconnue, copier tout
                    for item in items:
                        if not item.startswith('.') and not item == '__MACOSX':
                            source_path = os.path.join(temp_extract, item)
                            dest_path = os.path.join(extract_to, item)
                            if os.path.isdir(source_path):
                                shutil.copytree(source_path, dest_path)
                            else:
                                shutil.copy2(source_path, dest_path)
            
            # Nettoyer le dossier temporaire
            shutil.rmtree(temp_extract)
            
            # Afficher la structure détectée
            st.info(f"📁 Structure détectée : {len([d for d in os.listdir(extract_to) if os.path.isdir(os.path.join(extract_to, d))])} classes trouvées")
            classes_found = [d for d in os.listdir(extract_to) if os.path.isdir(os.path.join(extract_to, d))]
            st.write("Classes détectées :", ", ".join(classes_found))

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
        activation_function = st.sidebar.selectbox("Choisissez la fonction d'activation:", ("relu", "sigmoid", "tanh", "softmax"))
    else:
        optimizer_choice = "Adam"
        learning_rate = 0.001
        epochs = 20
        batch_size = 32
        num_conv_layers = 3
        filters_per_layer = [32, 64, 128]
        dense_units = 128
        dropout_rate = 0.3
        activation_function = "relu"

    # Générateur de données
    if st.button("Démarrer l'entraînement"):
        if train_data is not None:
            extract_zip(train_data, train_dir)
            datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
            train_generator = datagen.flow_from_directory(train_dir, target_size=(128, 128), batch_size=batch_size, class_mode='categorical', subset='training')
            val_generator = datagen.flow_from_directory(train_dir, target_size=(128, 128), batch_size=batch_size, class_mode='categorical', subset='validation')
            num_classes = len(train_generator.class_indices)
            class_names = list(train_generator.class_indices.keys())

            # initialisation de l'optimiseur
            if optimizer_choice == "Adam":
                optimizer = Adam(learning_rate=learning_rate)
            elif optimizer_choice == "SGD":
                optimizer = SGD(learning_rate=learning_rate)
            elif optimizer_choice == "RMSprop":
                optimizer = RMSprop(learning_rate=learning_rate)

            model = Sequential()
            for filters in filters_per_layer:
                model.add(Conv2D(filters, (3, 3), activation=activation_function, kernel_regularizer=l2(0.0001), padding='same'))
                model.add(BatchNormalization())
                model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(GlobalAveragePooling2D())
            model.add(Dense(dense_units, activation=activation_function, kernel_regularizer=l2(0.0001)))
            model.add(Dropout(dropout_rate))
            model.add(Dense(num_classes, activation='softmax'))

            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
            
            with st.spinner('Entraînement en cours...'):
                history = model.fit(train_generator, epochs=epochs, validation_data=val_generator)
            
            model.save(model_path)
            st.success("✅ Entraînement terminé et modèle sauvegardé !")

            # Affichage des courbes de convergence
            st.markdown("## Courbes de Convergence")
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
            st.markdown("## Matrice de Confusion")
            if val_generator.samples > 0:
                val_preds = model.predict(val_generator, verbose=0)
                y_pred = np.argmax(val_preds, axis=1)
                y_true = val_generator.classes

                if not class_names:
                    class_names = list(val_generator.class_indices.keys())

                cm = confusion_matrix(y_true, y_pred)
                fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
                disp.plot(ax=ax_cm, cmap='Blues', colorbar=True)
                ax_cm.set_title("Matrice de Confusion")
                st.pyplot(fig_cm)
            else:
                st.warning("Aucune donnée de validation disponible pour générer la matrice de confusion.")

    # Prédiction d'une image
    st.markdown("## Prédiction sur une Image")
    image_file = st.file_uploader("Choisissez une image pour prédiction", type=["jpg", "png"])
    if image_file and st.button("Prédire"):
        if model is None and os.path.exists(model_path):
            model = load_model(model_path)
        if model is not None:
            img = load_img(image_file, target_size=(128, 128))
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

# MODULE DE TRANSFERT DE STYLE
elif main_mode == "Transfert de Style":
    st.markdown("## 🎨 Module de Transfert de Style")
    
    # Paramètres du transfert de style dans la sidebar
    st.sidebar.markdown("### Paramètres du Style")
    style_weight = st.sidebar.slider("Poids du style", 1e-2, 1e6, 1e4, step=1e3, format="%.0e")
    content_weight = st.sidebar.slider("Poids du contenu", 1e0, 1e4, 1e3, step=1e2, format="%.0e")
    iterations = st.sidebar.number_input("Nombre d'itérations", min_value=10, max_value=1000, value=100, step=10)
    
    # Paramètres de débogage
    st.sidebar.markdown("### 🔧 Paramètres Avancés")
    debug_mode = st.sidebar.checkbox("Mode débogage", value=True)
    learning_rate = st.sidebar.slider("Taux d'apprentissage", 0.001, 0.1, 0.01, step=0.001)
    show_diagnostics = st.sidebar.checkbox("Diagnostic détaillé", value=False)
    
    def diagnose_image(image_file, name):
        """Diagnostique une image uploadée"""
        img = Image.open(image_file)
        img_array = np.array(img)
        
        st.write(f"**📊 Diagnostic de l'image {name}:**")
        st.write(f"- Format original: {img.format}")
        st.write(f"- Mode: {img.mode}")
        st.write(f"- Taille: {img.size}")
        st.write(f"- Forme du array: {img_array.shape}")
        st.write(f"- Type de données: {img_array.dtype}")
        st.write(f"- Valeurs min/max: {img_array.min()} / {img_array.max()}")
        
        # Afficher un histogramme simple
        if len(img_array.shape) == 3:
            st.write(f"- Moyennes par canal: R={img_array[:,:,0].mean():.2f}, G={img_array[:,:,1].mean():.2f}, B={img_array[:,:,2].mean():.2f}")
        
        return img_array

    # Fonctions pour le transfert de style
    @st.cache_resource
    def load_vgg_model():
        """Charge le modèle VGG19 pré-entraîné"""
        vgg = VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False
        return vgg

    def preprocess_image(image_path, max_dim=512):
        """Préprocesse une image pour le transfert de style"""
        if isinstance(image_path, str):
            img = tf.io.read_file(image_path)
            img = tf.image.decode_image(img, channels=3)
        else:
            # Si c'est un objet file de streamlit
            img = Image.open(image_path)
            img = np.array(img)
            
            # Gérer les différents formats d'image
            if len(img.shape) == 2:  # Image en niveaux de gris
                img = np.stack([img] * 3, axis=-1)
            elif len(img.shape) == 3 and img.shape[-1] == 4:  # RGBA
                img = img[:, :, :3]  # Supprimer le canal alpha
            
            # S'assurer que les valeurs sont dans la bonne plage
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            elif img.dtype == np.float32 and img.max() > 1.0:
                img = img / 255.0
            
            img = tf.constant(img, dtype=tf.float32)
        
        # Convertir en float32 si ce n'est pas déjà fait
        if img.dtype != tf.float32:
            img = tf.image.convert_image_dtype(img, tf.float32)
        
        # S'assurer que l'image a 3 channels
        if len(img.shape) == 3 and img.shape[-1] != 3:
            if img.shape[-1] == 1:  # Niveaux de gris
                img = tf.image.grayscale_to_rgb(img)
            elif img.shape[-1] == 4:  # RGBA
                img = img[:, :, :3]
        
        # Redimensionner en conservant le ratio d'aspect
        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim
        
        new_shape = tf.cast(shape * scale, tf.int32)
        img = tf.image.resize(img, new_shape)
        
        # Ajouter la dimension batch
        img = img[tf.newaxis, :]
        
        # Vérifier que les valeurs sont dans [0, 1]
        img = tf.clip_by_value(img, 0.0, 1.0)
        
        return img

    def deprocess_image(processed_img):
        """Convertit l'image traitée en format affichable"""
        x = processed_img.copy()
        if len(x.shape) == 4:
            x = np.squeeze(x, 0)
        
        # Assurer que les valeurs sont dans [0, 1]
        x = np.clip(x, 0, 1)
        
        # Convertir en [0, 255]
        x = (x * 255).astype('uint8')
        
        return x

    def get_content_and_style_representations(model, content_path, style_path):
        """Extrait les représentations de contenu et de style"""
        content_layers = ['block5_conv2']
        style_layers = ['block1_conv1',
                       'block2_conv1',
                       'block3_conv1', 
                       'block4_conv1', 
                       'block5_conv1']
        
        num_content_layers = len(content_layers)
        num_style_layers = len(style_layers)
        
        style_outputs = [model.get_layer(name).output for name in style_layers]
        content_outputs = [model.get_layer(name).output for name in content_layers]
        model_outputs = style_outputs + content_outputs
        
        feature_extractor = Model(model.input, model_outputs)
        
        content_image = preprocess_image(content_path)
        style_image = preprocess_image(style_path)
        
        content_image = preprocess_input(content_image * 255)
        style_image = preprocess_input(style_image * 255)
        
        style_features = feature_extractor(style_image)
        content_features = feature_extractor(content_image)
        
        style_features = [style_layer[0] for style_layer in style_features[:num_style_layers]]
        content_features = [content_layer[0] for content_layer in content_features[num_style_layers:]]
        
        return style_features, content_features

    def gram_matrix(input_tensor):
        """Calcule la matrice de Gram pour capturer le style"""
        # S'assurer que le tenseur a la bonne forme (batch, height, width, channels)
        if len(input_tensor.shape) == 3:
            input_tensor = tf.expand_dims(input_tensor, 0)
        
        # Reshape le tenseur en (batch, num_locations, num_features)
        batch_size, height, width, channels = tf.shape(input_tensor)[0], tf.shape(input_tensor)[1], tf.shape(input_tensor)[2], tf.shape(input_tensor)[3]
        features = tf.reshape(input_tensor, (batch_size, height * width, channels))
        
        # Calculer la matrice de Gram : G = F^T * F
        gram = tf.matmul(features, features, transpose_a=True)
        
        # Normaliser par le nombre de positions
        num_locations = tf.cast(height * width, tf.float32)
        return gram / num_locations

    def style_content_loss(outputs, style_targets, content_targets, style_weight, content_weight):
        """Calcule la perte combinée de style et de contenu"""
        style_outputs = outputs['style']
        content_outputs = outputs['content']
        
        # Calculer la perte de style
        style_loss = 0
        for name in style_targets.keys():
            style_loss += tf.reduce_mean((style_outputs[name] - style_targets[name])**2)
        style_loss *= style_weight / len(style_targets)
        
        # Calculer la perte de contenu
        content_loss = 0
        for name in content_targets.keys():
            content_loss += tf.reduce_mean((content_outputs[name] - content_targets[name])**2)
        content_loss *= content_weight / len(content_targets)
        
        total_loss = style_loss + content_loss
        return total_loss

    class StyleContentModel(tf.keras.models.Model):
        def __init__(self, style_layers, content_layers):
            super(StyleContentModel, self).__init__()
            self.vgg = load_vgg_model()
            self.style_layers = style_layers
            self.content_layers = content_layers
            self.num_style_layers = len(style_layers)
            self.vgg.trainable = False
            
            # Créer le modèle d'extraction des features
            style_outputs = [self.vgg.get_layer(name).output for name in style_layers]
            content_outputs = [self.vgg.get_layer(name).output for name in content_layers]
            model_outputs = style_outputs + content_outputs
            self.feature_extractor = tf.keras.Model([self.vgg.input], model_outputs)

        def call(self, inputs):
            # S'assurer que inputs a la bonne forme
            if len(inputs.shape) == 3:
                inputs = tf.expand_dims(inputs, 0)
            
            # Assurer que les valeurs sont dans [0, 1]
            inputs = tf.clip_by_value(inputs, 0.0, 1.0)
            
            # Préprocesser l'input pour VGG19 (conversion en [0, 255] puis normalisation ImageNet)
            inputs_scaled = inputs * 255.0
            preprocessed_input = preprocess_input(inputs_scaled)
            
            # Extraire les features
            outputs = self.feature_extractor(preprocessed_input)
            
            # Séparer les outputs de style et de contenu
            style_outputs = outputs[:self.num_style_layers]
            content_outputs = outputs[self.num_style_layers:]

            # Calculer les matrices de Gram pour le style
            style_features = []
            for i in range(self.num_style_layers):
                style_features.append(gram_matrix(style_outputs[i]))

            # Créer les dictionnaires de sortie
            content_dict = {}
            for i, content_name in enumerate(self.content_layers):
                content_dict[content_name] = content_outputs[i]

            style_dict = {}
            for i, style_name in enumerate(self.style_layers):
                style_dict[style_name] = style_features[i]

            return {'content': content_dict, 'style': style_dict}

    def perform_style_transfer(content_path, style_path, style_weight, content_weight, iterations):
        """Effectue le transfert de style"""
        content_layers = ['block5_conv2']
        style_layers = ['block1_conv1',
                       'block2_conv1',
                       'block3_conv1', 
                       'block4_conv1', 
                       'block5_conv1']

        # Créer le modèle d'extraction
        extractor = StyleContentModel(style_layers, content_layers)

        # Préprocesser les images
        content_image = preprocess_image(content_path)
        style_image = preprocess_image(style_path)

        # Afficher les images préprocessées pour vérification seulement en mode debug
        if debug_mode:
            st.write("🔍 **Vérification des images préprocessées:**")
            col_debug1, col_debug2 = st.columns(2)
            with col_debug1:
                debug_content = deprocess_image(content_image.numpy())
                st.image(debug_content, caption="Image de contenu préprocessée", width=200)
            with col_debug2:
                debug_style = deprocess_image(style_image.numpy())
                st.image(debug_style, caption="Image de style préprocessée", width=200)

        # Extraire les caractéristiques cibles (en mode eager)
        style_targets = extractor(style_image)['style']
        content_targets = extractor(content_image)['content']

        # Initialiser l'image de sortie avec l'image de contenu
        image = tf.Variable(content_image, dtype=tf.float32)
        
        # Optimiseur avec learning rate ajustable
        opt = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.99, epsilon=1e-1)

        # Barres de progression
        progress_bar = st.progress(0)
        status_text = st.empty()
        preview_placeholder = st.empty()
        
        # Fonction d'entraînement non compilée pour éviter les problèmes de graphe
        def train_step(image):
            with tf.GradientTape() as tape:
                outputs = extractor(image)
                loss = style_content_loss(outputs, style_targets, content_targets, 
                                        style_weight, content_weight)

            grad = tape.gradient(loss, image)
            opt.apply_gradients([(grad, image)])
            image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))
            return loss

        # Boucle d'optimisation en mode eager
        for i in range(iterations):
            try:
                loss = train_step(image)
                
                # Mise à jour de la barre de progression
                progress = (i + 1) / iterations
                progress_bar.progress(progress)
                status_text.text(f'Itération {i+1}/{iterations} - Perte: {loss:.4f}')
                
                # Afficher un aperçu toutes les 20 itérations seulement en mode debug
                if debug_mode and i % 20 == 0 and i > 0:
                    preview_img = deprocess_image(image.numpy())
                    with preview_placeholder.container():
                        st.image(preview_img, caption=f"Aperçu - Itération {i+1}", width=300)
                
            except Exception as e:
                st.error(f"Erreur à l'itération {i+1}: {str(e)}")
                break

        progress_bar.empty()
        status_text.empty()
        preview_placeholder.empty()
        
        return image

    # Interface utilisateur pour le transfert de style
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📸 Image de Contenu")
        content_file = st.file_uploader("Choisissez l'image de contenu", type=["jpg", "png", "jpeg"], key="content")
        if content_file:
            st.image(content_file, caption="Image de contenu", use_column_width=True)
            if show_diagnostics:
                diagnose_image(content_file, "contenu")
    
    with col2:
        st.markdown("### 🎨 Image de Style")
        style_file = st.file_uploader("Choisissez l'image de style", type=["jpg", "png", "jpeg"], key="style")
        if style_file:
            st.image(style_file, caption="Image de style", use_column_width=True)
            if show_diagnostics:
                diagnose_image(style_file, "style")
    
    # Bouton pour lancer le transfert de style
    if st.button("🚀 Lancer le Transfert de Style", type="primary"):
        if content_file and style_file:
            with st.spinner('Transfert de style en cours... Cela peut prendre quelques minutes.'):
                try:
                    # Effectuer le transfert de style
                    stylized_image = perform_style_transfer(
                        content_file, style_file, 
                        style_weight, content_weight, iterations
                    )
                    
                    # Convertir et afficher le résultat
                    result_image = deprocess_image(stylized_image.numpy())
                    
                    st.markdown("### 🎯 Résultat du Transfert de Style")
                    st.image(result_image, caption="Image stylisée", use_column_width=True)
                    
                    # Offrir la possibilité de télécharger l'image
                    result_pil = Image.fromarray(result_image)
                    buf = io.BytesIO()
                    result_pil.save(buf, format='PNG')
                    byte_im = buf.getvalue()
                    
                    st.download_button(
                        label="📥 Télécharger l'image stylisée",
                        data=byte_im,
                        file_name="image_stylisee.png",
                        mime="image/png"
                    )
                    
                    st.success("✅ Transfert de style terminé avec succès !")
                    
                except Exception as e:
                    st.error(f"❌ Erreur lors du transfert de style : {str(e)}")
        else:
            st.warning("⚠️ Veuillez charger une image de contenu et une image de style.")
    
    # Section d'information
    with st.expander("ℹ️ Comment ça marche ?"):
        st.markdown("""
        **Le transfert de style neuronal** utilise des réseaux de neurones convolutifs pour combiner le contenu d'une image avec le style d'une autre.
        
        **Paramètres :**
        - **Poids du style** : Contrôle l'intensité du style appliqué
        - **Poids du contenu** : Préserve les détails de l'image originale
        - **Itérations** : Plus d'itérations = meilleure qualité (mais plus lent)
        
        **Conseils :**
        - Utilisez des images de haute qualité pour de meilleurs résultats
        - Expérimentez avec différents poids pour trouver l'équilibre parfait
        - Les styles artistiques donnent généralement de meilleurs résultats
        """)