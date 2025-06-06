import streamlit as st
import base64
import zipfile
import os
import shutil
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
import time
from datetime import datetime

# 🌍 Variables globales - Comme des "boîtes de rangement" partagées dans toute l'application
model = None  # Notre modèle de classification - comme un cerveau entraîné
model_path = "saved_model.h5"  # Adresse où sauvegarder notre cerveau
class_names = []  # Liste des noms de classes - comme un dictionnaire

# 🎨 Configuration de la page Streamlit - Décoration de notre interface
st.set_page_config(page_title="SHEEP MORPH - photo style morphing", layout="wide")

# 🏷️ Interface utilisateur - Titres et présentation
st.markdown(
    """
    <h1 style="text-align: center; color: black;">SAFFIRE morphing</h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <h2 style="text-align: center; color: black;">Powered by AI and TensorFlow</h2>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style="text-align: center; color: black;">Changez le style de vos photos</p>
    """,
    unsafe_allow_html=True
)

# 🖼️ Configuration de l'image de fond - Comme changer le papier peint
background_image_path = "background.jpg"

if os.path.exists(background_image_path):
    with open(background_image_path, "rb") as image_file:
        # Conversion de l'image en format web (Base64) - comme traduire une langue
        encoded_string = base64.b64encode(image_file.read()).decode()
    
    # Injection du CSS pour l'arrière-plan - comme peindre les murs
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

# 🎛️ Interface de contrôle principale - Menu de navigation
st.sidebar.header("Configuration")
main_mode = st.sidebar.radio("Sélectionnez le module:", ["Classification", "Transfert de Style"])

# 📊 MODULE DE CLASSIFICATION
if main_mode == "Classification":
    mode = st.sidebar.radio("Select Mode:", ["Automatic", "Manual"])
    
    # Logo dans la barre latérale - Décoration
    if os.path.exists("logo.jpg"):
        st.sidebar.image("logo.jpg", width=150, caption="SAFFIRE")

    # 📦 Section de chargement des données
    st.markdown("## Chargement des Données")
    train_data = st.file_uploader("Importer les données d'entraînement (ZIP)", type=["zip"])
    train_dir = "temp_train_dir"

    def extract_zip(zip_file, extract_to):
        """
        🗂️ Fonction d'extraction intelligente du ZIP
        
        Analogie : Comme déballer un colis avec plusieurs boîtes imbriquées.
        Cette fonction est assez intelligente pour comprendre si votre ZIP a
        une boîte supplémentaire à l'intérieur et l'enlève automatiquement.
        
        Args:
            zip_file: Le fichier ZIP à déballer
            extract_to: Où mettre le contenu déballé
        """
        # 🧹 Nettoyage préalable - Comme vider une boîte avant de la remplir
        if os.path.exists(extract_to):
            shutil.rmtree(extract_to)
        os.makedirs(extract_to)
        
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            # 📂 Extraction dans un dossier temporaire - Zone de tri
            temp_extract = extract_to + "_temp"
            zip_ref.extractall(temp_extract)
            
            # 🔍 Analyse de la structure - Comme inspecter le contenu d'un colis
            items = os.listdir(temp_extract)
            
            # Cas 1: Structure directe (parfaite) - Comme un colis bien organisé
            if all(os.path.isdir(os.path.join(temp_extract, item)) for item in items if not item.startswith('.')):
                for item in items:
                    if not item.startswith('.') and not item == '__MACOSX':
                        source_path = os.path.join(temp_extract, item)
                        dest_path = os.path.join(extract_to, item)
                        if os.path.isdir(source_path):
                            shutil.copytree(source_path, dest_path)
                        else:
                            shutil.copy2(source_path, dest_path)
            
            # Cas 2: Dossier parent en trop - Comme une boîte dans une boîte
            else:
                parent_folder = None
                for item in items:
                    item_path = os.path.join(temp_extract, item)
                    if os.path.isdir(item_path) and not item.startswith('.') and item != '__MACOSX':
                        sub_items = os.listdir(item_path)
                        if any(os.path.isdir(os.path.join(item_path, sub)) for sub in sub_items if not sub.startswith('.')):
                            parent_folder = item_path
                            break
                
                if parent_folder:
                    # Extraction du contenu de la boîte interne
                    for item in os.listdir(parent_folder):
                        if not item.startswith('.') and not item == '__MACOSX':
                            source_path = os.path.join(parent_folder, item)
                            dest_path = os.path.join(extract_to, item)
                            if os.path.isdir(source_path):
                                shutil.copytree(source_path, dest_path)
                            else:
                                shutil.copy2(source_path, dest_path)
                else:
                    # Structure non reconnue - Copie tout tel quel
                    for item in items:
                        if not item.startswith('.') and not item == '__MACOSX':
                            source_path = os.path.join(temp_extract, item)
                            dest_path = os.path.join(extract_to, item)
                            if os.path.isdir(source_path):
                                shutil.copytree(source_path, dest_path)
                            else:
                                shutil.copy2(source_path, dest_path)
            
            # 🧹 Nettoyage du dossier temporaire - Ranger après le tri
            shutil.rmtree(temp_extract)
            
            # 📊 Rapport de ce qui a été trouvé - Inventaire
            classes_found = [d for d in os.listdir(extract_to) if os.path.isdir(os.path.join(extract_to, d))]
            st.info(f"📁 Structure détectée : {len(classes_found)} classes trouvées")
            st.write("Classes détectées :", ", ".join(classes_found))

    # ⚙️ Configuration des hyperparamètres selon le mode
    if mode == "Manual":
        st.sidebar.markdown("### 🎛️ Hyperparamètres d'Entraînement")
        
        # 🧠 Optimiseur - Le "professeur" qui guide l'apprentissage
        optimizer_choice = st.sidebar.selectbox(
            "Optimiseur (le 'professeur' de votre IA):", 
            ("Adam", "SGD", "RMSprop"),
            help="Adam = professeur patient et intelligent | SGD = professeur simple mais efficace | RMSprop = professeur qui s'adapte"
        )
        
        # 📏 Taux d'apprentissage - La "vitesse d'apprentissage"
        learning_rate = st.sidebar.number_input(
            "Taux d'apprentissage (vitesse d'apprentissage):", 
            min_value=0.0001, max_value=1.0, value=0.001, step=0.0001, format="%f",
            help="Trop rapide = votre IA devient confuse | Trop lent = votre IA apprend très lentement"
        )
        
        # 🔄 Epochs - Nombre de "cours" complets
        epochs = st.sidebar.number_input(
            "Nombre d'epochs (cours complets):", 
            min_value=1, max_value=100, value=30, step=1,
            help="Comme le nombre de fois que votre IA revoit tout le manuel d'apprentissage"
        )
        
        # 📦 Taille des batchs - Taille des "groupes d'étude"
        batch_size = st.sidebar.number_input(
            "Taille des batchs (taille des groupes d'étude):", 
            min_value=1, max_value=128, value=32, step=1,
            help="Nombre d'images que votre IA étudie en même temps. Plus grand = plus rapide mais plus de mémoire"
        )
        
        # 🏗️ Architecture du réseau neuronal
        st.sidebar.markdown("### 🏗️ Architecture du Réseau")
        
        # Nombre de couches convolutives - "Étages de détection"
        num_conv_layers = st.sidebar.slider(
            "Nombre de couches convolutives (étages de détection):", 
            min_value=1, max_value=5, value=3,
            help="Chaque étage détecte des patterns plus complexes : 1=lignes, 2=formes, 3=objets"
        )
        
        # Filtres par couche - "Nombre de détecteurs par étage"
        filters_per_layer = []
        for i in range(num_conv_layers):
            filters = st.sidebar.number_input(
                f"Filtres couche {i+1} (détecteurs à l'étage {i+1}):", 
                min_value=8, max_value=512, value=16 * (2**i), step=8,
                help=f"Étage {i+1}: Plus de détecteurs = plus de précision mais plus lent"
            )
            filters_per_layer.append(filters)
        
        # Neurones dans la couche dense - "Taille du cerveau de décision"
        dense_units = st.sidebar.number_input(
            "Neurones Dense (taille du cerveau de décision):", 
            min_value=8, max_value=512, value=64, step=8,
            help="Le cerveau final qui prend la décision. Plus gros = plus intelligent mais plus lent"
        )
        
        # Taux de Dropout - "Pourcentage d'oubli volontaire"
        dropout_rate = st.sidebar.slider(
            "Taux de Dropout (oubli volontaire pour éviter le par-cœur):", 
            min_value=0.0, max_value=0.9, value=0.5, step=0.05,
            help="Comme dire à votre IA d'oublier 50% de ce qu'elle voit pour ne pas apprendre par cœur"
        )
        
        # Fonction d'activation - "Type de réflexion"
        activation_function = st.sidebar.selectbox(
            "Fonction d'activation (type de réflexion):", 
            ("relu", "sigmoid", "tanh"),
            help="ReLU = pensée simple et rapide | Sigmoid = pensée nuancée | Tanh = pensée équilibrée"
        )
        
        # 🛡️ Régularisation avancée
        st.sidebar.markdown("### 🛡️ Régularisation Avancée")
        
        # Régularisation L2 - "Pénalité pour la complexité"
        l2_regularization = st.sidebar.slider(
            "Régularisation L2 (pénalité complexité):", 
            min_value=0.0, max_value=0.01, value=0.0001, step=0.0001, format="%.4f",
            help="Empêche votre IA de devenir trop compliquée. Comme limiter le nombre de règles qu'elle peut apprendre"
        )
        
        # Patience pour l'arrêt précoce
        early_stopping_patience = st.sidebar.number_input(
            "Patience arrêt précoce (patience avant abandon):", 
            min_value=5, max_value=50, value=10, step=1,
            help="Nombre d'epochs sans amélioration avant d'arrêter l'entraînement automatiquement"
        )
        
        # Réduction du learning rate
        reduce_lr_patience = st.sidebar.number_input(
            "Patience réduction LR (patience avant ralentissement):", 
            min_value=3, max_value=20, value=5, step=1,
            help="Si pas d'amélioration, ralentir l'apprentissage au lieu d'abandonner"
        )
        
        reduce_lr_factor = st.sidebar.slider(
            "Facteur réduction LR (niveau de ralentissement):", 
            min_value=0.1, max_value=0.9, value=0.5, step=0.1,
            help="De combien ralentir l'apprentissage (0.5 = moitié moins vite)"
        )
        
    else:
        # 🤖 Mode automatique - Configuration optimisée par défaut
        st.sidebar.markdown("### 🤖 Configuration Automatique Optimisée")
        st.sidebar.write("✅ Paramètres optimaux sélectionnés automatiquement")
        
        # Valeurs par défaut optimisées
        optimizer_choice = "Adam"
        learning_rate = 0.001
        epochs = 20
        batch_size = 32
        num_conv_layers = 3
        filters_per_layer = [32, 64, 128]
        dense_units = 128
        dropout_rate = 0.3
        activation_function = "relu"
        l2_regularization = 0.0001
        early_stopping_patience = 10
        reduce_lr_patience = 5
        reduce_lr_factor = 0.5

    # 🚀 Bouton d'entraînement
    if st.button("🚀 Démarrer l'entraînement", type="primary"):
        if train_data is not None:
            # 📦 Extraction et préparation des données
            extract_zip(train_data, train_dir)
            
            # 🎭 Augmentation des données - "Créer des variations"
            datagen = ImageDataGenerator(
                rescale=1./255,  # Normalisation - ramener les valeurs entre 0 et 1
                validation_split=0.2,  # 20% pour tester, 80% pour apprendre
                # Options d'augmentation possibles (commentées pour le mode de base)
                # rotation_range=20,  # Rotation aléatoire
                # width_shift_range=0.2,  # Décalage horizontal
                # height_shift_range=0.2,  # Décalage vertical
                # horizontal_flip=True  # Miroir horizontal
            )
            
            # 🏭 Création des générateurs de données - "Chaînes de production"
            train_generator = datagen.flow_from_directory(
                train_dir, 
                target_size=(128, 128),  # Redimensionner toutes les images
                batch_size=batch_size, 
                class_mode='categorical',  # Classification multi-classes
                subset='training'
            )
            
            val_generator = datagen.flow_from_directory(
                train_dir, 
                target_size=(128, 128), 
                batch_size=batch_size, 
                class_mode='categorical', 
                subset='validation'
            )
            
            # 📊 Analyse des données
            num_classes = len(train_generator.class_indices)
            class_names = list(train_generator.class_indices.keys())
            st.info(f"🎯 {num_classes} classes détectées : {', '.join(class_names)}")

            # 🧠 Sélection de l'optimiseur - "Choix du professeur"
            if optimizer_choice == "Adam":
                # Adam = Professeur intelligent qui s'adapte
                optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)
            elif optimizer_choice == "SGD":
                # SGD = Professeur traditionnel mais efficace
                optimizer = SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
            elif optimizer_choice == "RMSprop":
                # RMSprop = Professeur qui s'adapte aux difficultés
                optimizer = RMSprop(learning_rate=learning_rate, rho=0.9)

            # 🏗️ Construction du modèle - "Assemblage du cerveau"
            model = Sequential()
            
            # 👁️ Couches convolutives - "Étages de détection visuelle"
            for i, filters in enumerate(filters_per_layer):
                model.add(Conv2D(
                    filters, (3, 3),  # Filtres 3x3 comme des petites loupes
                    activation=activation_function,
                    kernel_regularizer=l2(l2_regularization),  # Éviter la sur-spécialisation
                    padding='same',  # Garder la même taille d'image
                    name=f'conv2d_{i+1}'
                ))
                
                # 🧼 Normalisation par batch - "Standardisation"
                model.add(BatchNormalization(name=f'batch_norm_{i+1}'))
                
                # 🔽 Max Pooling - "Résumé de zone"
                model.add(MaxPooling2D(pool_size=(2, 2), name=f'max_pool_{i+1}'))
            
            # 🌍 Globalisation des informations
            model.add(GlobalAveragePooling2D(name='global_avg_pool'))
            
            # 🧠 Couche de décision dense
            model.add(Dense(
                dense_units, 
                activation=activation_function,
                kernel_regularizer=l2(l2_regularization),
                name='dense_decision'
            ))
            
            # 🎲 Dropout - "Oubli volontaire pour éviter le par-cœur"
            model.add(Dropout(dropout_rate, name='dropout'))
            
            # 🎯 Couche de sortie finale
            model.add(Dense(
                num_classes, 
                activation='softmax',  # Probabilités qui totalisent 100%
                name='output'
            ))

            # ⚙️ Compilation du modèle - "Configuration finale"
            model.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',  # Fonction de coût pour classification
                metrics=['accuracy']  # Mesurer la précision
            )
            
            # 📊 Affichage de l'architecture
            st.write("🏗️ **Architecture du modèle créé :**")
            model_summary = []
            model.summary(print_fn=lambda x: model_summary.append(x))
            st.text('\n'.join(model_summary))

            # 🎓 Entraînement avec callbacks - "Supervision intelligente"
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
            
            callbacks = [
                # 🛑 Arrêt précoce si pas d'amélioration
                EarlyStopping(
                    monitor='val_accuracy',
                    patience=early_stopping_patience,
                    restore_best_weights=True,
                    verbose=1
                ),
                # 📉 Réduction du learning rate si plateau
                ReduceLROnPlateau(
                    monitor='val_accuracy',
                    factor=reduce_lr_factor,
                    patience=reduce_lr_patience,
                    min_lr=1e-7,
                    verbose=1
                )
            ]
            
            # 🚀 Lancement de l'entraînement
            with st.spinner('🎓 Entraînement en cours... Votre IA apprend !'):
                history = model.fit(
                    train_generator,
                    epochs=epochs,
                    validation_data=val_generator,
                    callbacks=callbacks,
                    verbose=1
                )
            
            # 💾 Sauvegarde du modèle entraîné
            model.save(model_path)
            st.success("✅ Entraînement terminé et modèle sauvegardé !")

            # 📈 Affichage des courbes de convergence - "Graphiques de progression"
            st.markdown("## 📈 Courbes de Convergence")
            fig, ax = plt.subplots(1, 2, figsize=(14, 5))
            
            # Graphique de la perte
            ax[0].plot(history.history['loss'], label='Perte Entraînement', color='blue')
            ax[0].plot(history.history['val_loss'], label='Perte Validation', color='red')
            ax[0].set_title('Évolution de la Perte (plus bas = mieux)')
            ax[0].set_xlabel('Epochs')
            ax[0].set_ylabel('Perte')
            ax[0].legend()
            ax[0].grid(True, alpha=0.3)

            # Graphique de la précision
            ax[1].plot(history.history['accuracy'], label='Précision Entraînement', color='green')
            ax[1].plot(history.history['val_accuracy'], label='Précision Validation', color='orange')
            ax[1].set_title('Évolution de la Précision (plus haut = mieux)')
            ax[1].set_xlabel('Epochs')
            ax[1].set_ylabel('Précision')
            ax[1].legend()
            ax[1].grid(True, alpha=0.3)
            
            st.pyplot(fig)

            # 🎯 Matrice de confusion - "Tableau des erreurs"
            st.markdown("## 🎯 Matrice de Confusion")
            if val_generator.samples > 0:
                # Prédictions sur les données de validation
                val_preds = model.predict(val_generator, verbose=0)
                y_pred = np.argmax(val_preds, axis=1)
                y_true = val_generator.classes

                # Mise à jour des noms de classes
                if not class_names:
                    class_names = list(val_generator.class_indices.keys())

                # Création et affichage de la matrice
                cm = confusion_matrix(y_true, y_pred)
                fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
                disp.plot(ax=ax_cm, cmap='Blues', colorbar=True)
                ax_cm.set_title("Matrice de Confusion\n(Diagonale = bonnes prédictions)")
                st.pyplot(fig_cm)
                
                # Statistiques détaillées
                accuracy = np.trace(cm) / np.sum(cm)
                st.write(f"**Précision globale :** {accuracy:.2%}")
                
            else:
                st.warning("⚠️ Aucune donnée de validation disponible.")

    # 🔮 Section de prédiction
    st.markdown("## 🔮 Prédiction sur une Image")
    image_file = st.file_uploader("Choisissez une image pour prédiction", type=["jpg", "png", "jpeg"])
    
    if image_file and st.button("🔮 Prédire", type="secondary"):
        # Chargement du modèle si nécessaire
        if model is None and os.path.exists(model_path):
            model = load_model(model_path)
            
        if model is not None:
            # Préprocessing de l'image
            img = load_img(image_file, target_size=(128, 128))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalisation
            
            # Prédiction
            prediction = model.predict(img_array)[0]
            predicted_class = np.argmax(prediction)
            predicted_label = class_names[predicted_class] if class_names else f"Classe {predicted_class}"
            confidence = round(prediction[predicted_class] * 100, 2)
            
            # Affichage des résultats
            col1, col2 = st.columns(2)
            with col1:
                st.image(image_file, caption="🖼️ Image à analyser")
            with col2:
                st.markdown("### 🎯 Résultat de la Prédiction")
                st.success(f"**Classe prédite :** {predicted_label}")
                st.info(f"**Confiance :** {confidence}%")
                
                # Barre de progression pour la confiance
                st.progress(confidence / 100)
                
                # Top 3 des prédictions
                if len(prediction) > 1:
                    st.markdown("#### 🏆 Top 3 des prédictions")
                    top_3_indices = np.argsort(prediction)[-3:][::-1]
                    for i, idx in enumerate(top_3_indices):
                        class_name = class_names[idx] if class_names else f"Classe {idx}"
                        score = prediction[idx] * 100
                        st.write(f"{i+1}. {class_name}: {score:.1f}%")
        else:
            st.error("❗ Aucun modèle chargé. Veuillez d'abord entraîner un modèle.")

# 🎨 MODULE DE TRANSFERT DE STYLE
elif main_mode == "Transfert de Style":
    st.markdown("## 🎨 Module de Transfert de Style Neural")
    st.write("*Transformez vos photos en œuvres d'art en combinant le contenu d'une image avec le style d'une autre*")
    
    # 🎛️ Hyperparamètres principaux dans la sidebar
    st.sidebar.markdown("### 🎨 Paramètres Artistiques")
    
    # Poids du style - "Intensité artistique"
    style_weight = st.sidebar.slider(
        "Poids du style (intensité artistique):", 
        1e-2, 1e6, 1e4, step=1e3, format="%.0e",
        help="Plus élevé = plus artistique mais moins ressemblant à l'original"
    )
    
    # Poids du contenu - "Préservation de l'original"
    content_weight = st.sidebar.slider(
        "Poids du contenu (préservation original):", 
        1e0, 1e4, 1e3, step=1e2, format="%.0e",
        help="Plus élevé = plus fidèle à l'image originale"
    )
    
    # Nombre d'itérations - "Temps de création"
    iterations = st.sidebar.number_input(
        "Nombre d'itérations (temps de création):", 
        min_value=10, max_value=1000, value=100, step=10,
        help="Plus d'itérations = meilleure qualité mais plus lent"
    )
    
    # 🔧 Paramètres avancés
    st.sidebar.markdown("### 🔧 Paramètres Avancés")
    
    # Taux d'apprentissage - "Vitesse d'amélioration"
    learning_rate = st.sidebar.slider(
        "Taux d'apprentissage (vitesse amélioration):", 
        0.001, 0.1, 0.01, step=0.001,
        help="Plus rapide = convergence rapide mais risque d'instabilité"
    )
    
    # Taille maximale d'image - "Qualité vs vitesse"
    max_image_size = st.sidebar.selectbox(
        "Taille max image (qualité vs vitesse):",
        [256, 384, 512, 768, 1024],
        index=2,  # 512 par défaut
        help="Plus grand = meilleure qualité mais beaucoup plus lent"
    )
    
    # Paramètres de l'optimiseur Adam
    st.sidebar.markdown("#### ⚙️ Optimiseur Adam")
    
    beta1 = st.sidebar.slider(
        "Beta1 (mémoire gradient):",
        0.8, 0.999, 0.99, step=0.01,
        help="Mémoire des gradients précédents. Plus élevé = plus de mémoire"
    )
    
    beta2 = st.sidebar.slider(
        "Beta2 (mémoire variance):",
        0.9, 0.999, 0.999, step=0.001,
        help="Mémoire de la variance. Généralement proche de 1.0"
    )
    
    epsilon = st.sidebar.selectbox(
        "Epsilon (stabilité numérique):",
        [1e-8, 1e-4, 1e-1, 1e-0],
        index=2,  # 1e-1 par défaut
        format_func=lambda x: f"{x:.0e}",
        help="Évite la division par zéro. Plus élevé = plus stable"
    )
    
    # 🎚️ Contrôle des couches de style
    st.sidebar.markdown("#### 🎨 Couches de Style Actives")
    st.sidebar.write("*Chaque couche capture différents aspects du style*")
    
    use_block1 = st.sidebar.checkbox(
        "Block1 (textures fines)", 
        value=True,
        help="Capture les détails fins : lignes, points, textures de base"
    )
    use_block2 = st.sidebar.checkbox(
        "Block2 (motifs simples)", 
        value=True,
        help="Capture les motifs simples : rayures, cercles, formes géométriques"
    )
    use_block3 = st.sidebar.checkbox(
        "Block3 (structures moyennes)", 
        value=True,
        help="Capture les structures moyennes : objets partiels, compositions"
    )
    use_block4 = st.sidebar.checkbox(
        "Block4 (formes complexes)", 
        value=True,
        help="Capture les formes complexes : objets entiers, relations spatiales"
    )
    use_block5 = st.sidebar.checkbox(
        "Block5 (composition globale)", 
        value=True,
        help="Capture la composition globale : distribution des éléments, style général"
    )
    
    # 🎭 Options de préprocessing
    st.sidebar.markdown("#### 🎭 Préprocessing des Images")
    
    preserve_colors = st.sidebar.checkbox(
        "Préserver couleurs originales",
        value=False,
        help="Garde les couleurs de l'image de contenu et applique seulement la texture du style"
    )
    
    enhance_contrast = st.sidebar.slider(
        "Amélioration contraste:",
        0.5, 2.0, 1.0, step=0.1,
        help="Ajuste le contraste de l'image finale. 1.0 = normal"
    )
    
    color_saturation = st.sidebar.slider(
        "Saturation couleurs:",
        0.0, 2.0, 1.0, step=0.1,
        help="Ajuste la vivacité des couleurs. 1.0 = normal"
    )
    
    # 🔍 Options de débogage
    st.sidebar.markdown("### 🔍 Débogage et Analyse")
    
    debug_mode = st.sidebar.checkbox(
        "Mode débogage", 
        value=True,
        help="Affiche les images préprocessées et aperçus pendant l'entraînement"
    )
    
    show_diagnostics = st.sidebar.checkbox(
        "Diagnostic détaillé", 
        value=False,
        help="Analyse technique des images uploadées (format, taille, valeurs)"
    )
    
    show_loss_breakdown = st.sidebar.checkbox(
        "Détail des pertes",
        value=False, 
        help="Affiche séparément la perte de style et de contenu"
    )
    
    preview_frequency = st.sidebar.number_input(
        "Fréquence aperçus:",
        min_value=5, max_value=50, value=20, step=5,
        help="Montrer un aperçu toutes les X itérations"
    )

    def diagnose_image(image_file, name):
        """
        🔬 Fonction de diagnostic d'image
        
        Analogie : Comme un médecin qui examine un patient.
        Cette fonction regarde tous les "signes vitaux" de votre image :
        - Sa taille, son format, ses couleurs
        - Détecte s'il y a des problèmes potentiels
        
        Args:
            image_file: L'image à diagnostiquer
            name: Nom de l'image pour l'affichage
        """
        img = Image.open(image_file)
        img_array = np.array(img)
        
        st.write(f"**📊 Diagnostic de l'image {name}:**")
        st.write(f"- Format original: {img.format} {'✅' if img.format in ['JPEG', 'PNG'] else '⚠️'}")
        st.write(f"- Mode couleur: {img.mode} {'✅' if img.mode == 'RGB' else '⚠️'}")
        st.write(f"- Dimensions: {img.size[0]}×{img.size[1]} pixels")
        st.write(f"- Forme du array: {img_array.shape}")
        st.write(f"- Type de données: {img_array.dtype}")
        st.write(f"- Plage valeurs: {img_array.min()} → {img_array.max()}")
        
        # Analyse des couleurs par canal
        if len(img_array.shape) == 3:
            r_mean = img_array[:,:,0].mean()
            g_mean = img_array[:,:,1].mean()
            b_mean = img_array[:,:,2].mean()
            st.write(f"- Moyennes RGB: R={r_mean:.1f}, G={g_mean:.1f}, B={b_mean:.1f}")
            
            # Détection de problèmes potentiels
            if abs(r_mean - g_mean) < 5 and abs(g_mean - b_mean) < 5:
                st.warning("⚠️ Image semble en niveaux de gris")
            if img_array.max() <= 1:
                st.info("ℹ️ Image déjà normalisée [0,1]")
        
        return img_array

    # 🎨 Fonctions pour le transfert de style
    @st.cache_resource
    def load_vgg_model():
        """
        🧠 Chargement du modèle VGG19 pré-entraîné
        
        Analogie : Comme emprunter les yeux d'un expert en art.
        VGG19 a été entraîné sur des millions d'images et "sait" reconnaître
        les formes, textures et styles artistiques.
        
        Returns:
            Modèle VGG19 figé (non-entraînable)
        """
        vgg = VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False  # On fige le modèle - pas d'apprentissage
        return vgg

    def preprocess_image(image_path, max_dim=512):
        """
        🛠️ Préprocessing intelligent des images
        
        Analogie : Comme préparer une toile avant de peindre.
        Cette fonction nettoie, redimensionne et normalise l'image
        pour qu'elle soit parfaite pour le transfert de style.
        
        Args:
            image_path: Chemin vers l'image ou objet file Streamlit
            max_dim: Dimension maximale (plus grand = plus lent)
            
        Returns:
            Tensor TensorFlow normalisé et redimensionné
        """
        if isinstance(image_path, str):
            # Chargement depuis fichier local
            img = tf.io.read_file(image_path)
            img = tf.image.decode_image(img, channels=3)
        else:
            # Chargement depuis Streamlit file uploader
            img = Image.open(image_path)
            img = np.array(img)
            
            # 🎨 Gestion des différents formats d'image
            if len(img.shape) == 2:  # Image en niveaux de gris
                # Conversion en RGB en dupliquant le canal
                img = np.stack([img] * 3, axis=-1)
                st.info("ℹ️ Image convertie de niveaux de gris vers RGB")
                
            elif len(img.shape) == 3 and img.shape[-1] == 4:  # RGBA (avec transparence)
                # Suppression du canal alpha (transparence)
                img = img[:, :, :3]
                st.info("ℹ️ Canal alpha supprimé (RGBA → RGB)")
            
            # 📊 Normalisation intelligente des valeurs
            if img.dtype == np.uint8:
                # Conversion standard uint8 [0,255] → float32 [0,1]
                img = img.astype(np.float32) / 255.0
            elif img.dtype == np.float32 and img.max() > 1.0:
                # Image float32 mal normalisée
                img = img / 255.0
                st.info("ℹ️ Image float32 renormalisée")
            
            # Conversion en tensor TensorFlow
            img = tf.constant(img, dtype=tf.float32)
        
        # 🔄 Vérifications et conversions finales
        if img.dtype != tf.float32:
            img = tf.image.convert_image_dtype(img, tf.float32)
        
        # Assurer que l'image a exactement 3 canaux
        if len(img.shape) == 3 and img.shape[-1] != 3:
            if img.shape[-1] == 1:  # Niveaux de gris
                img = tf.image.grayscale_to_rgb(img)
            elif img.shape[-1] == 4:  # RGBA
                img = img[:, :, :3]
        
        # 📏 Redimensionnement proportionnel
        # Analogie : Ajuster la taille d'une photo sans la déformer
        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim
        
        new_shape = tf.cast(shape * scale, tf.int32)
        img = tf.image.resize(img, new_shape)
        
        # 📦 Ajout de la dimension batch (pour le traitement par lots)
        img = img[tf.newaxis, :]
        
        # 🛡️ Sécurité : s'assurer que les valeurs sont dans [0,1]
        img = tf.clip_by_value(img, 0.0, 1.0)
        
        return img

    def deprocess_image(processed_img):
        """
        🖼️ Conversion de l'image traitée vers format affichable
        
        Analogie : Comme développer une photo depuis un négatif.
        Convertit le tensor TensorFlow normalisé en image PNG/JPEG standard.
        
        Args:
            processed_img: Tensor TensorFlow [0,1]
            
        Returns:
            Array NumPy uint8 [0,255] prêt pour l'affichage
        """
        x = processed_img.copy()
        
        # Suppression de la dimension batch si présente
        if len(x.shape) == 4:
            x = np.squeeze(x, 0)
        
        # 🛡️ Sécurité : clipper les valeurs dans [0,1]
        x = np.clip(x, 0, 1)
        
        # 🎨 Conversion vers format d'affichage [0,255]
        x = (x * 255).astype('uint8')
        
        return x

    def apply_color_adjustments(img, enhance_contrast, color_saturation):
        """
        🎨 Application d'ajustements colorimétriques
        
        Analogie : Comme ajuster les réglages d'un téléviseur.
        Modifie le contraste et la saturation pour améliorer le rendu final.
        
        Args:
            img: Image à ajuster
            enhance_contrast: Facteur de contraste (1.0 = normal)
            color_saturation: Facteur de saturation (1.0 = normal)
            
        Returns:
            Image ajustée
        """
        # Ajustement du contraste
        if enhance_contrast != 1.0:
            img = tf.image.adjust_contrast(img, enhance_contrast)
        
        # Ajustement de la saturation
        if color_saturation != 1.0:
            img = tf.image.adjust_saturation(img, color_saturation)
        
        # Re-clipper après ajustements
        img = tf.clip_by_value(img, 0.0, 1.0)
        
        return img

    def gram_matrix(input_tensor):
        """
        🔢 Calcul de la matrice de Gram pour capturer le style
        
        Analogie : Comme analyser les "empreintes digitales" artistiques.
        La matrice de Gram capture les corrélations entre différentes 
        caractéristiques visuelles, créant une signature unique du style.
        
        Math : G[i,j] = Σ(F[k,i] × F[k,j]) / N
        Où F sont les features et N le nombre de positions
        
        Args:
            input_tensor: Features extraites par VGG19
            
        Returns:
            Matrice de Gram (corrélations de style)
        """
        # 📏 Vérification et ajustement des dimensions
        if len(input_tensor.shape) == 3:
            input_tensor = tf.expand_dims(input_tensor, 0)
        
        # 📊 Extraction des dimensions
        batch_size = tf.shape(input_tensor)[0]
        height = tf.shape(input_tensor)[1] 
        width = tf.shape(input_tensor)[2]
        channels = tf.shape(input_tensor)[3]
        
        # 🔄 Reshape en matrice 2D : (positions, features)
        # Analogie : Comme étaler toutes les "observations" en lignes
        features = tf.reshape(input_tensor, (batch_size, height * width, channels))
        
        # 🧮 Calcul de la matrice de Gram : F^T × F
        # Analogie : Calculer toutes les corrélations entre features
        gram = tf.matmul(features, features, transpose_a=True)
        
        # ➗ Normalisation par le nombre de positions
        # Analogie : Faire une moyenne pour que la taille d'image n'influe pas
        num_locations = tf.cast(height * width, tf.float32)
        
        return gram / num_locations

    def build_style_layers_list():
        """
        🏗️ Construction de la liste des couches de style actives
        
        Retourne la liste des couches VGG19 à utiliser selon les checkboxes
        """
        style_layers = []
        if use_block1:
            style_layers.append('block1_conv1')
        if use_block2:
            style_layers.append('block2_conv1') 
        if use_block3:
            style_layers.append('block3_conv1')
        if use_block4:
            style_layers.append('block4_conv1')
        if use_block5:
            style_layers.append('block5_conv1')
            
        return style_layers

    def style_content_loss(outputs, style_targets, content_targets, style_weight, content_weight):
        """
        ⚖️ Calcul de la perte combinée style + contenu
        
        Analogie : Comme noter un devoir avec deux critères :
        - Respect du style artistique (originalité)
        - Préservation du contenu (fidélité)
        
        Args:
            outputs: Sorties actuelles du modèle
            style_targets: Cibles de style (ce qu'on veut atteindre)
            content_targets: Cibles de contenu (ce qu'on veut préserver)
            style_weight: Importance du style
            content_weight: Importance du contenu
            
        Returns:
            Perte totale à minimiser
        """
        style_outputs = outputs['style']
        content_outputs = outputs['content']
        
        # 🎨 Calcul de la perte de style
        # Analogie : Mesurer à quel point le style diffère de l'œuvre de référence
        style_loss = 0
        for name in style_targets.keys():
            # Différence quadratique entre matrices de Gram
            layer_loss = tf.reduce_mean((style_outputs[name] - style_targets[name])**2)
            style_loss += layer_loss
            
            # Debug : afficher les pertes par couche si demandé
            if show_loss_breakdown:
                st.write(f"🎨 Perte style {name}: {float(layer_loss):.4f}")
        
        # Normalisation par le nombre de couches de style
        style_loss *= style_weight / len(style_targets)
        
        # 📷 Calcul de la perte de contenu  
        # Analogie : Mesurer à quel point on s'éloigne de l'image originale
        content_loss = 0
        for name in content_targets.keys():
            # Différence quadratique entre features de contenu
            layer_loss = tf.reduce_mean((content_outputs[name] - content_targets[name])**2)
            content_loss += layer_loss
            
            if show_loss_breakdown:
                st.write(f"📷 Perte contenu {name}: {float(layer_loss):.4f}")
        
        # Normalisation par le nombre de couches de contenu
        content_loss *= content_weight / len(content_targets)
        
        # ⚖️ Perte totale = Style + Contenu
        total_loss = style_loss + content_loss
        
        # Debug détaillé
        if show_loss_breakdown:
            st.write(f"**Total - Style: {float(style_loss):.4f}, Contenu: {float(content_loss):.4f}**")
        
        return total_loss

    class StyleContentModel(tf.keras.models.Model):
        """
        🎭 Modèle d'extraction de style et contenu
        
        Analogie : Comme un critique d'art expert qui peut analyser
        séparément le style artistique et le contenu d'une œuvre.
        
        Cette classe utilise VGG19 pré-entraîné pour extraire :
        - Les caractéristiques de style (matrices de Gram)
        - Les caractéristiques de contenu (features sémantiques)
        """
        
        def __init__(self, style_layers, content_layers):
            """
            🏗️ Initialisation du modèle extracteur
            
            Args:
                style_layers: Liste des couches VGG19 pour le style
                content_layers: Liste des couches VGG19 pour le contenu
            """
            super(StyleContentModel, self).__init__()
            
            # 🧠 Chargement du modèle VGG19 pré-entraîné
            self.vgg = load_vgg_model()
            self.style_layers = style_layers
            self.content_layers = content_layers
            self.num_style_layers = len(style_layers)
            self.vgg.trainable = False  # Modèle figé
            
            # 🔌 Construction d'un extracteur unifié pour efficacité
            # Analogie : Créer un seul passage au lieu de plusieurs allers-retours
            style_outputs = [self.vgg.get_layer(name).output for name in style_layers]
            content_outputs = [self.vgg.get_layer(name).output for name in content_layers]
            model_outputs = style_outputs + content_outputs
            
            # 🏭 Modèle d'extraction unifié
            self.feature_extractor = tf.keras.Model([self.vgg.input], model_outputs)

        def call(self, inputs):
            """
            🔄 Forward pass - Extraction des caractéristiques
            
            Analogie : Passer une image dans les "yeux" de l'expert
            pour qu'il analyse le style et le contenu.
            
            Args:
                inputs: Image d'entrée normalisée [0,1]
                
            Returns:
                Dictionnaire avec features de style et contenu
            """
            # 📏 Vérification des dimensions d'entrée
            if len(inputs.shape) == 3:
                inputs = tf.expand_dims(inputs, 0)
            
            # 🛡️ Sécurité : clipper les valeurs dans [0,1]
            inputs = tf.clip_by_value(inputs, 0.0, 1.0)
            
            # 🎨 Ajustements colorimétriques si demandés
            inputs = apply_color_adjustments(inputs, enhance_contrast, color_saturation)
            
            # 🔄 Préprocessing pour VGG19
            # Conversion [0,1] → [0,255] puis normalisation ImageNet
            inputs_scaled = inputs * 255.0
            preprocessed_input = preprocess_input(inputs_scaled)
            
            # 🏭 Extraction des features via le modèle unifié
            outputs = self.feature_extractor(preprocessed_input)
            
            # 📊 Séparation des outputs style et contenu
            style_outputs = outputs[:self.num_style_layers]
            content_outputs = outputs[self.num_style_layers:]

            # 🎨 Calcul des matrices de Gram pour le style
            style_features = []
            for i in range(self.num_style_layers):
                gram = gram_matrix(style_outputs[i])
                style_features.append(gram)

            # 📦 Construction des dictionnaires de sortie
            content_dict = {}
            for i, content_name in enumerate(self.content_layers):
                content_dict[content_name] = content_outputs[i]

            style_dict = {}
            for i, style_name in enumerate(self.style_layers):
                style_dict[style_name] = style_features[i]

            return {'content': content_dict, 'style': style_dict}

    def perform_style_transfer(content_path, style_path, style_weight, content_weight, iterations):
        """
        🎨 Fonction principale de transfert de style
        
        Analogie : Comme un peintre qui mélange deux techniques :
        - Il garde la forme et structure de son modèle (contenu)
        - Il applique la technique d'un maître (style)
        
        Le processus est itératif, comme un artiste qui améliore 
        progressivement son œuvre coup de pinceau par coup de pinceau.
        
        Args:
            content_path: Image de contenu (ce qu'on veut styliser)
            style_path: Image de style (l'art qu'on veut imiter)
            style_weight: Importance du style artistique
            content_weight: Importance de rester fidèle au contenu
            iterations: Nombre d'améliorations à faire
            
        Returns:
            Image stylisée finale
        """
        # 📋 Configuration des couches d'analyse
        content_layers = ['block5_conv2']  # Couche sémantique profonde
        style_layers = build_style_layers_list()  # Selon sélection utilisateur
        
        if not style_layers:
            st.error("❌ Aucune couche de style sélectionnée ! Activez au moins une couche.")
            return None

        st.info(f"🎨 Utilisation de {len(style_layers)} couches de style : {', '.join(style_layers)}")

        # 🏗️ Création du modèle extracteur
        extractor = StyleContentModel(style_layers, content_layers)

        # 🛠️ Préprocessing des images d'entrée
        content_image = preprocess_image(content_path, max_dim=max_image_size)
        style_image = preprocess_image(style_path, max_dim=max_image_size)

        # 🔍 Affichage debug des images préprocessées
        if debug_mode:
            st.write("🔍 **Vérification des images préprocessées :**")
            col_debug1, col_debug2 = st.columns(2)
            
            with col_debug1:
                debug_content = deprocess_image(content_image.numpy())
                st.image(debug_content, caption=f"Contenu ({debug_content.shape[1]}×{debug_content.shape[0]})", width=200)
                
            with col_debug2:
                debug_style = deprocess_image(style_image.numpy())
                st.image(debug_style, caption=f"Style ({debug_style.shape[1]}×{debug_style.shape[0]})", width=200)

        # 🎯 Extraction des cibles (ce qu'on veut atteindre)
        # Analogie : Prendre des "mesures" de l'œuvre de référence
        style_targets = extractor(style_image)['style']
        content_targets = extractor(content_image)['content']

        # 🎨 Initialisation de l'image de travail
        # On commence avec l'image de contenu et on la modifie progressivement
        image = tf.Variable(content_image, dtype=tf.float32)
        
        # ⚙️ Configuration de l'optimiseur Adam
        # Analogie : Régler les paramètres du "pinceau intelligent"
        opt = tf.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=beta1,  # Mémoire des gradients
            beta_2=beta2,  # Mémoire de la variance
            epsilon=epsilon  # Stabilité numérique
        )

        # 📊 Interface de progression
        progress_bar = st.progress(0)
        status_text = st.empty()
        loss_placeholder = st.empty()
        preview_placeholder = st.empty()
        
        def train_step(image):
            """
            🎯 Une étape d'amélioration
            
            Analogie : Un coup de pinceau guidé par l'intelligence artificielle.
            L'IA regarde l'image actuelle, calcule ce qui ne va pas,
            et applique une petite correction.
            """
            with tf.GradientTape() as tape:
                # 🔍 Analyse de l'image actuelle
                outputs = extractor(image)
                
                # ⚖️ Calcul de l'erreur (perte)
                loss = style_content_loss(
                    outputs, style_targets, content_targets, 
                    style_weight, content_weight
                )

            # 📐 Calcul des gradients (direction d'amélioration)
            grad = tape.gradient(loss, image)
            
            # 🎨 Application de la correction
            opt.apply_gradients([(grad, image)])
            
            # 🛡️ Maintien des valeurs dans [0,1]
            image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))
            
            return loss

        # 🎨 Boucle principale d'amélioration artistique
        st.write(f"🎨 Début du transfert de style avec {iterations} itérations...")
        
        best_loss = float('inf')
        best_image = None
        
        for i in range(iterations):
            try:
                # 🎯 Une étape d'amélioration
                loss = train_step(image)
                loss_value = float(loss)
                
                # 📈 Suivi du meilleur résultat
                if loss_value < best_loss:
                    best_loss = loss_value
                    best_image = image.numpy().copy()
                
                # 📊 Mise à jour de l'interface
                progress = (i + 1) / iterations
                progress_bar.progress(progress)
                
                # 📈 Affichage détaillé du statut
                status_text.markdown(f"""
                **Itération {i+1}/{iterations}**
                - Perte actuelle: {loss_value:.4f}
                - Meilleure perte: {best_loss:.4f}
                - Progression: {progress:.1%}
                """)
                
                # 🔍 Aperçu périodique en mode debug
                if debug_mode and (i + 1) % preview_frequency == 0:
                    preview_img = deprocess_image(image.numpy())
                    with preview_placeholder.container():
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(preview_img, caption=f"Aperçu - Itération {i+1}", width=300)
                        with col2:
                            st.metric("Perte", f"{loss_value:.4f}", f"{loss_value - best_loss:.4f}")
                
            except Exception as e:
                st.error(f"❌ Erreur à l'itération {i+1}: {str(e)}")
                break

        # 🧹 Nettoyage de l'interface
        progress_bar.empty()
        status_text.empty()
        preview_placeholder.empty()
        
        # 🏆 Retour du meilleur résultat trouvé
        return tf.constant(best_image) if best_image is not None else image

    # 🖼️ Interface utilisateur pour le transfert de style
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📸 Image de Contenu")
        st.write("*L'image que vous voulez transformer*")
        content_file = st.file_uploader(
            "Choisissez l'image de contenu", 
            type=["jpg", "png", "jpeg"], 
            key="content",
            help="Votre photo personnelle qui sera stylisée"
        )
        
        if content_file:
            st.image(content_file, caption="Image de contenu", use_column_width=True)
            if show_diagnostics:
                diagnose_image(content_file, "contenu")
    
    with col2:
        st.markdown("### 🎨 Image de Style")
        st.write("*L'œuvre d'art dont vous voulez copier le style*")
        style_file = st.file_uploader(
            "Choisissez l'image de style", 
            type=["jpg", "png", "jpeg"], 
            key="style",
            help="Une peinture, dessin ou œuvre d'art dont vous aimez le style"
        )
        
        if style_file:
            st.image(style_file, caption="Image de style", use_column_width=True)
            if show_diagnostics:
                diagnose_image(style_file, "style")
    
    # 🎨 Configuration rapide prédéfinie
    st.markdown("### ⚡ Configurations Rapides")
    col_presets1, col_presets2, col_presets3 = st.columns(3)
    
    with col_presets1:
        if st.button("🖼️ Portrait Artistique"):
            # Configuration optimale pour portraits
            st.session_state.update({
                'style_weight': 5e3,
                'content_weight': 1e4, 
                'learning_rate': 0.008,
                'iterations': 150
            })
            st.success("Configuration portrait appliquée !")
    
    with col_presets2:
        if st.button("🏞️ Paysage Stylisé"):
            # Configuration optimale pour paysages
            st.session_state.update({
                'style_weight': 8e3,
                'content_weight': 1e3,
                'learning_rate': 0.012, 
                'iterations': 100
            })
            st.success("Configuration paysage appliquée !")
    
    with col_presets3:
        if st.button("⚡ Test Rapide"):
            # Configuration pour test rapide
            st.session_state.update({
                'style_weight': 1e4,
                'content_weight': 1e3,
                'learning_rate': 0.02,
                'iterations': 50
            })
            st.success("Configuration test appliquée !")
    
    # 📊 Prédiction de temps de calcul
    if content_file and style_file:
        # Estimation basée sur la taille et les itérations
        estimated_time = (max_image_size / 512) ** 2 * iterations * 0.05
        st.info(f"⏱️ Temps estimé : {estimated_time:.1f} minutes")
        
        if estimated_time > 10:
            st.warning("⚠️ Temps long prévu. Considérez réduire la taille d'image ou les itérations.")

    # 🚀 Bouton principal de lancement
    if st.button("🎨 Lancer le Transfert de Style", type="primary", use_container_width=True):
        if content_file and style_file:
            # 🎬 Début du processus
            start_time = st.empty()
            current_time = datetime.now().strftime('%H:%M:%S')
            start_time.write(f"🚀 **Démarrage du transfert de style à {current_time}**")
            
            with st.spinner('🎨 Transfert de style en cours... Votre IA crée une œuvre d\'art !'):
                try:
                    process_start = time.time()
                    
                    # 🎨 Exécution du transfert de style
                    stylized_image = perform_style_transfer(
                        content_file, style_file, 
                        style_weight, content_weight, iterations
                    )
                    
                    if stylized_image is not None:
                        # ⏱️ Calcul du temps écoulé
                        process_time = time.time() - process_start
                        
                        # 🖼️ Conversion et affichage du résultat
                        result_image = deprocess_image(stylized_image.numpy())
                        
                        # 🎯 Section des résultats
                        st.markdown("## 🎯 Résultat du Transfert de Style")
                        
                        # 📊 Comparaison avant/après
                        col_before, col_after = st.columns(2)
                        
                        with col_before:
                            st.markdown("#### 📸 Avant (Original)")
                            st.image(content_file, use_column_width=True)
                        
                        with col_after:
                            st.markdown("#### 🎨 Après (Stylisé)")
                            st.image(result_image, use_column_width=True)
                        
                        # 📈 Statistiques du processus
                        st.markdown("### 📊 Statistiques du Processus")
                        col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
                        
                        with col_stats1:
                            st.metric("⏱️ Temps", f"{process_time:.1f}s")
                        with col_stats2:
                            st.metric("🔄 Itérations", iterations)
                        with col_stats3:
                            st.metric("📏 Taille", f"{max_image_size}px")
                        with col_stats4:
                            efficiency = iterations / process_time if process_time > 0 else 0
                            st.metric("⚡ Vitesse", f"{efficiency:.1f} it/s")
                        
                        # 🎨 Informations sur la configuration utilisée
                        with st.expander("🔧 Configuration Utilisée"):
                            st.write(f"**Poids du style :** {style_weight:.0e}")
                            st.write(f"**Poids du contenu :** {content_weight:.0e}")
                            st.write(f"**Taux d'apprentissage :** {learning_rate}")
                            st.write(f"**Couches de style :** {', '.join(build_style_layers_list())}")
                            st.write(f"**Optimiseur Adam :** β₁={beta1}, β₂={beta2}, ε={epsilon:.0e}")
                        
                        # 💾 Téléchargement du résultat
                        result_pil = Image.fromarray(result_image)
                        
                        # 🎨 Options de sauvegarde
                        st.markdown("### 💾 Téléchargement")
                        
                        col_download1, col_download2 = st.columns(2)
                        
                        with col_download1:
                            # PNG haute qualité
                            buf_png = io.BytesIO()
                            result_pil.save(buf_png, format='PNG', optimize=True)
                            
                            st.download_button(
                                label="📥 Télécharger PNG (Haute Qualité)",
                                data=buf_png.getvalue(),
                                file_name=f"saffire_stylized_{int(time.time())}.png",
                                mime="image/png",
                                use_container_width=True
                            )
                        
                        with col_download2:
                            # JPEG optimisé
                            buf_jpg = io.BytesIO()
                            result_pil.save(buf_jpg, format='JPEG', quality=95, optimize=True)
                            
                            st.download_button(
                                label="📥 Télécharger JPEG (Optimisé)",
                                data=buf_jpg.getvalue(),
                                file_name=f"saffire_stylized_{int(time.time())}.jpg",
                                mime="image/jpeg",
                                use_container_width=True
                            )
                        
                        # 🎉 Message de succès avec conseils
                        st.success("✅ Transfert de style terminé avec succès !")
                        
                        # 💡 Conseils pour améliorer
                        st.markdown("### 💡 Conseils pour améliorer le résultat")
                        st.info("""
                        **Pour plus de style artistique :** Augmentez le poids du style
                        
                        **Pour préserver plus l'original :** Augmentez le poids du contenu
                        
                        **Si l'image semble floue :** Augmentez le nombre d'itérations
                        
                        **Si les couleurs sont ternes :** Ajustez la saturation dans les paramètres avancés
                        """)
                        
                        # 📊 Analyse de qualité automatique
                        st.markdown("### 🔍 Analyse de Qualité")
                        
                        # Calculs simples de qualité
                        original_array = np.array(Image.open(content_file).resize((256, 256)))
                        result_resized = np.array(result_pil.resize((256, 256)))
                        
                        # Mesure de similarité (simple MSE)
                        mse = np.mean((original_array.astype(float) - result_resized.astype(float)) ** 2)
                        similarity = max(0, 100 - mse / 100)  # Score approximatif
                        
                        col_quality1, col_quality2 = st.columns(2)
                        with col_quality1:
                            st.metric("🎯 Similarité contenu", f"{similarity:.1f}%")
                        with col_quality2:
                            color_variance = np.var(result_resized)
                            st.metric("🌈 Richesse couleurs", f"{color_variance:.0f}")
                    
                    else:
                        st.error("❌ Erreur lors du transfert de style.")
                        
                except Exception as e:
                    st.error(f"❌ Erreur lors du transfert de style : {str(e)}")
                    
                    # 🔧 Suggestions de résolution
                    st.markdown("### 🔧 Suggestions de résolution :")
                    st.write("1. Vérifiez que vos images sont au format JPG/PNG")
                    st.write("2. Essayez de réduire la taille maximale d'image") 
                    st.write("3. Réduisez le nombre d'itérations pour un test")
                    st.write("4. Vérifiez qu'au moins une couche de style est activée")
                    
        else:
            st.warning("⚠️ Veuillez charger une image de contenu ET une image de style.")

# 🔄 MODULE DE TRANSFORMATION INVERSE
elif main_mode == "Transformation Inverse":
    st.markdown("## 🔄 Module de Transformation Inverse")
    st.write("*Récupérez le contenu original ou extrayez le style d'une image stylisée*")
    
    # 🎛️ Hyperparamètres pour la transformation inverse
    st.sidebar.markdown("### 🔄 Paramètres de Transformation Inverse")
    
    # Type de transformation inverse
    inverse_mode = st.sidebar.radio(
        "Type de transformation:",
        ["Extraction de Contenu", "Extraction de Style", "Déstylisation Complète"],
        help="Choisissez quel aspect récupérer de l'image stylisée"
    )
    
    # Intensité de la transformation inverse
    inverse_strength = st.sidebar.slider(
        "Intensité de récupération:",
        0.1, 2.0, 1.0, step=0.1,
        help="Plus élevé = récupération plus agressive"
    )
    
    # Nombre d'itérations pour l'optimisation inverse
    inverse_iterations = st.sidebar.number_input(
        "Itérations d'optimisation:",
        min_value=50, max_value=500, value=200, step=25,
        help="Plus d'itérations = meilleure qualité mais plus lent"
    )
    
    # Paramètres avancés
    st.sidebar.markdown("### 🔧 Paramètres Avancés")
    
    inverse_learning_rate = st.sidebar.slider(
        "Taux d'apprentissage inverse:",
        0.001, 0.05, 0.01, step=0.001,
        help="Vitesse de récupération - plus lent mais plus stable"
    )
    
    content_preservation = st.sidebar.slider(
        "Préservation structure:",
        0.0, 2.0, 1.0, step=0.1,
        help="Force de préservation de la structure originale"
    )
    
    # Régularisation pour éviter les artefacts
    regularization_weight = st.sidebar.slider(
        "Régularisation (anti-artefacts):",
        0.0, 0.1, 0.01, step=0.005,
        help="Évite les pixels aberrants et lisse le résultat"
    )
    
    # Type de perte pour l'optimisation inverse
    loss_type = st.sidebar.selectbox(
        "Type de perte d'optimisation:",
        ["MSE", "Perceptual", "Mixed"],
        help="MSE=simple | Perceptual=réaliste | Mixed=équilibré"
    )
    
    # Options de post-traitement
    st.sidebar.markdown("#### 🎨 Post-traitement")
    
    enhance_details = st.sidebar.checkbox(
        "Amélioration des détails",
        value=True,
        help="Renforce les contours et textures récupérés"
    )
    
    noise_reduction = st.sidebar.slider(
        "Réduction du bruit:",
        0.0, 1.0, 0.3, step=0.1,
        help="Lisse les artefacts de reconstruction"
    )
    
    color_correction = st.sidebar.checkbox(
        "Correction colorimétrique",
        value=True,
        help="Ajuste automatiquement les couleurs récupérées"
    )

    def create_inverse_model(target_size=(512, 512)):
        """
        🔄 Création du modèle de transformation inverse
        
        Analogie : Comme un "détective artistique" qui analyse une œuvre
        pour retrouver les éléments originaux cachés dessous.
        
        Le modèle utilise un autoencoder avec skip connections pour
        reconstruire le contenu ou style original.
        
        Args:
            target_size: Taille de l'image de sortie
            
        Returns:
            Modèle TensorFlow pour transformation inverse
        """
        from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Concatenate, UpSampling2D
        from tensorflow.keras.layers import LeakyReLU, BatchNormalization
        
        # 📥 Entrée : Image stylisée
        inputs = Input(shape=(*target_size, 3), name='stylized_input')
        
        # 🔽 Encodeur - "Analyse de l'image stylisée"
        # Analogie : Décomposer l'image en éléments compréhensibles
        
        # Block 1: Extraction des features de base
        e1 = Conv2D(64, 3, padding='same', name='encoder_1')(inputs)
        e1 = LeakyReLU(alpha=0.2)(e1)
        e1 = BatchNormalization()(e1)
        
        # Block 2: Features intermédiaires
        e2 = Conv2D(128, 3, strides=2, padding='same', name='encoder_2')(e1)
        e2 = LeakyReLU(alpha=0.2)(e2)
        e2 = BatchNormalization()(e2)
        
        # Block 3: Features profondes
        e3 = Conv2D(256, 3, strides=2, padding='same', name='encoder_3')(e2)
        e3 = LeakyReLU(alpha=0.2)(e3)
        e3 = BatchNormalization()(e3)
        
        # Block 4: Représentation latente
        e4 = Conv2D(512, 3, strides=2, padding='same', name='encoder_4')(e3)
        e4 = LeakyReLU(alpha=0.2)(e4)
        e4 = BatchNormalization()(e4)
        
        # 🔼 Décodeur - "Reconstruction du contenu original"
        # Analogie : Remonter du puzzle décomposé vers l'image originale
        
        # Block 1: Début de reconstruction
        d1 = Conv2DTranspose(256, 3, strides=2, padding='same', name='decoder_1')(e4)
        d1 = LeakyReLU(alpha=0.2)(d1)
        d1 = BatchNormalization()(d1)
        d1 = Concatenate()([d1, e3])  # Skip connection pour préserver les détails
        
        # Block 2: Reconstruction intermédiaire
        d2 = Conv2DTranspose(128, 3, strides=2, padding='same', name='decoder_2')(d1)
        d2 = LeakyReLU(alpha=0.2)(d2)
        d2 = BatchNormalization()(d2)
        d2 = Concatenate()([d2, e2])  # Skip connection
        
        # Block 3: Reconstruction finale
        d3 = Conv2DTranspose(64, 3, strides=2, padding='same', name='decoder_3')(d2)
        d3 = LeakyReLU(alpha=0.2)(d3)
        d3 = BatchNormalization()(d3)
        d3 = Concatenate()([d3, e1])  # Skip connection
        
        # 🎯 Sortie finale
        outputs = Conv2D(3, 3, activation='tanh', padding='same', name='output')(d3)
        
        # Création du modèle
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='InverseTransformModel')
        
        return model

    def perceptual_loss(y_true, y_pred, vgg_model):
        """
        👁️ Calcul de la perte perceptuelle
        
        Analogie : Au lieu de comparer pixel par pixel (comme un robot),
        on compare ce que "voit" un expert (réseau VGG19 pré-entraîné).
        
        Args:
            y_true: Image cible
            y_pred: Image prédite
            vgg_model: Modèle VGG19 pour extraction de features
            
        Returns:
            Perte perceptuelle basée sur les features VGG19
        """
        # Préprocessing pour VGG19
        y_true_vgg = preprocess_input(y_true * 255.0)
        y_pred_vgg = preprocess_input(y_pred * 255.0)
        
        # Extraction des features
        true_features = vgg_model(y_true_vgg)
        pred_features = vgg_model(y_pred_vgg)
        
        # Calcul de la différence perceptuelle
        loss = 0
        for true_feat, pred_feat in zip(true_features, pred_features):
            loss += tf.reduce_mean(tf.square(true_feat - pred_feat))
        
        return loss

    def total_variation_loss(image):
        """
        🌊 Perte de variation totale pour réduction du bruit
        
        Analogie : Comme lisser une surface rugueuse pour la rendre plus naturelle.
        Cette fonction pénalise les variations brutales entre pixels voisins.
        
        Args:
            image: Image à lisser
            
        Returns:
            Perte de variation totale
        """
        # Différences horizontales et verticales
        h_diff = image[:, 1:, :, :] - image[:, :-1, :, :]
        w_diff = image[:, :, 1:, :] - image[:, :, :-1, :]
        
        # Somme des variations
        return tf.reduce_mean(tf.square(h_diff)) + tf.reduce_mean(tf.square(w_diff))

    def perform_inverse_transform(stylized_image, reference_image=None):
        """
        🔄 Exécution de la transformation inverse
        
        Analogie : Comme un restaurateur d'art qui enlève les couches
        de peinture ajoutées pour retrouver l'œuvre originale en dessous.
        
        Args:
            stylized_image: Image stylisée à transformer
            reference_image: Image de référence (optionnelle)
            
        Returns:
            Image avec transformation inverse appliquée
        """
        # 🏗️ Préparation du modèle VGG19 pour perte perceptuelle
        vgg = VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False
        
        # Sélection des couches pour perte perceptuelle
        feature_layers = ['block1_conv2', 'block2_conv2', 'block3_conv3']
        feature_outputs = [vgg.get_layer(name).output for name in feature_layers]
        feature_model = tf.keras.Model([vgg.input], feature_outputs)
        
        # 🎯 Initialisation de l'image de travail
        # On commence avec l'image stylisée et on la modifie progressivement
        target_image = tf.Variable(stylized_image, dtype=tf.float32)
        
        # ⚙️ Optimiseur pour la transformation inverse
        optimizer = tf.optimizers.Adam(learning_rate=inverse_learning_rate)
        
        # 📊 Interface de progression
        progress_bar = st.progress(0)
        status_text = st.empty()
        preview_placeholder = st.empty()
        
        @tf.function
        def inverse_step():
            """
            🔄 Une étape d'optimisation inverse
            
            Calcule et applique une correction pour se rapprocher
            de l'objectif de transformation inverse.
            """
            with tf.GradientTape() as tape:
                # 📊 Calcul des différentes pertes
                total_loss = 0
                
                if loss_type in ["MSE", "Mixed"]:
                    # 📏 Perte MSE simple (pixel par pixel)
                    if reference_image is not None:
                        mse_loss = tf.reduce_mean(tf.square(target_image - reference_image))
                        total_loss += mse_loss * inverse_strength
                
                if loss_type in ["Perceptual", "Mixed"]:
                    # 👁️ Perte perceptuelle (basée sur la vision)
                    if reference_image is not None:
                        perc_loss = perceptual_loss(reference_image, target_image, feature_model)
                        total_loss += perc_loss * inverse_strength * 0.1
                
                # 🛡️ Régularisation pour éviter les artefacts
                if regularization_weight > 0:
                    tv_loss = total_variation_loss(target_image)
                    total_loss += tv_loss * regularization_weight
                
                # 🏗️ Préservation de la structure si demandée
                if content_preservation > 0:
                    structure_loss = tf.reduce_mean(tf.square(
                        tf.image.sobel_edges(target_image) - 
                        tf.image.sobel_edges(stylized_image)
                    ))
                    total_loss += structure_loss * content_preservation
            
            # 📐 Calcul et application des gradients
            gradients = tape.gradient(total_loss, target_image)
            optimizer.apply_gradients([(gradients, target_image)])
            
            # 🛡️ Maintien des valeurs dans [0,1]
            target_image.assign(tf.clip_by_value(target_image, 0.0, 1.0))
            
            return total_loss
        
        # 🔄 Boucle d'optimisation inverse
        st.write(f"🔄 Début de la transformation inverse ({inverse_mode})...")
        
        best_loss = float('inf')
        best_image = None
        
        for i in range(inverse_iterations):
            try:
                # 🎯 Une étape d'amélioration
                loss = inverse_step()
                loss_value = float(loss)
                
                # 📈 Suivi du meilleur résultat
                if loss_value < best_loss:
                    best_loss = loss_value
                    best_image = target_image.numpy().copy()
                
                # 📊 Mise à jour de l'interface
                progress = (i + 1) / inverse_iterations
                progress_bar.progress(progress)
                
                status_text.markdown(f"""
                **Itération {i+1}/{inverse_iterations}**
                - Perte: {loss_value:.6f}
                - Meilleure: {best_loss:.6f}
                - Mode: {inverse_mode}
                """)
                
                # 🔍 Aperçu périodique
                if (i + 1) % 25 == 0:
                    preview_img = deprocess_image(target_image.numpy())
                    with preview_placeholder.container():
                        st.image(preview_img, caption=f"Progression - Itération {i+1}", width=300)
                
            except Exception as e:
                st.error(f"❌ Erreur à l'itération {i+1}: {str(e)}")
                break
        
        # 🧹 Nettoyage interface
        progress_bar.empty()
        status_text.empty()
        preview_placeholder.empty()
        
        # 🎨 Post-traitement si demandé
        final_image = tf.constant(best_image) if best_image is not None else target_image
        
        if enhance_details:
            # 🔍 Amélioration des détails via filtre passe-haut
            kernel = tf.constant([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=tf.float32)
            kernel = tf.reshape(kernel, [3, 3, 1, 1])
            kernel = tf.tile(kernel, [1, 1, 3, 1])  # Pour les 3 canaux RGB
            
            details = tf.nn.conv2d(final_image, kernel, strides=[1, 1, 1, 1], padding='SAME')
            final_image = final_image + details * 0.1  # Ajout subtil des détails
        
        if noise_reduction > 0:
            # 🌊 Réduction du bruit par filtrage gaussien
            final_image = tf.image.gaussian_filter2d(final_image, sigma=noise_reduction)
        
        # 🛡️ Clipping final
        final_image = tf.clip_by_value(final_image, 0.0, 1.0)
        
        return final_image

    # 🖼️ Interface utilisateur
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎨 Image Stylisée")
        st.write("*L'image stylisée dont vous voulez extraire des éléments*")
        stylized_file = st.file_uploader(
            "Choisissez l'image stylisée", 
            type=["jpg", "png", "jpeg"], 
            key="stylized",
            help="L'image qui a subi un transfert de style"
        )
        
        if stylized_file:
            st.image(stylized_file, caption="Image stylisée", use_column_width=True)
    
    with col2:
        st.markdown("### 📸 Image de Référence (Optionnelle)")
        st.write("*L'image originale pour guider la transformation inverse*")
        reference_file = st.file_uploader(
            "Choisissez l'image de référence", 
            type=["jpg", "png", "jpeg"], 
            key="reference",
            help="L'image originale avant stylisation (optionnel)"
        )
        
        if reference_file:
            st.image(reference_file, caption="Image de référence", use_column_width=True)
    
    # ℹ️ Explication du mode sélectionné
    if inverse_mode == "Extraction de Contenu":
        st.info("🎯 **Mode Extraction de Contenu** : Récupère les formes et structures originales en supprimant les effets de style")
    elif inverse_mode == "Extraction de Style":
        st.info("🎨 **Mode Extraction de Style** : Isole les éléments stylistiques (textures, coups de pinceau) pour les réutiliser")
    else:
        st.info("🔄 **Mode Déstylisation Complète** : Tente de retrouver l'image originale complète avant stylisation")
    
    # 🚀 Bouton de lancement
    if st.button("🔄 Lancer la Transformation Inverse", type="primary", use_container_width=True):
        if stylized_file:
            with st.spinner(f'🔄 Transformation inverse en cours ({inverse_mode})...'):
                try:
                    start_time = time.time()
                    
                    # 🛠️ Préparation des images
                    stylized_image = preprocess_image(stylized_file, max_dim=512)
                    reference_image = None
                    
                    if reference_file:
                        reference_image = preprocess_image(reference_file, max_dim=512)
                        # Redimensionner pour correspondre à l'image stylisée
                        ref_shape = tf.shape(reference_image)
                        sty_shape = tf.shape(stylized_image)
                        if ref_shape[1] != sty_shape[1] or ref_shape[2] != sty_shape[2]:
                            reference_image = tf.image.resize(reference_image, [sty_shape[1], sty_shape[2]])
                    
                    # 🔄 Exécution de la transformation inverse
                    result_image = perform_inverse_transform(stylized_image, reference_image)
                    
                    # ⏱️ Calcul du temps
                    process_time = time.time() - start_time
                    
                    # 🖼️ Affichage des résultats
                    result_array = deprocess_image(result_image.numpy())
                    
                    st.markdown("## 🎯 Résultat de la Transformation Inverse")
                    
                    # 📊 Comparaison avant/après
                    if reference_file:
                        col_original, col_stylized, col_recovered = st.columns(3)
                        
                        with col_original:
                            st.markdown("#### 📸 Original")
                            st.image(reference_file, use_column_width=True)
                        
                        with col_stylized:
                            st.markdown("#### 🎨 Stylisé")
                            st.image(stylized_file, use_column_width=True)
                        
                        with col_recovered:
                            st.markdown("#### 🔄 Récupéré")
                            st.image(result_array, use_column_width=True)
                    else:
                        col_before, col_after = st.columns(2)
                        
                        with col_before:
                            st.markdown("#### 🎨 Avant (Stylisé)")
                            st.image(stylized_file, use_column_width=True)
                        
                        with col_after:
                            st.markdown("#### 🔄 Après (Transformé)")
                            st.image(result_array, use_column_width=True)
                    
                    # 📈 Statistiques
                    st.markdown("### 📊 Statistiques du Processus")
                    col_stats1, col_stats2, col_stats3 = st.columns(3)
                    
                    with col_stats1:
                        st.metric("⏱️ Temps", f"{process_time:.1f}s")
                    with col_stats2:
                        st.metric("🔄 Itérations", inverse_iterations)
                    with col_stats3:
                        efficiency = inverse_iterations / process_time if process_time > 0 else 0
                        st.metric("⚡ Vitesse", f"{efficiency:.1f} it/s")
                    
                    # 💾 Téléchargement
                    result_pil = Image.fromarray(result_array)
                    buf = io.BytesIO()
                    result_pil.save(buf, format='PNG')
                    
                    st.download_button(
                        label="📥 Télécharger le Résultat",
                        data=buf.getvalue(),
                        file_name=f"saffire_inverse_{inverse_mode.lower().replace(' ', '_')}_{int(time.time())}.png",
                        mime="image/png",
                        use_container_width=True
                    )
                    
                    # ✅ Message de succès
                    st.success(f"✅ Transformation inverse ({inverse_mode}) terminée avec succès !")
                    
                    # 💡 Conseils d'amélioration
                    st.markdown("### 💡 Conseils pour Améliorer")
                    if inverse_mode == "Extraction de Contenu":
                        st.info("💡 Si le contenu n'est pas assez récupéré, augmentez l'intensité de récupération ou utilisez une image de référence")
                    elif inverse_mode == "Extraction de Style":
                        st.info("💡 Pour isoler mieux le style, essayez de réduire la préservation de structure")
                    else:
                        st.info("💡 Pour une meilleure déstylisation, fournissez l'image originale comme référence")
                
                except Exception as e:
                    st.error(f"❌ Erreur lors de la transformation inverse : {str(e)}")
                    st.write("🔧 **Solutions possibles :**")
                    st.write("- Vérifiez le format de votre image (JPG/PNG)")
                    st.write("- Réduisez le nombre d'itérations pour un test")
                    st.write("- Essayez avec une image plus petite")
        else:
            st.warning("⚠️ Veuillez charger au minimum une image stylisée.")
    
    # 📚 Section d'information
    with st.expander("ℹ️ Comment fonctionne la Transformation Inverse ?"):
        st.markdown("""
        ### 🔄 Principe de la Transformation Inverse
        
        La transformation inverse tente de "défaire" les effets du transfert de style pour récupérer 
        les éléments originaux cachés dans l'image stylisée.
        
        ### 🧠 Processus Technique
        
        **1. Analyse de l'Image Stylisée** 🔍
        - Décomposition en features via un réseau encodeur-décodeur
        - Identification des éléments de contenu vs style
        - Séparation des composantes visuelles
        
        **2. Optimisation Inverse** ⚙️
        - Utilisation de gradients pour "remonter le temps"
        - Minimisation de la différence avec l'objectif
        - Régularisation pour éviter les artefacts
        
        **3. Reconstruction** 🏗️
        - Assemblage des éléments récupérés
        - Post-traitement pour améliorer la qualité
        - Lissage et correction des couleurs
        
        ### 🎯 Modes de Transformation
        
        **Extraction de Contenu** 📸
        - Récupère les formes et structures
        - Supprime les textures artistiques
        - Idéal pour retrouver la géométrie originale
        
        **Extraction de Style** 🎨
        - Isole les éléments stylistiques
        - Garde les textures et coups de pinceau
        - Utile pour créer des templates de style
        
        **Déstylisation Complète** 🔄
        - Tente de retrouver l'image originale
        - Combine récupération de contenu et suppression de style
        - Meilleur résultat avec image de référence
        
        ### ⚙️ Paramètres Clés
        
        **Intensité de Récupération** 💪
        - Contrôle la force de la transformation inverse
        - Plus élevé = récupération plus agressive
        - Risque : artefacts si trop élevé
        
        **Préservation Structure** 🏗️
        - Maintient la géométrie de base
        - Important pour l'extraction de contenu
        - Évite les déformations excessives
        
        **Régularisation** 🛡️
        - Évite les pixels aberrants
        - Lisse le résultat final
        - Équilibrer avec la qualité des détails
        
        ### 💡 Conseils d'Utilisation
        
        **Pour de Meilleurs Résultats** ✨
        - Utilisez l'image originale comme référence si disponible
        - Commencez avec des paramètres conservateurs
        - Augmentez progressivement l'intensité
        - Testez différents modes selon votre objectif
        
        **Limitations** ⚠️
        - La transformation inverse n'est jamais parfaite
        - Certaines informations sont définitivement perdues
        - La qualité dépend du niveau de stylisation initial
        - Plus l'image était stylisée, plus difficile la récupération
        """)

# 📚 Section d'information détaillée
    with st.expander("ℹ️ Comment fonctionne le Transfert de Style Neural ?"):
        st.markdown("""
        ### 🧠 Principe de Base
        
        Le transfert de style neural utilise l'intelligence artificielle pour **séparer** et **recombiner** 
        deux aspects d'une image :
        
        1. **Le Contenu** 📸 : La structure, les formes, les objets (QUOI est dans l'image)
        2. **Le Style** 🎨 : Les textures, couleurs, coups de pinceau (COMMENT c'est peint)
        
        ### 🔬 Le Processus Technique
        
        **Étape 1 - Analyse** 🔍
        - L'IA "regarde" votre photo avec les "yeux" d'un réseau VGG19 pré-entraîné
        - Elle identifie les formes et objets (contenu) dans les couches profondes
        - Elle analyse les textures et patterns (style) dans plusieurs couches
        
        **Étape 2 - Extraction des "Signatures"** 📊
        - **Contenu** : Features maps de la couche block5_conv2 (compréhension sémantique)
        - **Style** : Matrices de Gram des couches block1 à block5 (corrélations de textures)
        
        **Étape 3 - Optimisation Itérative** 🎯
        - L'IA commence avec votre photo originale
        - À chaque itération, elle la modifie légèrement pour :
          - Garder le même contenu (fidélité à l'original)
          - Adopter le style de l'œuvre d'art (transformation artistique)
        - Le processus s'arrête quand l'équilibre optimal est trouvé
        
        ### ⚖️ Les Hyperparamètres Expliqués
        
        **Poids du Style vs Contenu** 🎚️
        - **Style élevé** → Plus artistique, moins ressemblant
        - **Contenu élevé** → Plus fidèle, moins stylisé
        - **Équilibre** → Transformation harmonieuse
        
        **Nombre d'Itérations** 🔄
        - Comme un peintre qui affine son œuvre
        - Plus d'itérations = meilleur résultat mais plus lent
        - 50-100 pour test, 200-500 pour qualité finale
        
        **Taux d'Apprentissage** ⚡
        - Vitesse des "coups de pinceau" de l'IA
        - Trop rapide → instable, trop lent → convergence lente
        - 0.01 est généralement optimal
        
        ### 🎨 Conseils d'Utilisation
        
        **Choix des Images** 📸
        - **Contenu** : Photos nettes, bien contrastées
        - **Style** : Œuvres d'art avec textures riches (Van Gogh, Picasso, etc.)
        
        **Premiers Tests** ⚡
        - Commencez avec la configuration "Test Rapide"
        - Ajustez selon le résultat obtenu
        - Expérimentez avec différents styles
        
        **Optimisation** 🎯
        - Portrait → Privilégier le contenu
        - Paysage → Équilibrer style/contenu  
        - Art abstrait → Privilégier le style
        """)
    
    # 🎭 Galerie d'exemples (si vous voulez ajouter des exemples)
    with st.expander("🖼️ Galerie d'Exemples et Inspirations"):
        st.markdown("""
        ### 🎨 Styles Artistiques Populaires
        
        **Impressionnisme** 🌅
        - Van Gogh, Monet, Renoir
        - Effet : Coups de pinceau visibles, couleurs vives
        - Idéal pour : Paysages, portraits
        
        **Cubisme** 🔷
        - Picasso, Braque
        - Effet : Formes géométriques, perspectives multiples
        - Idéal pour : Portraits, objets
        
        **Art Japonais** 🗾
        - Hokusai, style manga
        - Effet : Lignes nettes, couleurs plates
        - Idéal pour : Tous types d'images
        
        **Art Moderne** 🎭
        - Kandinsky, Mondrian
        - Effet : Abstraction, couleurs pures
        - Idéal pour : Créations artistiques audacieuses
        
        ### 💡 Astuces de Pro
        
        1. **Testez différents ratios style/contenu** pour le même couple d'images
        2. **Utilisez des styles contrastés** avec votre photo pour des effets saisissants
        3. **Les œuvres avec textures prononcées** donnent de meilleurs résultats
        4. **Combinez plusieurs passes** : style léger puis style prononcé
        5. **Post-traitez** : ajustez luminosité/contraste après le transfert
        """)
    
    # 🔧 Section de dépannage
    with st.expander("🛠️ Dépannage et Résolution de Problèmes"):
        st.markdown("""
        ### ❌ Problèmes Courants
        
        **"L'image reste floue ou déformée"** 🌫️
        - **Cause** : Trop d'itérations ou learning rate trop élevé
        - **Solution** : Réduire les itérations à 50-100, learning rate à 0.005
        
        **"Le style ne s'applique pas assez"** 🎨
        - **Cause** : Poids du style trop faible
        - **Solution** : Augmenter le poids du style à 1e5 ou plus
        
        **"L'image originale disparaît complètement"** 📸
        - **Cause** : Poids du contenu trop faible
        - **Solution** : Augmenter le poids du contenu à 1e4 ou plus
        
        **"Le processus est très lent"** ⏳
        - **Cause** : Image trop grande ou trop d'itérations
        - **Solution** : Réduire taille à 256px, limiter à 50-100 itérations
        
        **"Erreur de mémoire"** 💾
        - **Cause** : Image trop grande pour votre système
        - **Solution** : Utiliser 256px maximum, redémarrer l'application
        
        **"Couleurs étranges ou saturées"** 🌈
        - **Cause** : Problème de normalisation ou contraste
        - **Solution** : Ajuster saturation et contraste dans paramètres avancés
        
        ### 🔍 Diagnostic Auto
        
        Activez le **"Diagnostic détaillé"** pour voir :
        - Format et qualité de vos images
        - Problèmes potentiels détectés
        - Suggestions d'optimisation automatiques
        
        ### 🚀 Optimisation Performance
        
        **Pour des résultats plus rapides :**
        - Taille 256px, 50 itérations
        - Désactiver le mode débogage
        - Utiliser moins de couches de style
        
        **Pour la meilleure qualité :**
        - Taille 512px minimum
        - 200-500 itérations
        - Toutes les couches de style activées
        - Learning rate réduit (0.005)
        """)
    
    # 📊 Panneau de monitoring avancé (si debug activé)
    if debug_mode and content_file and style_file:
        st.markdown("### 🔬 Monitoring Avancé (Mode Debug)")
        
        # Informations système
        col_sys1, col_sys2, col_sys3 = st.columns(3)
        
        with col_sys1:
            st.metric("🖥️ Backend", "TensorFlow")
        with col_sys2:
            st.metric("🧠 Modèle", "VGG19 ImageNet")
        with col_sys3:
            st.metric("⚡ Device", "CPU" if not tf.config.list_physical_devices('GPU') else "GPU")

# 📝 Footer avec informations
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p><strong>SAFFIRE Detection System</strong> - Powered by TensorFlow & Streamlit</p>
    <p>🎨 Module Classification: Détection intelligente de feu et fumée</p>
    <p>🖼️ Module Style Transfer: Transformation artistique par IA</p>
    <p><em>Développé pour allier sécurité et créativité</em></p>
</div>
""", unsafe_allow_html=True)