import streamlit as st
import pandas as pd
import keras
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

st.set_page_config(page_title='🍎 Fruit Classification App 🍌', 
                   page_icon='🖼️', layout='wide', 
                   initial_sidebar_state='auto')

st.title('🍎 Fruits Classification App 🍌')
if st.sidebar.toggle("A propos de l'auteur"):
    with st.expander("Auteur", True) : 
        c_1, c_2 = st.columns([1,2])
        with c_1 :
            st.image("About the author.png")
        with c_2 : 
            st.header(""" **S. Abraham Z. KOLOBOE**""")
            st.markdown("""
                
                *:blue[Data Scientist | Ingénieur en Mathématiques et Modélisation]*
                Bonjour,
                Je suis Abraham, un Data Scientist et Ingénieur en Mathématiques et Modélisation. 
                Mon expertise se situe dans les domaines des sciences de données et de l'intelligence artificielle. 
                Avec une approche technique et concise, je m'engage à fournir des solutions efficaces et précises dans mes projets.
                        
                * Email : <abklb27@gmail.com>
                * WhatsApp : +229 91 83 84 21
                * Linkedin : [Abraham KOLOBOE](https://www.linkedin.com/in/abraham-zacharie-koloboe-data-science-ia-generative-llms-machine-learning)
                    
                                    """)
with st.sidebar : 
    st.markdown("""
        ## Auteur
        :blue[Abraham KOLOBOE]
        * Email : <abklb27@gmail.com>
        * WhatsApp : +229 91 83 84 21
        * Linkedin : [Abraham KOLOBOE](https://www.linkedin.com/in/abraham-zacharie-koloboe-data-science-ia-generative-llms-machine-learning)
                    """)
    
def get_class_names():
    with open('class_names.txt', 'r') as file:
        class_names = file.readlines()
    return [class_name.strip() for class_name in class_names]

uploaded_file = st.file_uploader("Choose an image...",
                                 type=["jpg","webp", "png", "jpeg"])

if uploaded_file is not None:
    image = keras.preprocessing.image.load_img(uploaded_file, target_size=(100, 100))
    logging.info('Image successfully resized')
    
    model = keras.models.load_model('model/best_model_cnn.keras')
    logging.info('Model loaded successfully')
    
    prediction = model.predict(np.expand_dims(image, axis=0))
    logging.info('Prediction made')
    
    class_names = get_class_names()
    predicted_label = class_names[np.argmax(prediction, axis=1)[0]]
    logging.info('Label predicted')
    
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.image(image, caption='Uploaded Image.', use_column_width=True)
    with col2:
        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                st.write("""**:blue[Prediction]**""")
                st.write(f""":green[{predicted_label}]""")
        confidence_score = np.max(prediction)
        #st.write(f'Confidence Score: {confidence_score:.2f}')
        with c2:
            with st.container(border=True):
                st.write(""":blue[Confidence Score]""")

                st.markdown(f""":green[{100*confidence_score:.2f} %]""")
            
        top_k = 5
        top_k_indices = np.argsort(prediction[0])[-top_k:][::-1]
        top_k_labels = [class_names[i] for i in top_k_indices]
        top_k_scores = prediction[0][top_k_indices]
        
        # Create a DataFrame for the top-k predictions
        df = pd.DataFrame({
            'Class': top_k_labels,
            'Confidence': top_k_scores
        }).set_index('Class')
        
        # Sort the DataFrame by confidence in descending order
        df = df.sort_values(by='Confidence', ascending=False)
        
        # Highlight the predicted class
        colors = ['green' if label == predicted_label else 'red' for label in df.index]
        with st.container(border=True):
            st.write("""**:blue[Top-5 Predictions]**""")
            # Plot the bar chart with custom colors
            st.bar_chart(df.style.apply(lambda x: ['background-color: ' + color for color in colors], axis=0))
else : 
    
    st.markdown("""
    ## Bienvenue dans l'application de classification de fruits ! 🍇🍉🍍
    
    Cette application utilise un modèle de deep learning pour classifier des images de fruits. Vous pouvez télécharger une image de fruit, et l'application affichera la classe correspondante ainsi que le score de confiance de la prédiction. De plus, un graphique en barres montrera les scores de confiance des 5 classes les plus proches.
    
    ### Fonctionnalités
    - 📷 **Téléchargement d'image** : Chargez une image de fruit au format JPG, PNG ou JPEG.
    - 🧠 **Prédiction** : Le modèle de deep learning prédit la classe du fruit.
    - 📊 **Visualisation** : Affichez un graphique en barres des scores de confiance des 5 classes les plus proches.
    
    ### Comment utiliser l'application
    1. **Téléchargez une image** : Cliquez sur le bouton "Choose an image..." et sélectionnez une image de fruit depuis votre appareil.
    2. **Affichage de l'image** : L'image téléchargée sera affichée dans la première colonne.
    3. **Résultats de la prédiction** : La classe prédite et le score de confiance seront affichés dans la deuxième colonne.
    4. **Graphique en barres** : Un graphique en barres montrera les scores de confiance des 5 classes les plus proches.
    
    
    Profitez de l'application et amusez-vous à classifier vos fruits préférés ! 🍇🍉🍍
    
    N'hésitez pas à contribuer ou à signaler des problèmes via les issues du dépôt [GitHub](https://github.com/abrahamkoloboe27/Fruits-Recognition-Training) .
    """)