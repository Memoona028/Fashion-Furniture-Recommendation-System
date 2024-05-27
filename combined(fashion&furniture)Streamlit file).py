# -*- coding: utf-8 -*-
"""
Created on Wed May  1 17:34:53 2024
@author: DELL
"""

import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
#from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
#from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow as tf
#norm for linalg(linear alegbra)can measure the Euclidean distance between user preference vectors or item feature vectors. By comparing these distances,
# the system can recommend fashion items that are most similar items to that uploaded image
from numpy.linalg import norm
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications import ResNet50, MobileNetV2 
from sklearn.neighbors import NearestNeighbors
from streamlit_option_menu import option_menu

with st.sidebar:
    selected = option_menu("Recommendation System", ["Fashion","Furniture"], 
                           default_index=0,
                           icons=['Fashion','Furniture'])

if selected == 'Fashion':
    st.title('Fashion Recommender System')
    st.markdown("""
    ### AI Project by :
    - Memoona Basharat(21-CS-97)
    - Areeba Nazim (21-CS-79)
    - Sara Ahmed (21-CS-01)
    ### Submitted to
    - Sir Javed Iqbal
    """
    )
    features_list = np.array(pickle.load(open(r'C:\Users\DELL\OneDrive\Documents\AI_SEM_PROJECT\features_list_res.pkl', 'rb')))
    filename_list = pickle.load(open(r'C:\Users\DELL\OneDrive\Documents\AI_SEM_PROJECT\image_filenames_res.pkl', 'rb'))
    #we remove the directory path 'images _fashion_furniture\\' from each filename in filename_list,
    # leaving only the base filenames as it was giving error because of the space in images folder name
    filename_list = [filename.replace('images _fashion_furniture\\', '') for filename in filename_list]

    # Fashion
    model = ResNet50(weights="imagenet",
                        include_top=False,
                        input_shape=(224, 224, 3))

    model.trainable = False
    model = tf.keras.Sequential([
        model,
        GlobalMaxPooling2D()
    ])

    def save_uploaded_file(uploaded_file):
        try:
            with open(os.path.join(r'C:\Users\DELL\OneDrive\Documents\AI_SEM_PROJECT\uploads', uploaded_file.name), 'wb') as f:
                f.write(uploaded_file.getbuffer())
            return 1
        except:
            return 0

    def feature_extraction(img_path, model):
        img = image.load_img(img_path, target_size=(224, 224, 3))
        img_array = image.img_to_array(img)
        #idhar we added extra dimension here  to the array using np.expand_dims(img_array, axis=0) is
        #necessary because dl models typically expect input data in batches, even if it's a single image like the one we will uplaod 
        expanded_img_array = np.expand_dims(img_array, axis=0)
        # next the function (preprocess_input )normalizes the image data to the format expected by the pre-trained model i.e resnet
        preprocessed_img = preprocess_input(expanded_img_array)
        result = model.predict(preprocessed_img).flatten()
        normalized_result = result / norm(result)
        return normalized_result

    def recommend(features, features_list):
        neighbors = NearestNeighbors(n_neighbors=7, algorithm='brute', metric='euclidean')
        neighbors.fit(features_list)
        distances, indices = neighbors.kneighbors([features])
        return indices

    # File upload -> save
    uploaded_file = st.file_uploader("Choose an image")
    if uploaded_file is not None:
        if save_uploaded_file(uploaded_file):
            st.markdown('File uploaded Successfully')
            # Display the file
            display_image = Image.open(uploaded_file)
            resized_display_image = display_image.resize((100, 100))  # Resizing
            st.image(resized_display_image)
            features = feature_extraction(os.path.join(r"C:\Users\DELL\OneDrive\Documents\AI_SEM_PROJECT\uploads", uploaded_file.name), model)
            st.text(features)
            indices = recommend(features, features_list)
            st.markdown('Recommendations according to your uploaded image')
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            for i in range(6):
                recommended_image = Image.open(os.path.join(r'C:\Users\DELL\OneDrive\Documents\AI_SEM_PROJECT\images _fashion_furniture', filename_list[indices[0][i]]))
                resized_recommended_image = recommended_image.resize((100, 100))  # Resize the recommended images
                with locals()[f"col{i+1}"]:
                    st.image(resized_recommended_image)
        else:
         st.header("Some error occurred in file upload")


if selected == 'Furniture':   
    st.title('Furniture Recommender System')
    st.markdown("""
    ### AI Project by :
    - Memoona Basharat(21-CS-97)
    - Areeba Nazim (21-CS-79)
    - Sara Ahmed (21-CS-01)
    ### Submitted to
    - Sir Javed Iqbal
    """
    )
        
    features_list = np.array(pickle.load(open(r'C:\Users\DELL\OneDrive\Documents\AI_SEM_PROJECT\features_list_res.pkl', 'rb')))
    filename_list = pickle.load(open(r'C:\Users\DELL\OneDrive\Documents\AI_SEM_PROJECT\image_filenames_res.pkl', 'rb'))
    filename_list = [filename.replace('images _fashion_furniture\\', '') for filename in filename_list]

    # Furniture
    model = ResNet50(weights="imagenet",
                        include_top=False,
                        input_shape=(224, 224, 3))

    model.trainable = False
    model = tf.keras.Sequential([
        model,
        GlobalMaxPooling2D()
    ])

    def save_uploaded_file(uploaded_file):
        try:
            with open(os.path.join(r'C:\Users\DELL\OneDrive\Documents\AI_SEM_PROJECT\uploads', uploaded_file.name), 'wb') as f:
                f.write(uploaded_file.getbuffer())
            return 1
        except:
            return 0

    def feature_extraction(img_path, model):
        img = image.load_img(img_path, target_size=(224, 224, 3))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        result = model.predict(preprocessed_img).flatten()
        normalized_result = result / norm(result)
        return normalized_result

    def recommend(features, features_list):
        neighbors = NearestNeighbors(n_neighbors=7, algorithm='brute', metric='euclidean')
        neighbors.fit(features_list)
        distances, indices = neighbors.kneighbors([features])
        return indices

    # File upload -> save
    uploaded_file = st.file_uploader("Choose an image")
    if uploaded_file is not None:
        if save_uploaded_file(uploaded_file):
            st.markdown('File uploaded Successfully')
            # Display the file
            display_image = Image.open(uploaded_file)
            resized_display_image = display_image.resize((100, 100))  # Resizing
            st.image(resized_display_image)
            features = feature_extraction(os.path.join(r"C:\Users\DELL\OneDrive\Documents\AI_SEM_PROJECT\uploads", uploaded_file.name), model)
            st.text(features)
            indices = recommend(features, features_list)
            st.markdown('Recommendations according to your uploaded image')
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            for i in range(6):
                recommended_image = Image.open(os.path.join(r'C:\Users\DELL\OneDrive\Documents\AI_SEM_PROJECT\images _fashion_furniture', filename_list[indices[0][i]]))
                resized_recommended_image = recommended_image.resize((100, 100))  # Resize the recommended images
                with locals()[f"col{i+1}"]:
                    st.image(resized_recommended_image)
        else:
           st.header("Some error occurred in file upload")
