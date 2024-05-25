import streamlit as st
import mysql.connector
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import random  # Importing the random module

# Function to create a connection to the MySQL database
def create_connection():
    try:
        conn = mysql.connector.connect(
            host=os.getenv("DB_HOST", "localhost"),
            user=os.getenv("DB_USER", "root"),
            password=os.getenv("DB_PASSWORD", "Kc830320@"),
            database=os.getenv("DB_NAME", "users")
        )
        if conn.is_connected():
            print("Connection successful.")
        return conn
    except mysql.connector.Error as err:
        print(f"Error connecting to database: {err}")
        return None

# Function to save an uploaded file to the server and extract its features
def save_uploaded_file(uploaded_file, user_id, model):
    try:
        os.makedirs('uploads', exist_ok=True)
        file_path = os.path.join('uploads', uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        features = feature_extraction(file_path, model)
        conn = create_connection()
        if conn is None:
            st.error("Database connection failed.")
            return False
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM user_images WHERE user_id = %s", (user_id,))
        count = cursor.fetchone()[0]
        if count >= 2:
            cursor.execute("SELECT MIN(id) FROM user_images WHERE user_id = %s", (user_id,))
            oldest_id = cursor.fetchone()[0]
            cursor.execute("DELETE FROM user_images WHERE id = %s", (oldest_id,))

        cursor.execute("INSERT INTO user_images (user_id, image_path, features) VALUES (%s, %s, %s)", 
                       (user_id, uploaded_file.name, pickle.dumps(features)))
        conn.commit()
        cursor.close()
        conn.close()

        return True
    except Exception as e:
        st.error(f"Error saving uploaded file: {e}")
        return False

# Function to extract features from an image using a pre-trained model
def feature_extraction(img_path, model):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        result = model.predict(preprocessed_img).flatten()
        normalized_result = result / norm(result)
        return normalized_result
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

# Function to recommend similar images based on features
def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# Main function for the fashion recommender system
def fashion_recommender(show_history=False):
    st.markdown("<h1 style='text-align: center; color: orange; white-space: nowrap;'>👗 Fashion Recommender System 👠</h1>", unsafe_allow_html=True)

    try:
        feature_list = np.array(pickle.load(open('feature_embedding.pkl', 'rb')))
        filenames = pickle.load(open('filenames.pkl', 'rb'))
    except Exception as e:
        st.error(f"Error loading precomputed features: {e}")
        return

    model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model.trainable = False
    model = tf.keras.Sequential([model, GlobalMaxPooling2D()])

    uploaded_file = st.file_uploader("Choose an image")
    if uploaded_file is not None:
        if save_uploaded_file(uploaded_file, st.session_state.user_id, model):
            display_image = Image.open(uploaded_file)
            st.image(display_image, caption='Uploaded Image', use_column_width=True)

            features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)
            if features is not None:
                indices_latest = recommend(features, feature_list)

                if show_history:
                    user_id = st.session_state.user_id
                    conn = create_connection()
                    if conn is None:
                        st.error("Database connection failed.")
                        return
                    cursor = conn.cursor()
                    cursor.execute("SELECT features, image_path FROM user_images WHERE user_id = %s ORDER BY id DESC LIMIT 2", (user_id,))
                    user_data = cursor.fetchall()
                    cursor.close()
                    conn.close()

                    if len(user_data) == 2:
                        user_features_latest = pickle.loads(user_data[0][0])
                        user_features_second_latest = pickle.loads(user_data[1][0])

                        indices_second_latest = recommend(user_features_second_latest, feature_list)

                        st.subheader("Recommended Products:")
                        cols = st.columns(5)
                        for i, col in enumerate(cols):
                            with col:
                                if i < 3:
                                    recommended_image_path = filenames[indices_latest[0][i]]
                                    price = random.randint(500, 2000)  # Random price between 500 and 2000
                                    st.image(recommended_image_path, use_column_width=True)
                                    st.write(f"Price: ₹{price}")  # Display price in rupees
                                else:
                                    recommended_image_path = filenames[indices_second_latest[0][i - 3]]
                                    price = random.randint(500, 2000)  # Random price between 500 and 2000
                                    st.image(recommended_image_path, use_column_width=True)
                                    st.write(f"Price: ₹{price}")  # Display price in rupees
                    else:
                        st.info("Insufficient search history to provide recommendations from both searches.")
                        st.subheader("Recommended Products:")
                        cols = st.columns(5)
                        for i, col in enumerate(cols):
                            with col:
                                recommended_image_path = filenames[indices_latest[0][i]]
                                price = random.randint(500, 2000)  # Random price between 500 and 2000
                                st.image(recommended_image_path, use_column_width=True)
                                st.write(f"Price: ₹{price}")  # Display price in rupees
                else:
                    st.subheader("Recommended Products:")
                    cols = st.columns(5)
                    for i, col in enumerate(cols):
                        with col:
                            if i < len(indices_latest[0]):
                                recommended_image_path = filenames[indices_latest[0][i]]
                                price = random.randint(500, 2000)  # Random price between 500 and 2000
                                st.image(recommended_image_path, use_column_width=True)
                                st.write(f"Price: ₹{price}")  # Display price in rupees
        else:
            st.header("Some error occurred in file upload")

    elif show_history:
        user_id = st.session_state.user_id
        conn = create_connection()
        if conn is None:
            st.error("Database connection failed.")
            return
        cursor = conn.cursor()
        cursor.execute("SELECT features, image_path FROM user_images WHERE user_id = %s ORDER BY id DESC LIMIT 2", (user_id,))
        user_data = cursor.fetchall()
        cursor.close()
        conn.close()

        if len(user_data) == 2:
            user_features_latest = pickle.loads(user_data[0][0])
            user_features_second_latest = pickle.loads(user_data[1][0])

            indices_latest = recommend(user_features_latest, feature_list)
            indices_second_latest = recommend(user_features_second_latest, feature_list)

            st.subheader("Based on your recent activity:")
            cols = st.columns(6)
            for i, col in enumerate(cols):
                with col:
                    if i < 3:
                        recommended_image_path = filenames[indices_latest[0][i]]
                        price = random.randint(500, 2000)  # Random price between 500 and 2000
                        st.image(recommended_image_path, use_column_width=True)
                        st.write(f"Price: ₹{price}")  # Display price in rupees
                    else:
                        recommended_image_path = filenames[indices_second_latest[0][i - 3]]
                        price = random.randint(500, 2000)  # Random price between 500 and 2000
                        st.image(recommended_image_path, use_column_width=True)
                        st.write(f"Price: ₹{price}")  # Display price in rupees
        elif len(user_data) == 1:
            user_features_latest = pickle.loads(user_data[0][0])
            indices_latest = recommend(user_features_latest, feature_list)
            st.subheader("Based on your recent activity:")
            cols = st.columns(5)
            for i, col in enumerate(cols):
                with col:
                    recommended_image_path = filenames[indices_latest[0][i]]
                    price = random.randint(500, 2000)  # Random price between 500 and 2000
                    st.image(recommended_image_path, use_column_width=True)
                    st.write(f"Price: ₹{price}")  # Display price in rupees

# Main function
def main():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        Registration()
    else:
        if "first_login" not in st.session_state:
            st.session_state.first_login = True

        if st.session_state.first_login:
            fashion_recommender(show_history=True)
            st.session_state.first_login = False
        else:
            fashion_recommender(show_history=False)

