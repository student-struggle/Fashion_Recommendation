import streamlit as st
import numpy as np
import pickle
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import os
from database import create_user, authenticate_user, save_uploaded_file, get_user_images

def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

def fashion_recommender(show_history=False):
    st.title('Fashion Recommender System')

    with open('filenames.pkl', 'rb') as f:
        github_urls = pickle.load(f)
    with open('feature_embedding.pkl', 'rb') as f:
        feature_list = np.array(pickle.load(f))

    model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model.trainable = False
    model = tf.keras.Sequential([model, GlobalMaxPooling2D()])

    uploaded_file = st.file_uploader("Choose an image")
    if uploaded_file is not None:
        if save_uploaded_file(uploaded_file, st.session_state.user_id, model):
            display_image = Image.open(uploaded_file)
            st.image(display_image, caption='Uploaded Image', use_column_width=True)

            features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)
            indices_latest = recommend(features, feature_list)

            if show_history:
                user_data = get_user_images(st.session_state.user_id)
                if len(user_data) == 2:
                    user_features_latest = pickle.loads(user_data[0][0])
                    user_features_second_latest = pickle.loads(user_data[1][0])
                    indices_second_latest = recommend(user_features_latest, feature_list)
                    st.subheader("Recommended Products:")
                    cols = st.columns(5)
                    for i, col in enumerate(cols):
                        with col:
                            if i < 3:
                                recommended_image_path = github_urls[indices_latest[0][i]]
                            else:
                                recommended_image_path = github_urls[indices_second_latest[0][i - 3]]
                            st.image(recommended_image_path, use_column_width=True, caption=f"Recommendation {i+1}")
                else:
                    st.info("Insufficient search history to provide recommendations from both searches.")
                    st.subheader("Recommended Products:")
                    cols = st.columns(5)
                    for i, col in enumerate(cols):
                        with col:
                            recommended_image_path = github_urls[indices_latest[0][i]]
                            st.image(recommended_image_path, use_column_width=True, caption=f"Recommendation {i+1}")
            else:
                st.subheader("Recommended Products:")
                cols = st.columns(5)
                for i, col in enumerate(cols):
                    with col:
                        if i < len(indices_latest[0]):
                            recommended_image_path = github_urls[indices_latest[0][i]]
                            st.image(recommended_image_path, use_column_width=True, caption=f"Recommendation {i+1}")
        else:
            st.header("Some error occurred in file upload")
    elif show_history:
        user_data = get_user_images(st.session_state.user_id)
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
                        recommended_image_path = github_urls[indices_latest[0][i]]
                    else:
                        recommended_image_path = github_urls[indices_second_latest[0][i - 3]]
                    st.image(recommended_image_path, use_column_width=True, caption=f"Recommendation {i+1}")
        elif len(user_data) == 1:
            user_features_latest = pickle.loads(user_data[0][0])
            indices_latest = recommend(user_features_latest, feature_list)
            st.subheader("Based on your recent activity:")
            cols = st.columns(5)
            for i, col in enumerate(cols):
                with col:
                    recommended_image_path = github_urls[indices_latest[0][i]]
                    st.image(recommended_image_path, use_column_width=True, caption=f"Recommendation {i+1}")

def Registration():
    st.title("Registration and Login")
    st.sidebar.title("Menu")
    menu = st.sidebar.selectbox("Choose an option", ["Login", "Register"])

    if menu == "Login":
        st.subheader("Login")
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type='password')
        if st.button("Login"):
            user = authenticate_user(email, username, password)
            if user:
                st.session_state.authenticated = True
                st.session_state.user_id = user[0]
                st.success("Login successful!")
            else:
                st.error("Invalid username or password.")
    elif menu == "Register":
        st.subheader("Register")
        new_username = st.text_input("Username")
        new_email = st.text_input("Email")
        new_password = st.text_input("Password", type='password')
        confirm_password = st.text_input("Confirm Password", type='password')
        if st.button("Register"):
            if new_password == confirm_password:
                create_user(new_username, new_email, new_password)
            else:
                st.error("Passwords do not match.")

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

if __name__ == "__main__":
    main()
