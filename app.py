import streamlit as st
import mysql.connector
import bcrypt
import os
from main_file import *

# Connect to MySQL database
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

# Create a new user
def create_user(username, email, password):
    conn = create_connection()
    if conn is None:
        st.error("Database connection failed.")
        return
    cursor = conn.cursor()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    try:
        cursor.execute("INSERT INTO user_info (name, email, password) VALUES (%s, %s, %s)", (username, email, hashed_password))
        conn.commit()
        st.success("User created successfully!")
    except mysql.connector.Error as err:
        st.error(f"Error creating user: {err}")
    finally:
        cursor.close()
        conn.close()

# Authenticate user
def authenticate_user(email, username, password):
    conn = create_connection()
    if conn is None:
        st.error("Database connection failed.")
        return None
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM user_info WHERE email = %s AND name = %s", (email, username))
        user = cursor.fetchone()
    except mysql.connector.Error as err:
        st.error(f"Error querying user: {err}")
        user = None
    finally:
        cursor.close()
        conn.close()
    if user and bcrypt.checkpw(password.encode('utf-8'), user[3].encode('utf-8')):
        return user
    return None

# Registration function
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
    