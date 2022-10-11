import streamlit as st
from PIL import Image

def open_img(file_path) :
    
    img = Image.open(file_path)
    st.image(img)

if __name__ == "__main__" :
    open_img(file_path)
