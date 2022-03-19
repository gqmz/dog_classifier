"""
contains all front-end code for streamlit app to run
"""

from distutils.command.upload import upload
import os

#plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='darkgrid')
sns.set()
from PIL import Image

#utitlies
from utils.predictions import predictor

import streamlit as st

def save_uploaded_file(uploaded_file):
    """
    Save uploaded image file to static/uploaded_images
    """
    try:
        with open(os.path.join("static/uploaded_images", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def page_setup():
    """
    Basic setup for app page
    """
    # avoid StreamlitAPIException: set_page_config() can only be called once per app, and must be called as the first Streamlit command in your script.
    st.set_page_config(page_title="Dog Breed Classifier")
    #app name
    st.title("Dog Breed Classifier")
    

if __name__ == '__main__':
    #set up title, etc
    page_setup()

    # display file uploader widget, return UploadedFile object
    uploaded_file = st.file_uploader(label="Upload your image here",
                                    type=['jpg'], #allowed extensions
                                    accept_multiple_files=False,
                                    key='image_uploader')

    if uploaded_file:
        #check if able to save uploaded file, if yes continue
        if save_uploaded_file(uploaded_file):

            #display image
            display_image = Image.open(uploaded_file)
            st.image(display_image,
                    caption="Here is a very good boy")

            #make prediction
            prediction = predictor(os.path.join('static/uploaded_images', uploaded_file.name))
            #remove file after making prediction
            os.remove('static/uploaded_images/' + uploaded_file.name)

            #draw graphs
            st.text("Predictions:")
            fig, ax = plt.subplots()
            ax = sns.barplot(y='name', x='probability', data=prediction,
                            order=prediction.sort_values('probability', ascending=False).name)
            ax.set(xlabel="Probability", ylabel="Predicted Breed")
            st.pyplot(fig)
        else:
            #display error in uploading condition on app
            st.text("File upload failed!")
