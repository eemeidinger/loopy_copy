import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from collections import OrderedDict
from warnings import filterwarnings
import skimage
from scipy import ndimage as ndi
from skimage import color, data, filters, graph, measure, morphology,io
from skimage.filters import threshold_otsu,threshold_li
from skimage.measure import label, regionprops, regionprops_table
from skimage.color.colorconv import rgb2gray
import plotly
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.patches as mpatches
import streamlit as st

from loopy import get_image, unshear, image_processer, connected_components, visualize_component, visualize_bounding_box, display_components

### Description of the app functions

st.write("""
# This software will help you be able to easily filter unnecessary text or blotches out of any image 
---
In this app, you will be able to use four different functions to perform different tasks

Default Filtering: This function will take a raw image and convert it to black and white using preset filtering thresholds

Custom Filtering: This function does the same as the above but calculates thresholding values unique to the image

Noise Removal: This function takes an already filtered image and removes smaller, unimportant instances of text

Highlighting: This function will take an already filtered image and highlight the different instances of text

Boxing: This function will take an already filtered image and create boxes around each distinct instance of text
""")

st.image('../first_image/workflow.png', caption = 'The image processor works with many different threshold values to be precise in which text it wants to keep')

### Choose a file to run the functions on!

uploaded_file = st.file_uploader("Choose a png file", accept_multiple_files=False)
bytes_data = uploaded_file.read()
st.write("filename:", uploaded_file.name)
st.write(bytes_data)

uploaded = cv2.imread(uploaded_file)

### Select the desired function you want to run

option = st.selectbox(
   "Please select your function",
   ("Default Filtering", "Custom Filtering","Noise Removal", "Highlighting","Boxing"),
   index=None,
   placeholder="Options",
)

st.write('You selected:', option)

if option == "Default Filtering":
    img, t0, t1, t2, t3, t4 = image_processor(uploaded, name=str(uploaded_file.name), t0=0.205, t1=0.3465, t2=0.4657, t3=0.5472, t4=0.5974)

    st.download_button(
        label="Download your image!",
        data=img,
        file_name='filtered' + str(uploaded_file.name)
    )

if option == "Custom Filtering":
    img, t0, t1, t2, t3, t4 = image_processor(uploaded, name=str(uploaded_file.name))

    st.download_button(
        label="Download your image!",
        data=img,
        file_name='filtered' + str(uploaded_file.name)
    )


        
if option == "Noise Removal":
    img = connected_components(uploaded)

    st.download_button(
        label="Download your image!",
        data=img,
        file_name= 'filtered'+str(uploaded_file.name)
    )

if option == "Highlighting":
    display_components(uploaded)


if option == "Boxing":
    img = visualize_bounding_box(uploaded)

    st.download_button(
        label="Download your image!",
        data=img,
        file_name= 'boxed'+str(uploaded_file.name)
    )
