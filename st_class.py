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

import io as bio

from loopy import get_image, unshear, image_processor, connected_components, visualize_component, visualize_bounding_box, display_components


class ImageProcessorApp:
    def __init__(self):
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

        # st.image('workflow.png', caption='The image processor works with many different threshold values to be precise in which text it wants to keep')
        self.uploaded = None  # Initialize self.uploaded to None
        self.uploaded_file = None
        self.option = None

    def image(self):
        self.uploaded_file = st.file_uploader("Choose a png file")
        if self.uploaded_file is not None:
            # Read the file data into a BytesIO object
            bytes_io = bio.BytesIO(self.uploaded_file.getvalue())

            # Read the image data from the BytesIO object
            self.uploaded = cv2.imdecode(np.frombuffer(bytes_io.read(), np.uint8), -1)

        self.option = st.selectbox(
            "Please select your function",
            ("Default Filtering", "Custom Filtering", "Noise Removal", "Highlighting", "Boxing"),
            index=None,
            placeholder="Options",
        )

        st.write('You selected:', self.option)

        return self.uploaded, self.uploaded_file, self.option
        
    def run(self):
        if self.uploaded is not None:  # Check if self.uploaded is defined
            if self.option == "Default Filtering":
                img, t0, t1, t2, t3, t4 = image_processor(self.uploaded , t0=0.205, t1=0.3465,
                                                           t2=0.4657, t3=0.5472, t4=0.5974)

                img_bytes = cv2.imencode(".png", img)[1].tobytes()
                st.download_button(
                    label="Download your image!",
                    data=img_bytes,
                    file_name='filtered ' + self.uploaded_file.name
                )

            if self.option == "Custom Filtering":
                img, t0, t1, t2, t3, t4 = image_processor(self.uploaded)

                img_bytes = cv2.imencode(".png", img)[1].tobytes()

                st.download_button(
                    label="Download your image!",
                    data=img_bytes,
                    file_name='filtered ' + self.uploaded_file.name
                )

            if self.option == "Noise Removal":
                self.uploaded = cv2.cvtColor(self.uploaded, cv2.COLOR_BGR2GRAY)
                labeled_image = connected_components(self.uploaded, t=0.5)

                labeled_image = labeled_image.astype(np.uint8)

                binary_result = (labeled_image > 0) * 255
                
    
                st.download_button(
                    label="Download your image!",
                    data=binary_result,
                    file_name='filtered ' + self.uploaded_file.name
                )

            if self.option == "Highlighting":
                display_components(self.uploaded)

            if self.option == "Boxing":
                img = visualize_bounding_box(self.uploaded)
                img_bytes = cv2.imencode(".png", img)[1].tobytes()

                st.download_button(
                    label="Download your image!",
                    data=img_bytes,
                    file_name='boxed ' + self.uploaded_file.name
                )
        else:
            st.warning("Please upload an image before selecting a function.")
