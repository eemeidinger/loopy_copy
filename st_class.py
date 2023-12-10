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

import loopy

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

       st.image('../first_image/workflow.png', caption = 'The image processor works with many different threshold values to be precise in which text it wants to keep')

       self.uploaded_file = st.file_uploader("Choose a png file", accept_multiple_files=False)
       self.bytes_data = self.uploaded_file.read()
       st.write("filename:", self.uploaded_file.name)
       st.write(self.bytes_data)

       self.uploaded = cv2.imread(self.uploaded_file)

       self.option = st.selectbox(
           "Please select your function",
           ("Default Filtering", "Custom Filtering","Noise Removal", "Highlighting","Boxing"),
           index=None,
           placeholder="Options",
       )

       st.write('You selected:', self.option)

   def run(self):
       if self.option == "Default Filtering":
           img, t0, t1, t2, t3, t4 = image_processor(self.uploaded, name=str(self.uploaded_file.name), t0=0.205, t1=0.3465, t2=0.4657, t3=0.5472, t4=0.5974)

           st.download_button(
               label="Download your image!",
               data=img,
               file_name='filtered' + str(self.uploaded_file.name)
           )

       if self.option == "Custom Filtering":
           img, t0, t1, t2, t3, t4 = image_processor(self.uploaded, name=str(self.uploaded_file.name))

           st.download_button(
               label="Download your image!",
               data=img,
               file_name='filtered' + str(self.uploaded_file.name)
           )

       if self.option == "Noise Removal":
           img = connected_components(self.uploaded)

           st.download_button(
               label="Download your image!",
               data=img,
               file_name= 'filtered'+str(self.uploaded_file.name)
           )

       if self.option == "Highlighting":
           display_components(self.uploaded)

       if self.option == "Boxing":
           img = visualize_bounding_box(self.uploaded)

           st.download_button(
               label="Download your image!",
               data=img,
               file_name= 'boxed'+str(self.uploaded_file.name)
           )

if __name__ == "__main__":
   app = ImageProcessorApp()
   app.run()
