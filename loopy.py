## THis is the file that is designed to be used by the streamlit app. The differences between this file and the loopy_for_nb file are minor, but important

import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from collections import OrderedDict
from warnings import filterwarnings
import skimage
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
from skimage import color, data, filters, graph, measure, morphology,io
from skimage.filters import threshold_otsu,threshold_li
from skimage.measure import label, regionprops, regionprops_table
from skimage.color.colorconv import rgb2gray
import plotly
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.patches as mpatches
import io as bio


#an easy way for people to get images with less coding
def get_image(name: str):
  img = cv2.imread(name)
  plt.imshow(img)
  return img


#these functions work to unslant a page by calculating and working with shear angles to process the images in a way that make them ready to analyze by a network

def deskew(img):
    thresh=img
    edges = cv2.Canny(thresh,50,200,apertureSize = 3)

    lines = cv2.HoughLines(edges,1,np.pi/1000, 55)
    try:
        d1 = OrderedDict()
        for i in range(len(lines)):
            for rho,theta in lines[i]:
                deg = np.rad2deg(theta)
#                print(deg)
                if deg in d1:
                    d1[deg] += 1
                else:
                    d1[deg] = 1

        t1 = OrderedDict(sorted(d1.items(), key=lambda x:x[1] , reverse=False))
        print(list(t1.keys())[0],'Angle' ,thresh.shape)
        non_zero_pixels = cv2.findNonZero(thresh)
        center, wh, theta = cv2.minAreaRect(non_zero_pixels)
        angle=list(t1.keys())[0]
        if angle>160:
            angle=180-angle
        if angle<160 and angle>20:
            angle=12
        root_mat = cv2.getRotationMatrix2D(center, angle, 1)
        rows, cols = img.shape
        rotated = cv2.warpAffine(img, root_mat, (cols, rows), flags=cv2.INTER_CUBIC)

    except:
        rotated=img
        pass
    return rotated

def unshear(img):
    gray = img
    thresh = img.copy()
    plt.imshow(thresh)
    plt.show()
    trans = thresh.transpose()

    arr=[]
    for i in range(thresh.shape[1]):
        arr.insert(0,trans[i].sum())

    arr=[]
    for i in range(thresh.shape[0]):
        arr.insert(0,thresh[i].sum())

    y = thresh.shape[0]-1-np.nonzero(arr)[0][0]
    y_top = thresh.shape[0]-1-np.nonzero(arr)[0][-1]

    trans1 = thresh.transpose()
    sum1=[]
    for i in range(trans1.shape[0]):
        sum1.insert(i,trans1[i].sum())

    height = y - y_top
    max_value = 255*height
    prev_num = len([i for i in sum1 if i>=(0.6*max_value)])
    final_ang = 0

    for ang in range(-25,25,3):
        thresh = gray.copy()
        print('Ang',ang)
        if ang>0:
            #print(ang)
            for i in range(y):
                temp = thresh[i]
                move = int((y-i)*(math.tan(math.radians(ang))))
                if move >= temp.size:
                    move = temp.size
                thresh[i][:temp.size-move]=temp[move:]
                thresh[i][temp.size-move:] = [0 for m in range(move)]
        else:
            #print(ang)
            for i in range(y):
                temp = thresh[i]
                move = int((y-i)*(math.tan(math.radians(-ang))))
                if move >= temp.size:
                    move = temp.size
                thresh[i][move:]=temp[:temp.size-move]
                thresh[i][:move]=[0 for m in range(move)]
        trans1 = thresh.transpose()
        sum1=[]
        for i in range(trans1.shape[0]):
            sum1.insert(i,trans1[i].sum())
        #print(sum1)
        num = len([i for i in sum1 if i>=(0.60*max_value)])
        #print(num, prev_num)
        if(num>=prev_num):
            prev_num=num
            final_ang = ang

    thresh= gray.copy()
    if final_ang>0:
        for i in range(y):
            temp = thresh[i]
            move = int((y-i)*(math.tan(math.radians(final_ang))))
            if move >= temp.size:
                move = temp.size
            thresh[i][:temp.size-move]=temp[move:]
            thresh[i][temp.size-move:] = [0 for m in range(move)]
    else:
        for i in range(y):
            temp = thresh[i]
            move = int((y-i)*(math.tan(math.radians(-final_ang))))
            if move >= temp.size:
                move = temp.size
            thresh[i][move:]=temp[:temp.size-move]
            thresh[i][:move]=[0 for m in range(move)]

    plt.imshow(thresh, cmap='gray_r')
    return thresh


#this is our basic image processing function, taking in an image and customizable thresholding values in order to be precise in what text it keeps for a page

def image_processor(img , t0 = None ,t1 = None ,t2 = None,t3 = None,t4 = None):
  retina = img[:,:,:3]
  retina = color.rgb2gray(retina)
  if t0 is None and t1 is None and t2 is None and t3 is None and t4 is None:
      t0, t1, t2, t3, t4 = filters.threshold_multiotsu(retina, classes=6)
  mask = (retina < t0)
  vessels = filters.sato(retina, sigmas=range(1, 10)) * mask
  thresholded = filters.apply_hysteresis_threshold(vessels, 0.01, 0.03)
  labeled = ndi.label(thresholded)[0]


  thresh = threshold_otsu(retina)
  binary = thresh > retina
  binary = unshear(binary * 255)
  plt.axis('off')
  plt.imshow(binary, cmap = 'gray_r')
  plt.savefig('processed_image',dpi=300,bbox_inches='tight',pad_inches=0)
  return binary,t0,t1,t2,t3,t4

#this function is the one that is substantially different between the files, but they both perform the same task, which is filtering out the smaller pieces of text inside a filtered image
def connected_components(image, t=0.5, connectivity=2, min_area=30):
    # Convert the image to grayscale if needed
    if len(image.shape) == 2:
        grayscale_image = image
    elif len(image.shape) == 3 and image.shape[2] == 3:
        grayscale_image = color.rgb2gray(image)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        # Convert RGBA to RGB
        grayscale_image = color.rgba2rgb(image)
        grayscale_image = color.rgb2gray(grayscale_image)
    else:
        raise ValueError("Unsupported image format")

    # Ensure the image is in the range [0, 1]
    grayscale_image = grayscale_image.astype(np.float32) / 255.0

    # Mask the image according to threshold
    binary_mask = (grayscale_image > t).astype(np.uint8)

    # Perform connected component analysis
    labeled_image, count = measure.label(binary_mask, connectivity=connectivity, return_num=True)

    object_features = measure.regionprops(labeled_image)
    object_areas = [objf["area"] for objf in object_features]

    for object_id, objf in enumerate(object_features, start=1):
        if objf["area"] < min_area:
            labeled_image[labeled_image == objf["label"]] = 0

    # Use binary mask instead of labeled image for remove_small_objects
    object_mask = morphology.remove_small_objects(labeled_image, min_area)

    labeled_image, n = measure.label(object_mask, connectivity=2, return_num=True)

    # Convert the labeled image to a three-channel RGB image
    labeled_image_rgb = np.stack([labeled_image] * 3, axis=-1)

    binary_result = (labeled_image > 0).astype(np.uint8) * 255
    
    # Save the figure as an image
    buffer = bio.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    return binary_result
    #return buffer.read()

#gets the properties of each of the connected components
def get_prop(img):
    img4 = rgb2gray(img)
    label_img = label(img4)
    prop = regionprops_table(label_img, properties=('centroid',
                                                     'orientation',
                                                     'axis_major_length',
                                                     'axis_minor_length',
                                                    'perimeter'))
    return prop, label_img

#to highlight the connected components and their properties in a filtered image
def highlight_component(img):

    #img = cv2.imread(img)
    if len(img.shape) != 2:
       img = skimage.color.rgb2gray(img)
    threshold = filters.threshold_otsu(img)
    mask = img > threshold*0
    labels = measure.label(mask)
    
    fig = px.imshow(img, binary_string=True)
    fig.update_traces(hoverinfo='skip') # hover is only for label info
    
    props = measure.regionprops(labels, img)
    properties = ['area', 'eccentricity', 'perimeter', 'intensity_mean']
    
    
    # For each label, add a filled scatter trace for its contour,
    # and display the properties of the label in the hover of this trace.
    for index in range(1, labels.max()):
        label_i = props[index].label
        contour = measure.find_contours(labels == label_i, 0.5)[0]
        y, x = contour.T
        comp = x, y
        hoverinfo = ''
        for prop_name in properties:
            hoverinfo += f'<b>{prop_name}: {getattr(props[index], prop_name):.2f}</b><br>'
        fig.add_trace(go.Scatter(
            x=x, y=y, name=label_i,
            mode='lines', fill='toself', showlegend=False,
            hovertemplate=hoverinfo, hoveron='points+fills'))
    
    plotly.io.show(fig)

#put an image that you have filtered and selected the connected components
def display_components(img):

  img = cv2.imread(img)
  threshold = filters.threshold_otsu(img)
  mask = img > threshold*0
  labels = measure.label(mask)

  fig = px.imshow(img, binary_string=True)
  fig.update_traces(hoverinfo='skip') # hover is only for label info

  props = measure.regionprops(labels, img)
  properties = ['area', 'eccentricity', 'perimeter', 'intensity_mean']
  for i in range(len(props)):
      #for each object, print out its centroid and print the object itself
      print("Coordinates of the object number {} is {}".format(i+1, np.round(props[i]['centroid'],2)))
      mask = np.zeros(shape=props[i]['image'].shape)
      mask[props[i]['image']]=1
      plt.figure(figsize=(2,2))
      plt.imshow(mask, cmap=plt.cm.gray)
      plt.show()

#use on an image that you have already processed to display boxes around the individual connected components
def visualize_bounding_box(img):

  #img = cv2.imread(img)
  if len(img.shape) != 2:
       img = skimage.color.rgb2gray(img)
  threshold = filters.threshold_otsu(img)
  mask = img > threshold*0
  labels = measure.label(mask)

  props = measure.regionprops(labels, img)
  properties = ['area', 'eccentricity', 'perimeter', 'intensity_mean']
  fig, ax = plt.subplots(figsize=(10, 6))
  for i in range(len(props)):
      #Get the coordinates of the big objects:
      minr, minc, maxr, maxc = props[i].bbox
      #Apply a red rectangle enclosing each object of interest
      rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                            fill=False, edgecolor='red', linewidth=1)
      ax.add_patch(rect)
  plt.imshow(img)
