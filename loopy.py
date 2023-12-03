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

def image_processer_2(img):
  retina = img[:,:,:3]
  retina = color.rgb2gray(retina)
  t0,t1,t2,t3,t4 = filters.threshold_multiotsu(retina, classes=6)

  mask = (retina < t0)
  vessels = filters.sato(retina, sigmas=range(1, 10)) * mask
  thresholded = filters.apply_hysteresis_threshold(vessels, 0.01, 0.03)
  labeled = ndi.label(thresholded)[0]


  thresh = threshold_otsu(retina)
  binary = thresh > retina
  binary = unshear(binary * 255)
  plt.axis('off')
  plt.imshow(binary)
  plt.savefig('processed_image',dpi=300,bbox_inches='tight',pad_inches=0)
  return binary

def connected_components(img, sigma=1.0, t=0.5, connectivity=2):
    # load the image
    image = img
    # convert the image to grayscale
    gray_image = skimage.color.rgb2gray(image)
    # mask the image according to threshold
    binary_mask = gray_image > t
    # perform connected component analysis
    labeled_image, count = skimage.measure.label(binary_mask,
                                                 connectivity=connectivity, return_num=True)
    return labeled_image, count, binary_mask

def get_prop(img3):
    img4 = rgb2gray(img3)
    label_img = label(img4)
    prop = regionprops_table(label_img, properties=('centroid',
                                                     'orientation',
                                                     'axis_major_length',
                                                     'axis_minor_length',
                                                    'perimeter'))
    return prop, label_img

#to visualize the connected components
def visualize_component(img4):
    threshold = filters.threshold_otsu(img4)
    mask = img4 > threshold*0
    labels = measure.label(mask)
    
    fig = px.imshow(img4, binary_string=True)
    fig.update_traces(hoverinfo='skip') # hover is only for label info
    
    props = measure.regionprops(labels, img4)
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

def get_components(img4):
  threshold = filters.threshold_otsu(img4)
  mask = img4 > threshold*0
  labels = measure.label(mask)

  fig = px.imshow(img4, binary_string=True)
  fig.update_traces(hoverinfo='skip') # hover is only for label info

  props = measure.regionprops(labels, img4)
  properties = ['area', 'eccentricity', 'perimeter', 'intensity_mean']
  for i in range(len(props)):
      #for each object, print out its centroid and print the object itself
      print("Coordinates of the object number {} is {}".format(i+1, np.round(props[i]['centroid'],2)))
      mask = np.zeros(shape=props[i]['image'].shape)
      mask[props[i]['image']]=1
      plt.figure(figsize=(2,2))
      plt.imshow(mask, cmap=plt.cm.gray)
      plt.show()

