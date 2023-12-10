# loopy_copy

In this repository we will work with historical documents to be able to filter out unnecessary text, ink bleedthrough, or blotches on the pages, as well as accounting for potential curvature of pages. 

## Filtering Workflow
Below is an example workflow of the filtering algorithm used here. Inside, we use multiotsu thresholding to divide the pixel intensities into five distinct classes, represented by the colors in the words. The algorithm assigns almost no weight to the ink bleedthrough from the other side of the page, which is represented by its lack of color. After the algorithm decides which text it wants to take, it converts it to a two colored image, represented here by the black and white image.

<p align="center">
  <img src="https://github.com/eemeidinger/loopy_copy/blob/main/first_image/thresholding_pipeline.png" alt="final_image">
</p>


## Connected Components
After we do the image filtering, you can see that we still have a little bit of noise or extra text throughout the page, most clearly seen in the bottom left corner of the black and white image above. Noise like this could potentially harm the accuracy and efficiency of text recognition models, so we want to try and filter out as much unnecessary information as possible while still retaining all the relevant text information. As a rule of thumb, the algorithm will usually retain slightly more than is necessary as to not filter out any important text information. For this specific project, we want to keep as much data as possible. Below is the black and white image from above that has been filtered to remove a good amount of noise.


<p align="center">
  <img src="https://github.com/eemeidinger/loopy_copy/blob/main/first_image/filtered_image_for_github.png" alt="final_image">
</p>

Following this, we can access each of the connected components and store their individual properties for later analysis. Future instances of this project will involve a text recognition model that will be able to read each connected component

There will also be a colab notebook in this file that will display the connected components in different ways, which are highlighting, boxing them in, and displaying the outlines and coordinates of them.

The link below will take you to a streamlit app where you can filter your own images without the need to code yourself!
[Click here to access the filtering app!](https://loopycopy.streamlit.app/)
