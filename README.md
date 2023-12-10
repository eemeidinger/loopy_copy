# loopy_copy

In this repository we will work with historical documents to be able to filter out unnecessary text, ink bleedthrough, or blotches on the pages, as well as accounting for potential curvature of pages. 

Below is an example workflow of the filtering algorithm used here. Inside, we use multiotsu thresholding to divide the pixel intensities into five distinct classes, represented by the colors in the words. The algorithm assigns almost no weight to the ink bleedthrough from the other side of the page, which is represented by its lack of color. After the algorithm decides which text it wants to take, it converts it to a two colored image, represented here by the first of the black and white images.

![workflow](https://github.com/eemeidinger/loopy_copy/blob/main/first_image/pipeline.png)
![final_image]()



The goal is to conduct OCR analysis to help with reading and transcription of Spanish port documents. While this project is designed to be used on old Spanish documents, it is intended to be flexible enough to be trained on any form of writing.

I hope to give this to the history department by the end of the semester

[Click here to access the filtering app!](https://loopycopy.streamlit.app/)
