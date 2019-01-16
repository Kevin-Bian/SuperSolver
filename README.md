# Super Solver (Reformatting Code)
###### Solves sudoku given image file
Built with OpenCV, Numpy, and Scikit-Learn on **Python 3.6**

------------



<p align="center">**Here are some photos of the program**</p>

</p>
<p align="center">
  <img src="https://i.imgur.com/mqD4yd3.png" align="center" height="70%" width="70%" >
  <p align="center">
  <img src="https://i.imgur.com/q51bXBp.png" align="center" height="70%" width="70%" >

  <p align="center">
  <img src="https://i.imgur.com/ntRA6ES.jpg" align="center" height="70%" width="70%" >
  <p align="center">
  <img src="https://i.imgur.com/6i8COw7.png" align="center" height="70%" width="70%" >
  
  
------------
### How it is done:
1. Used image processing techniques (Gaussian Blur, dilation with kernel, etc.) and OpenCV to prepare the image for processing.
2. Isolate the sudoku grid from the rest of the image by finding and sorting contours using Numpy.
3. Warped and cropped image to obtain isolated grid.
4. Processed the grid and applied a Hough Line Transform to return a list of rho and theta (bascially to get gridlines).
5. Obtained the bounding rectangles for each digit using contours again.
6. Extracted the digits and used a KNN classifier to identify them.
7. Represnted the digits in an array and appled a backtracking algorithm to solve the puzzle.
8. Put the solved numbers onto the original photo.
9. Do a victory dance!
------------

### Future Changes:
-  Implementing a better OCR (perhaps using a CNN).
- Custom datasets to avoid bias.
- Implementing a custom KNN classifier. 
- Reformatting README.md


------------



### Links to MNIST data:
[Found here!](https://drive.google.com/open?id=18lcvdxFtZICvVe8FTfT_u0Jyme_snMTL "Found here!")
