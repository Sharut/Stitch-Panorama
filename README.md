# Stitch-Panorama

To generate a panoramic image by stitching together multiple images of a scene.

Example:
![alt text](https://github.com/Sharut/Stitch-Panorama/blob/master/AssignmentImages/fig1.png?raw=true)
![alt text](https://github.com/Sharut/Stitch-Panorama/blob/master/AssignmentImages/fig2.png?raw=true)



# Extracting Feature Points
You may use any descriptor of your choice available in OpenCV.


# Homography Estimation
Compute the pair-wise image transformation matrix by applying RANSAC on the set of extracted feature points.
Develop some notion for scoring mechanism to choose which transformation pairs to consider for the panorama construction.

# Stitch and Blend images
With the obtained transformations, first, estimate the size of the overall picture that would be formed upon stitching.
For estimating the pixel values in regions where multiple images are superimposed, apply a blending technique of your choice. For a start, check out Alpha blending, Laplacian Pyramid Blending, 2-band blending, Graph cut blending.


# Approach and Results

# 1. Feature detector:
ORB feature detector is used to find the Key-points. A total of 20,000 key points were detected for the purpose of matching
  
# 2. Feature matching
Brute Force Matcher (BFMatcher) along with Best KNN match was used. Top 2 KNN matches were considered.
Ratio Test was also applied to further choose only valid key points wherein For each pair of features (f1, f2), if the distance between f1 and f2 is within a certain ratio, we keep it, otherwise, we throw it away.

# 3. Homography matrix for every two distinct matrices was found out.
Using this we created a sequence of the images in the order of panorama stitching from left to right.

# 4. Panorama Creation
After the last step we have our image order and the corresponding Homography matrices.

A. Naive Idea of Panorama construction:
• One simple approach could be to take the leftmost image as the base and keep moving right say I1, I1, I3....
• The next image (I2) is stitched with the base image (I1) using its pairwise Homography matrix.
• For the image I3, we have H23 i.e. the homography between I2 and I3. We multiply H12 and H23 to get H13 i.e. homography of Image 3 i.e. I3 w.r.t. I1(base image)
• This method would lead to error propagation and distorted panoramas.

B. Smart Idea of Panorama construction:
• One approach is to take the centre image as the base and stitch the left images I−1, I−2, I−3 . . . and the right images I1, I2, I3 . . . w.r.t to the centre image i.e. I0.
• Error is heavily reduced and better results are obtained.

# 5. Image Blending using Laplacian Blending
 Image blending and stiching is done through Laplace blending.
 
![alt text](https://github.com/Sharut/Stitch-Panorama/blob/master/AssignmentImages/fig3.png?raw=true)
![alt text](https://github.com/Sharut/Stitch-Panorama/blob/master/AssignmentImages/fig4.png?raw=true)
![alt text](https://github.com/Sharut/Stitch-Panorama/blob/master/AssignmentImages/fig5.png?raw=true)
![alt text](https://github.com/Sharut/Stitch-Panorama/blob/master/AssignmentImages/fig6.png?raw=true)
![alt text](https://github.com/Sharut/Stitch-Panorama/blob/master/AssignmentImages/fig7.png?raw=true)
![alt text](https://github.com/Sharut/Stitch-Panorama/blob/master/AssignmentImages/fig8.png?raw=true)
![alt text](https://github.com/Sharut/Stitch-Panorama/blob/master/AssignmentImages/fig9.png?raw=true)
![alt text](https://github.com/Sharut/Stitch-Panorama/blob/master/AssignmentImages/fig10.png?raw=true)
![alt text](https://github.com/Sharut/Stitch-Panorama/blob/master/AssignmentImages/fig11.png?raw=true)
![alt text](https://github.com/Sharut/Stitch-Panorama/blob/master/AssignmentImages/fig12.png?raw=true)
![alt text](https://github.com/Sharut/Stitch-Panorama/blob/master/AssignmentImages/fig13.png?raw=true)
![alt text](https://github.com/Sharut/Stitch-Panorama/blob/master/AssignmentImages/fig14.png?raw=true)
![alt text](https://github.com/Sharut/Stitch-Panorama/blob/master/AssignmentImages/fig15.png?raw=true)


