Horizon Line Detection with OpenCV

This C++ program uses OpenCV to detect prominent straight lines in an image, filter them based on length and slope, and fit a smooth curve through the remaining line endpoints. 
The pipeline applies Canny edge detection, probabilistic Hough transform, and a custom polynomial regression routine to estimate a “horizon” or other dominant guide line in the scene. 
Intermediate results (edges, detected lines, filtered lines) are displayed and saved for debugging and analysis. Ideal as a starting point for computer vision tasks like lane detection, horizon tracking, or scene geometry estimation.
