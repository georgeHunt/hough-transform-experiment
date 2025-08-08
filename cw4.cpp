#include <iostream>
#include <stdio.h>
#include <string>
#include <math.h>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
using namespace cv;
using namespace std;


int thresholdV = 70;

int thresholdValue=0;
int thresholdType=0;
int const maxBinVal=255;
int const maxValue=255;
int const maxType=10;
float deltaX;
float deltaY;
float lineLength;
Mat img, dst, cdst, cdstP, cdstPOriented, cdstPShort;
vector<Vec4i> lines;
const char* trackbarType = "Enter 0: Binary, 1: Inverted Binary, 2:Truncate, 3:To Zero, 4: To Zero Inverted";
const char* trackbarValue="Value";

float gradient;
int lengthThreshold;

static void Threshold(int, void*){
  threshold(cdst, dst, thresholdValue,maxBinVal,thresholdType);
  imshow("Thresholding",dst);
}

//Polynomial regression function
vector<double> fitPoly(vector<cv::Point> points, int n)
{
  //Number of points
  int nPoints = points.size();

  //vectors for all the points' xs and ys
  vector<float> xValues = vector<float>();
  vector<float> yValues = vector<float>();

  //Split the points into two vectors for x and y values
  for(int i = 0; i < nPoints; i++)
  {
    xValues.push_back(points[i].x);
    yValues.push_back(points[i].y);
  }

  //Augmented matrix
  double matrixSystem[n+1][n+2];
  for(int row = 0; row < n+1; row++)
  {
    for(int col = 0; col < n+1; col++)
    {
      matrixSystem[row][col] = 0;
      for(int i = 0; i < nPoints; i++)
        matrixSystem[row][col] += pow(xValues[i], row + col);
    }

    matrixSystem[row][n+1] = 0;
    for(int i = 0; i < nPoints; i++)
      matrixSystem[row][n+1] += pow(xValues[i], row) * yValues[i];

  }

  //Array that holds all the coefficients
  double coeffVec[n+2] = {};  // the "= {}" is needed in visual studio, but not in Linux

  //Gauss reduction
  for(int i = 0; i <= n-1; i++)
    for (int k=i+1; k <= n; k++)
    {
      double t=matrixSystem[k][i]/matrixSystem[i][i];

      for (int j=0;j<=n+1;j++)
        matrixSystem[k][j]=matrixSystem[k][j]-t*matrixSystem[i][j];

    }

  //Back-substitution
  for (int i=n;i>=0;i--)
  {
    coeffVec[i]=matrixSystem[i][n+1];
    for (int j=0;j<=n+1;j++)
      if (j!=i)
        coeffVec[i]=coeffVec[i]-matrixSystem[i][j]*coeffVec[j];

    coeffVec[i]=coeffVec[i]/matrixSystem[i][i];
  }

  //Construct the cv vector and return it
  vector<double> result = vector<double>();
  for(int i = 0; i < n+1; i++)
    result.push_back(coeffVec[i]);
  return result;
}

//Returns the point for the equation determined
//by a vector of coefficents, at a certain x location
cv::Point pointAtX(vector<double> coeff, double x)
{
  double y = 0;
  for(int i = 0; i < coeff.size(); i++)
  y += pow(x, i) * coeff[i];
  return cv::Point(x, y);
}


int main(int argc, char *argv[])
{
  if(argc!=2)
  {
    printf("Wrong arguments entered, aboritng program.\n");
    return -1;
  }
  img=cv::imread(argv[1]);
  if(img.empty())
  {
    printf("Image not found, aborting program.\n");
    return -1;
  }
  Canny(img, dst, 50, 150, 3);
  cvtColor(dst, cdst, COLOR_GRAY2BGR);
  cdstP = cdst.clone();
  vector<Vec4i> lines;
  HoughLinesP(dst, lines, 1, CV_PI/180, 50, 50, 15);
  int maxLength=0;
  for(size_t i = 0; i < lines.size(); i++ ){ //Find the length of the longest line for comparison
    deltaX = lines.at(i)[2] - lines.at(i)[0];
    deltaY = lines.at(i)[3] - lines.at(i)[1];
    lineLength = sqrt(pow(deltaX,2) + pow(deltaY,2));
    if (lineLength>maxLength){
      maxLength=lineLength;
    }
  }
  for( size_t i = 0; i < lines.size(); i++ ){
      Vec4i l = lines[i];
      line( cdstP, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, LINE_AA);
  }
  lengthThreshold=0.7*(maxLength);
  vector<Vec4i> linesCopy = lines;
  for(size_t i = 0; i < lines.size(); i++ ){
    deltaX = lines.at(i)[2] - lines.at(i)[0];
    deltaY = lines.at(i)[3] - lines.at(i)[1];
    lineLength = sqrt(pow(deltaX,2) + pow(deltaY,2)); //Length of line with pythag
    gradient=(fabs(deltaY/deltaX)); // fabs is just abs() for floats
    if (lineLength<lengthThreshold){
      for (int j=0; j<4;j++){
        linesCopy.at(i)[j]=0;
      }

    for( size_t i = 0; i < linesCopy.size(); i++ ){
        Vec4i l = linesCopy[i];
        line( cdstPShort, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, LINE_AA);
      }
    if (gradient >= 0.4){
      for (int j =0; j<4;j++){
        linesCopy.at(i)[j]=0;
      }
    for( size_t i = 0; i < linesCopy.size(); i++ ){
        Vec4i l = linesCopy[i];
        line( cdstPOriented, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, LINE_AA);
      }
    }
    }
  }
  for( size_t i = 0; i < linesCopy.size(); i++ ){
      Vec4i l = linesCopy[i];
      line( cdstP, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, LINE_AA);
  }
  vector<Point> points;
  vector<double> coeffVec;
  for(size_t i = 0; i < linesCopy.size(); i++){
    Vec4i aLine = linesCopy[i];
    Point a = Point(aLine[0],aLine[1]);
    Point b = Point(aLine[2],aLine[3]);
    points.push_back(a);
    points.push_back(b);
  }
  Point start;
  coeffVec=fitPoly(points, 2);
  for(int i = 0; i<img.size().width; i++){
    start=pointAtX(coeffVec,i);
    circle(img,start,1,Scalar(0,0,255));
  }


  imshow("Edges", cdst);
  imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP);
  imshow("Horizon Line Estimation",img);
  waitKey(0);
  imwrite("canny.bmp",cdst);
  imwrite("houghtransform.bmp", cdstP);
  imwrite("filtershort.bmp",cdstPShort);
  imwrite("filteroriented.bmp",cdstPOriented);
  destroyAllWindows();
  return 0;
}
