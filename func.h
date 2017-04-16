#ifndef FUNC_H_
#define FUNC_H_

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

vector<Point2f> matToPoints(Mat img);
vector<Point> rectToPoints(Rect rect);
vector<Point2i> pointsFToI(vector<Point2f> points);
vector<Point2i> pointsToI(vector<Point> points);
vector<Point2f> pointsToF(vector<Point> points);
vector<Point2f> pointsIToF(vector<Point> points);
Point pointsCenter(vector<Point> points);
Size pointsMax(vector<Point> points);
Rect hullToRect(vector<Point> hull);

#endif
