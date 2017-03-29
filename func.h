#ifndef FUNC_H_
#define FUNC_H_

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

vector<Point2f> matToPoints(Mat img);
vector<Point2f> rectToPoints(Rect rect);
vector<Point2i> pointsFToI(vector<Point2f> points);
Point pointsCenter(vector<Point2f> points);
Size pointsMax(vector<Point2f> points);

#endif
