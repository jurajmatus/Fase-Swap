#include <opencv2/opencv.hpp>
using namespace cv;

typedef struct {
	Rect rect;
	Rect face;
	std::vector<Point> faceRectPoints;
	std::vector<Point> faceHullPoints;
	Rect eye1;
	Rect eye2;
} Head;
