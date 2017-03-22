#include <opencv2/opencv.hpp>
using namespace cv;

typedef struct {
	Rect rect;
	Rect face;
	std::vector<Point> facePoints;
	Rect eye1;
	Rect eye2;
} Head;
