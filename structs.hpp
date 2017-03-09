#include <opencv2/opencv.hpp>
using namespace cv;

typedef struct {
	Rect face;
	Rect eye1;
	Rect eye2;
} Head;
