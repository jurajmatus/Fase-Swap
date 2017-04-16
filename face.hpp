#ifndef FACE_HPP_
#define FACE_HPP_

#define DLIB_NO_GUI_SUPPORT
//#define NO_MAKEFILE

#include <opencv2/opencv.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/opencv/cv_image.h>
#include <iostream>

using namespace std;
using namespace cv;

typedef struct {
	Rect rect;
	vector<Point> points;
} Face;

Face findFace(Mat& _img);

#endif
