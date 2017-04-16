#include <opencv2/opencv.hpp>

#include "face.hpp"
#include "func.h"

using namespace cv;
using namespace std;

const int NUM_FRAMES_TO_REFRESH = 30;

Face camFace;
int camFaceAge = 0;

Mat swapFaceImg;
Face swapFace;

void findFace(Mat& img, Mat& gray) {
	camFace = findFace(img);
	camFaceAge = 0;
}

Mat oldGray;
bool refreshFace(Mat& img, Mat& gray) {
	gray.copyTo(oldGray);
	if (camFace.points.empty() || camFaceAge >= NUM_FRAMES_TO_REFRESH) {
		return false;
	}

	vector<uchar> status;
	Mat err;
	vector<Point2f> oldPoints = pointsToF(camFace.points);
	vector<Point2f> newPoints;

	TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
	calcOpticalFlowPyrLK(oldGray, gray, oldPoints, newPoints, status, err, Size(10, 10), 3, termcrit, 0, 0.001);

	if (newPoints.size() != oldPoints.size()) {
		return false;
	}
	camFace.points = pointsFToI(newPoints);

	camFaceAge++;
	return true;
}

void drawFace(Mat& img, Face& face) {
	for (auto &point : face.points) {
		circle(img, point, 2, Scalar(0, 255, 0), 3);
	}
	rectangle(img, face.rect, Scalar(255, 0, 0), 2);
}

Mat process(Mat& img) {

	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);

	if (!refreshFace(img, gray)) {
		findFace(img, gray);
	}
	drawFace(img, camFace);

	/*
		seamlessClone(trReplHead, img, trMask, pointsCenter(facePoints), img, NORMAL_CLONE);
	}*/

	return img;
}

int main(int argc, char** argv) {
	VideoCapture cap(0);
	static const string WIN = "Face swapper - camera";
	static const string WIN_SW = "Face swapper - swap";
	Mat frame;
	bool noError = cap.isOpened();

	if (!noError) {
		return -1;
	}

	namedWindow(WIN, WINDOW_AUTOSIZE);
	namedWindow(WIN_SW, WINDOW_AUTOSIZE);

	for (;;) {
		cap >> frame;

		imshow(WIN, process(frame));
		if (!swapFaceImg.empty()) {
			imshow(WIN_SW, swapFaceImg);
		}

		int key = waitKey(1);

		if (key == 27) {// ESC
			break;
		} else if (key == 48 || key == 176) {// 0
			swapFaceImg.release();
			swapFaceImg = Mat();
		} else if (key == 49 || key == 177) {// 1
			swapFaceImg = imread("./img/face1.jpg");
			swapFace = findFace(swapFaceImg);
			drawFace(swapFaceImg, swapFace);
		}
	}

	return 0;
}
