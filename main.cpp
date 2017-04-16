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
}

void drawTriangles(Mat& img, String name, vector<Vec6f> triangles) {

	Mat drawing = Mat::zeros(img.size(), CV_8UC3);
	vector<Point> p(3);

	for (auto &tr : triangles) {
		for (int i = 0; i < 3; i++) {
			p[i] = Point(cvRound(tr[i * 2]), cvRound(tr[i * 2 + 1]));
		}
		for (int i = 0; i < 3; i++) {
			line(drawing, p[i], p[(i + 1) % 3], Scalar(128, 128, 0), 2);
		}
	}

	imshow(name, drawing);
}

vector<Vec6f> triangluateHull(Mat& img, Face& face) {

	Rect rect = Rect(0, 0, img.cols, img.rows);
	Subdiv2D subdiv(rect);
	for (auto &p : face.points) {
		subdiv.insert(Point2f(p.x, p.y));
	}

	vector<Vec6f> triangleList;
	vector<Vec6f> ret;
	subdiv.getTriangleList(triangleList);
	vector<Point> p(3);

	for (auto &tr : triangleList) {
		bool outside = false;
		for (int i = 0; i < 3; i++) {
			p[i] = Point(cvRound(tr[i * 2]), cvRound(tr[i * 2 + 1]));
			if (!rect.contains(p[i])) {
				outside = true;
				break;
			}
		}
		if (outside) {
			continue;
		}
		ret.push_back(tr);
	}

	cout << "Number of triangles: " << ret.size() << endl;
	return ret;

}

void doSwap(Mat& src, Face& srcFace, Mat& dst, Face& dstFace) {

	auto srcTriangles = triangluateHull(src, srcFace);
	drawTriangles(src, "Triangulation - camera", srcTriangles);

	auto dstTriangles = triangluateHull(dst, dstFace);
	drawTriangles(src, "Triangulation - swap", dstTriangles);

	for (int i = 0; i < min(srcTriangles.size(), dstTriangles.size()); i++) {
		// TODO - compute transform
		// TODO - transform part
		// TODO - seamless clone
	}

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
