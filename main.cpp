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
	// TODO - hull points
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

	return ret;

}

void copyTriangle(Mat& src, vector<Point2f> srcTr, Mat& dst, vector<Point2f> dstTr) {

	Rect srcR = boundingRect(srcTr);
	Rect dstR = boundingRect(dstTr);

	for (auto &p : srcTr) {
		p.x -= srcR.x;
		p.y -= srcR.y;
	}
	for (auto &p : dstTr) {
		p.x -= dstR.x;
		p.y -= dstR.y;
	}

	Mat small;
	src(srcR).copyTo(small);

	Mat transform = getAffineTransform(srcTr, dstTr);
	Mat transformed = Mat::zeros(Size(dstR.width, dstR.height), dst.type());
	warpAffine(src, transformed, transform, transformed.size(), INTER_LINEAR, BORDER_REFLECT_101);

	Mat mask = Mat::zeros(transformed.size(), CV_8U);
	vector<Point> maskTr = pointsFToI(dstTr);
	fillConvexPoly(mask, maskTr, Scalar(255));

	seamlessClone(transformed, dst, mask, pointsCenter(rectToPoints(dstR)), dst, NORMAL_CLONE);

}

void doSwap(Mat& src, Face& srcFace, Mat& dst, Face& dstFace) {

	// TODO - cache
	auto srcTriangles = triangluateHull(src, srcFace);
	drawTriangles(src, "Triangulation - camera", srcTriangles);

	auto dstTriangles = triangluateHull(dst, dstFace);
	drawTriangles(src, "Triangulation - swap", dstTriangles);

	for (uint i = 0; i < min(srcTriangles.size(), dstTriangles.size()); i++) {

		auto st = srcTriangles[i];
		auto dt = dstTriangles[i];

		vector<Point2f> srcPoints = {Point2f(st[0], st[1]), Point2f(st[2], st[3]), Point2f(st[4], st[5])};
		vector<Point2f> dstPoints = {Point2f(dt[0], dt[1]), Point2f(dt[2], dt[3]), Point2f(dt[4], dt[5])};
		copyTriangle(src, srcPoints, dst, dstPoints);
	}

}

Mat process(Mat& img) {

	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);

	if (!refreshFace(img, gray)) {
		findFace(img, gray);
	}

	if (!swapFace.points.empty() && !camFace.points.empty()) {
		doSwap(swapFaceImg, swapFace, img, camFace);
	}

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
