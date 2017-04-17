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
	return false;

	if (camFace.points.empty() || camFaceAge >= NUM_FRAMES_TO_REFRESH) {
		return false;
	}
	if (oldGray.empty()) {
		gray.copyTo(oldGray);
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

	gray.copyTo(oldGray);
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
	warpAffine(small, transformed, transform, transformed.size());

	Mat srcMask = Mat::zeros(transformed.size(), CV_8U);
	vector<Point> srcMaskTr = pointsFToI(dstTr);
	fillConvexPoly(srcMask, srcMaskTr, Scalar(255));

	transformed.copyTo(dst(dstR), srcMask);

}

void extractFace(Mat& img, Mat& outputFaceImg, Mat& outputFaceMask, Face& face) {
	Rect rect = boundingRect(face.hullPoints);
	Size size = Size(rect.width, rect.height);
	outputFaceMask = Mat::zeros(size, CV_8U);

	vector<Point> hull = face.hullPoints;
	for (auto &p : hull) {
		p.x -= rect.x;
		p.y -= rect.y;
	}
	fillConvexPoly(outputFaceMask, hull, Scalar(255));

	img(rect).copyTo(outputFaceImg, outputFaceMask);
}

void doSwap(Mat& src, Face& srcFace, Mat& dst, Face& dstFace) {

	vector<Point2f> srcPoints = pointsToF(srcFace.points);
	Rect srcRect = boundingRect(srcPoints);
	for (auto &p : srcPoints) {
		p.x -= srcRect.x;
		p.y -= srcRect.y;
	}

	vector<Point2f> dstPoints = pointsToF(dstFace.points);
	Rect dstRect = boundingRect(dstPoints);
	for (auto &p : dstPoints) {
		p.x -= dstRect.x;
		p.y -= dstRect.y;
	}

	Mat homography = findHomography(srcPoints, dstPoints);

	Mat face1, mask1;
	extractFace(src, face1, mask1, srcFace);

	Mat face2, mask2;
	warpPerspective(face1, face2, homography, Size(dstRect.width, dstRect.height));

	mask2 = Mat::zeros(face2.size(), CV_8U);
	vector<Point> maskPoints = dstFace.hullPoints;
	for (auto &p : maskPoints) {
		p.x -= dstRect.x;
		p.y -= dstRect.y;
	}
	fillConvexPoly(mask2, maskPoints, Scalar(255));

	seamlessClone(face2, dst, mask2, pointsCenter(rectToPoints(dstRect)), dst, NORMAL_CLONE);

}

void doSwapByTriangluation(Mat& src, Face& srcFace, Mat& dst, Face& dstFace) {
	auto srcTriangles = triangluateHull(src, srcFace);
	drawTriangles(src, "Triangulation - swap", srcTriangles);

	auto dstTriangles = triangluateHull(dst, dstFace);
	drawTriangles(src, "Triangulation - camera", dstTriangles);
	Rect dstRect = boundingRect(dstFace.hullPoints);

	Size newFaceS = Size(dstRect.width, dstRect.height);
	Mat newFace = Mat::zeros(newFaceS, dst.type());
	Mat newFaceMask = Mat::zeros(newFaceS, CV_8U);

	uint l = min(srcTriangles.size(), dstTriangles.size());
	for (uint i = 0; i < l; i++) {

		auto st = srcTriangles[i];
		auto dt = dstTriangles[i];

		vector<Point2f> srcPoints = {Point2f(st[0], st[1]), Point2f(st[2], st[3]), Point2f(st[4], st[5])};
		vector<Point2f> dstPoints = {Point2f(dt[0], dt[1]), Point2f(dt[2], dt[3]), Point2f(dt[4], dt[5])};
		for (auto &p : dstPoints) {
			p.x -= max(0, dstRect.x);
			p.y -= max(0, dstRect.y);
		}
		copyTriangle(src, srcPoints, newFace, dstPoints);
	}

	vector<Point> newFacePoints = dstFace.hullPoints;
	for (auto &p : newFacePoints) {
		p.x -= dstRect.x;
		p.y -= dstRect.y;
	}
	fillConvexPoly(newFaceMask, newFacePoints, Scalar(255));

	seamlessClone(newFace, dst, newFaceMask, pointsCenter(rectToPoints(dstRect)), dst, NORMAL_CLONE);

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
		} else if ((key >= 49 && key <= 55) || (key >= 177 && key <= 183)) {// 1 - 7
			int num = key - 48;
			if (num < 1 || num > 7) {
				num = key - 176;
			}
			if (num < 1 || num > 7) {
				num = 1;
			}

			swapFaceImg = imread(format("./img/face%d.jpg", num));
			swapFace = findFace(swapFaceImg);
		}
	}

	return 0;
}
