#include <opencv2/opencv.hpp>

#include "face.hpp"
#include "func.h"

using namespace cv;
using namespace std;

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

Mat repl;
Mat process(Mat& img) {

	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);

	auto faces = findFaces(img);
	img.copyTo(repl);

	for (uint i = 0; i < faces.size(); i++) {
		uint j = (i + 1) % faces.size();
		if (i == j) {
			break;
		}

		doSwap(img, faces[i], repl, faces[j]);
	}

	return repl;
}

int main(int argc, char** argv) {
	VideoCapture cap(0);
	static const string WIN = "Face swapper - camera";
	Mat frame;
	bool noError = cap.isOpened();

	if (!noError) {
		return -1;
	}

	namedWindow(WIN, WINDOW_AUTOSIZE);

	for (;;) {
		cap >> frame;

		imshow(WIN, process(frame));

		int key = waitKey(1);

		if (key == 27) {// ESC
			break;
		}
	}

	return 0;
}
