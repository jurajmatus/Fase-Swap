#include <opencv2/opencv.hpp>
#include "structs.hpp"
#include "func.h"
#include "dlib_test.hpp"

using namespace cv;
using namespace std;

// Classifiers, Trackers
CascadeClassifier faceDet;
CascadeClassifier eyeDet;

// Tracking data
Mat oldGray;
vector<Point2f> oldFeatures;
Head head;
vector<Point2f> detectedFeatures;
Mat headTransform;
Mat replHead;

const int NUM_FRAMES_TO_REFRESH = 10;
bool tryRefresh = true;
void refresh(int* counter) {
	(*counter)++;
	if (*counter >= NUM_FRAMES_TO_REFRESH) {
		*counter = 0;
		tryRefresh = true;
	}
}

Head findHead(Mat img, Mat gray) {

	Head head;

	vector<Rect> faces;

	faceDet.detectMultiScale(gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE,
			Size(30, 30));

	for (uint i = 0; i < faces.size(); i++) {

		Mat faceROI = gray(faces[i]);
		std::vector<Rect> eyes;

		eyeDet.detectMultiScale(faceROI, eyes, 1.1, 2,0 | CASCADE_SCALE_IMAGE, Size(30, 30));

		if (eyes.size() < 1 || eyes.size() > 3) {
			continue;
		}

		head.rect = faces[i];
		head.eye1 = eyes[0];
		if (eyes.size() >= 2) {
			head.eye2 = eyes[1];
		}

		int x1 = head.rect.x;
		int x2 = x1 + head.rect.width;
		int y1 = head.rect.y;
		int y2 = y1 + head.rect.height;

		int xx = (x2 - x1) / 8;
		int yy = (y2 - y1) / 8;
		x1 += xx;
		x2 -= xx;
		y1 += yy * 2;
		y2 -= yy;

		head.face = Rect(x1, y1, x2 - x1, y2 - y1);
		head.facePoints = rectToPoints(head.face);

		break;
	}

	return head;
}

vector<Point2f> findFeatures(Mat img, Mat gray, Rect inside = Rect(0, 0, 0, 0)) {

	vector<Point2f> corners;
	Mat mask;
	if (inside.width > 0) {
		mask = Mat::zeros(img.size(), CV_8U);
		mask(inside) = 1;
	}

	goodFeaturesToTrack(gray, corners, 0, 0.01, 6, mask, 3, false, 0.04);

	vector<int> indices;
	vector<Point2f> hullPoints;
	convexHull(corners, indices, false, false);
	for (auto &i : indices) {
		hullPoints.push_back(corners.at(i));
	}

	return hullPoints;
}

vector<Point2f> computeFlow(Mat img, Mat gray, vector<Point2f> oldFeatures) {

	vector<uchar> status;
	Mat err;
	vector<Point2f> features;

	TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
	calcOpticalFlowPyrLK(oldGray, gray, oldFeatures, features, status, err, Size(10, 10), 3, termcrit, 0, 0.001);

	return features;
}

Mat process(Mat img) {

	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);
	vector<Point2f> facePoints;

	Head _head;
	if (tryRefresh) {
		_head = findHead(img, gray);
	}

	if (!_head.facePoints.empty()) {
		head = _head;
		tryRefresh = false;
		detectedFeatures = findFeatures(img, gray, head.rect);
		oldFeatures = detectedFeatures;
		facePoints = head.facePoints;
	} else {
		oldFeatures = computeFlow(img, gray, oldFeatures);
		if (!head.facePoints.empty()) {
			headTransform = estimateRigidTransform(detectedFeatures, oldFeatures, true);
			if (!headTransform.empty()) {
				transform(head.facePoints, facePoints, headTransform);
			} else {
				facePoints = head.facePoints;
			}
		}
	}

	if (!facePoints.empty() && !replHead.empty()) {
		Mat trReplHead;
		vector<Point2f> replPoints = matToPoints(replHead);
		vector<Point2f> trReplPoints;

		Mat replTransformPers = findHomography(replPoints, facePoints, CV_RANSAC);
		Mat replTransform = estimateRigidTransform(replPoints, facePoints, true);
		transform(replPoints, trReplPoints, replTransform);

		Size trSize = pointsMax(trReplPoints);
		Mat trMask = Mat::zeros(trSize, CV_8U);
		auto trReplPointsI = pointsFToI(trReplPoints);
		fillConvexPoly(trMask, trReplPointsI, Scalar(255));

		warpPerspective(replHead, trReplHead, replTransformPers, trSize);
		seamlessClone(trReplHead, img, trMask, pointsCenter(facePoints), img, NORMAL_CLONE);
	}

	gray.copyTo(oldGray);

	return img;
}

Mat cropHead(Mat img) {

	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);

	Head _head = findHead(img, gray);
	vector<Point2f> features = findFeatures(img, gray, _head.rect);

	return _head.face.width > 0 ? img(_head.face) : Mat();
}

int main(int argc, char** argv) {
	return executeDlib("data/face_features.dat", vector<string>({"./img/face1.jpg"}));

	VideoCapture cap(0);
	static const string WIN = "Face swapper";
	Mat frame;

	bool noError = faceDet.load("/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml");
	noError = noError && eyeDet.load("/usr/share/opencv/haarcascades/haarcascade_eye.xml");
	noError = noError && cap.isOpened();

	if (!noError) {
		return -1;
	}

	namedWindow(WIN, WINDOW_AUTOSIZE);

	int i = 0;
	for (;;) {
		refresh(&i);

		cap >> frame;

		imshow(WIN, process(frame));

		int key = waitKey(1);

		if (key == 27) {// ESC
			break;
		} else if (key == 48 || key == 176) {// 0
			replHead.release();
			replHead = Mat();
		} else if (key == 49 || key == 177) {// 1
			replHead = cropHead(imread("./img/face1.jpg"));
		}
	}

	return 0;
}
