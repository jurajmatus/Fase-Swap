#include <opencv2/opencv.hpp>
#include "structs.hpp"
#include "func.h"

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
Head replHeadH;

const int NUM_FRAMES_TO_REFRESH = 10;
bool tryRefresh = true;
void refresh(int* counter) {
	(*counter)++;
	if (*counter >= NUM_FRAMES_TO_REFRESH) {
		*counter = 0;
		tryRefresh = true;
	}
}

vector<Point> estimateFacePoints(Mat& img, Mat& gray, Rect inside) {

	int histSize = 16;
	vector<Mat> planes;
	Mat hsv;
	cvtColor(img, hsv, CV_BGR2HSV);
	split(hsv, planes);
	float range[] = {0, 256};
	const float* histRange = {range};

	vector<Mat> hists = {Mat(), Mat(), Mat()};
	Mat mask = Mat::zeros(img.size(), CV_8U);
	mask(inside) = 255;

	for (int i = 0; i < 3; i++) {
		calcHist(&planes[i], 1, 0, mask, hists[i], 1, &histSize, &histRange, true, false);
		normalize(hists[i], hists[i], 0, 1, NORM_MINMAX);
	}

	Mat faceMask = Mat::zeros(mask.size(), mask.type());
	for (int row  = inside.y; row < inside.br().y; row++) {
		for (int col = inside.x; col < inside.br().x; col++) {
			bool yes = true;
			float sum = 0;
			for (int i = 0; i < 2; i++) {
				int index = planes[i].at<uchar>(row, col) / 16;
				//yes = yes && (hists[i].at<float>(index, 0) > 0.5);
				sum += hists[i].at<float>(index, 0);
			}
			yes = yes && sum > 1.3;
			faceMask.at<uchar>(row, col) = yes ? 255 : 0;
		}
	}
	imshow("Face mask", faceMask);

	vector<vector<Point>> contours;
	vector<Point> facePoints;
	vector<Vec4i> hierarchy;
	findContours(faceMask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	if (contours.size() > 0) {
		vector<Point> contour = *max_element(contours.begin(), contours.end(), [](vector<Point> v1, vector<Point> v2) {
			return v1.size() < v2.size();
		});

		convexHull(Mat(contour), facePoints);
	} else {
		facePoints = rectToPoints(inside);
	}

	RNG rng(12345);
	Mat faceHullImg = Mat::zeros(img.size(), CV_8UC3);
	vector<vector<Point>> polygons = {facePoints};
	Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255));
	drawContours(faceHullImg, polygons, 0, color, 3, 8, hierarchy, 0, Point());
	imshow("Contours", faceHullImg);

	return facePoints;

}

vector<Vec6f> triangluateHull(Mat& img, Mat& gray, Head& head) {

	Subdiv2D subdiv(head.face);
	for (auto &p : head.faceHullPoints) {
		subdiv.insert(p);
	}

	vector<Vec6f> triangleList;
	vector<Vec6f> ret;
	subdiv.getTriangleList(triangleList);
	vector<Point> p(3);

	Mat triangles = Mat::zeros(img.size(), CV_8UC3);

	for (auto &tr : triangleList) {
		bool outside = false;
		for (int i = 0; i < 3; i++) {
			p[i] = Point(cvRound(tr[i * 2]), cvRound(tr[i * 2 + 1]));
			if (!head.face.contains(p[i])) {
				outside = true;
				break;
			}
		}
		if (outside) {
			continue;
		}
		ret.push_back(tr);
		for (int i = 0; i < 3; i++) {
			line(triangles, p[i], p[(i + 1) % 3], Scalar(128, 128, 0), 2);
		}
	}

	imshow("Triangulation", triangles);

	return ret;

}

Head findHead(Mat& img, Mat& gray) {

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

		int xx = (x2 - x1) / 12;
		int yy = (y2 - y1) / 12;
		x1 += xx;
		x2 -= xx;
		y1 += yy;

		head.face = Rect(x1, y1, x2 - x1, y2 - y1);
		head.faceRectPoints = rectToPoints(head.face);
		head.faceHullPoints = estimateFacePoints(img, gray, head.face);
		head.faceTriangles = triangluateHull(img, gray, head);

		break;
	}

	return head;
}

vector<Point2f> findFeatures(Mat& img, Mat& gray, Rect inside = Rect(0, 0, 0, 0)) {

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

vector<Point2f> computeFlow(Mat& img, Mat& gray, vector<Point2f> oldFeatures) {

	vector<uchar> status;
	Mat err;
	vector<Point2f> features;

	TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
	calcOpticalFlowPyrLK(oldGray, gray, oldFeatures, features, status, err, Size(10, 10), 3, termcrit, 0, 0.001);

	return features;
}

Mat process(Mat& img) {

	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);
	vector<Point> facePoints;

	Head _head;
	if (tryRefresh) {
		_head = findHead(img, gray);
	}

	if (!_head.faceRectPoints.empty()) {
		head = _head;
		tryRefresh = false;
		detectedFeatures = findFeatures(img, gray, head.rect);
		oldFeatures = detectedFeatures;
		facePoints = head.faceRectPoints;
	} else {
		oldFeatures = computeFlow(img, gray, oldFeatures);
		if (!head.faceRectPoints.empty()) {
			headTransform = estimateRigidTransform(detectedFeatures, oldFeatures, true);
			if (!headTransform.empty()) {
				transform(head.faceRectPoints, facePoints, headTransform);
			} else {
				facePoints = head.faceRectPoints;
			}
		}
	}

	if (!facePoints.empty() && !replHead.empty()) {
		Mat trReplHead;
		vector<Point2f> replPoints = matToPoints(replHead);
		vector<Point2f> trReplPoints;

		Mat replTransformPers = findHomography(replPoints, facePoints, CV_RANSAC);
		vector<Point2f> _facePoints = pointsToF(facePoints);
		Mat replTransform = estimateRigidTransform(replPoints, _facePoints, true);
		transform(replPoints, trReplPoints, replTransform);

		Size trSize = pointsMax(pointsFToI(trReplPoints));
		Mat trMask = Mat::zeros(trSize, CV_8U);
		auto trReplPointsI = pointsFToI(trReplPoints);
		fillConvexPoly(trMask, trReplPointsI, Scalar(255));

		warpPerspective(replHead, trReplHead, replTransformPers, trSize);
		Point2f center = pointsCenter(facePoints);
		seamlessClone(trReplHead, img, trMask, Point2i(center.x, center.y), img, NORMAL_CLONE);
	}

	gray.copyTo(oldGray);

	return img;
}

Mat cropHead(Mat& img, Head& head) {

	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);

	head = findHead(img, gray);
	vector<Point2f> features = findFeatures(img, gray, head.rect);

	return head.face.width > 0 ? img(head.face) : Mat();
}

int main(int argc, char** argv) {
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
			Mat face = imread("./img/face1.jpg");
			replHead = cropHead(face, replHeadH);
		}
	}

	return 0;
}
