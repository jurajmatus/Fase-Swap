#include <opencv2/opencv.hpp>
#include "structs.hpp"
using namespace cv;
using namespace std;

// Classifiers, Trackers
CascadeClassifier faceDet;
CascadeClassifier eyeDet;

Head findHead(Mat img, Mat gray) {

	Head head;

	vector<Rect> faces;

	faceDet.detectMultiScale(gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE,
			Size(30, 30));

	for (uint i = 0; i < faces.size(); i++) {

		Mat faceROI = gray(faces[i]);
		std::vector<Rect> eyes;

		eyeDet.detectMultiScale(faceROI, eyes, 1.1, 2,0 | CASCADE_SCALE_IMAGE, Size(30, 30));

		if (eyes.size() < 2) {
			continue;
		}

		head.face = faces[i];
		head.eye1 = eyes[0];
		head.eye2 = eyes[1];
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

	goodFeaturesToTrack(gray, corners, 0, 0.01, 10, mask, 3, false, 0.04);

	return corners;
}

Mat oldGray;
vector<Point2f> oldFeatures;

vector<Point2f> computeFlow(Mat img, Mat gray, vector<Point2f> oldFeatures) {

	vector<uchar> status;
	vector<float> err;
	vector<Point2f> features;

	TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
	calcOpticalFlowPyrLK(oldGray, gray, oldFeatures, features, status, err, Size(10, 10), 3, termcrit, 0, 0.001);

	return features;
}

Mat process(Mat img) {

	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);

	if (oldFeatures.empty()) {
		Head head = findHead(img, gray);
		vector<Point2f> features = findFeatures(img, gray, head.face);
		oldFeatures = features;
	} else {
		oldFeatures = computeFlow(img, gray, oldFeatures);
	}

	for (uint i = 0; i < oldFeatures.size(); i++) {
		circle(img, oldFeatures[i], 2, Scalar(0, 255, 0), 2);
	}

	gray.copyTo(oldGray);

	return img;
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

	for (;;) {
		cap >> frame;

		imshow(WIN, process(frame));

		if (waitKey(30) != 255) {
			break;
		}
	}

	return 0;
}
