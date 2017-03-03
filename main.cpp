#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

CascadeClassifier faceDet;
CascadeClassifier eyeDet;

Mat process(Mat img) {

	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);

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

		Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
		ellipse(img, center, Size(faces[i].width / 2, faces[i].height / 2), 0,
				0, 360, Scalar(255, 0, 255), 4, 8, 0);

		for (uint j = 0; j < eyes.size(); j++) {
			Point eyeCenter(faces[i].x + eyes[j].x + eyes[j].width / 2,	faces[i].y + eyes[j].y + eyes[j].height / 2);
			int radius = cvRound((eyes[j].width + eyes[j].height) * 0.25);
			circle(img, eyeCenter, radius, Scalar(255, 0, 0), 4, 8, 0);
		}
	}

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
