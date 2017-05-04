#include <opencv2/opencv.hpp>

#include "face.hpp"
#include "func.h"

using namespace cv;
using namespace std;

/**
 * Function that takes face from image img according to face instance,
 * copies it into outputFaceImg and fills outputFaceMask
 */
void extractFace(Mat& img, Mat& outputFaceImg, Mat& outputFaceMask, Face& face) {
	// Prepare the mask of face, as big as the bounding box of the face
	Rect rect = boundingRect(face.hullPoints);
	Size size = Size(rect.width, rect.height);
	outputFaceMask = Mat::zeros(size, CV_8U);

	// Shift hull points to small mask coordinates (originally they are for the full image)
	vector<Point> hull = face.hullPoints;
	for (auto &p : hull) {
		p.x -= rect.x;
		p.y -= rect.y;
	}
	// Fill hull - it's the mask of face
	fillConvexPoly(outputFaceMask, hull, Scalar(255));

	// Copy the part inside the face rectangle defined by face mask into the output
	img(rect).copyTo(outputFaceImg, outputFaceMask);
}

/**
 * Takes the face defined by srcFace from src image and puts it into place defined by dstFace
 * in dst image
 */
void doSwap(Mat& src, Face& srcFace, Mat& dst, Face& dstFace) {

	// Find points of source face and shift them relative to the face's bounding box
	vector<Point2f> srcPoints = pointsToF(srcFace.points);
	Rect srcRect = boundingRect(srcPoints);
	for (auto &p : srcPoints) {
		p.x -= srcRect.x;
		p.y -= srcRect.y;
	}

	// Find points of destination face and shift them relative to the face's bounding box
	vector<Point2f> dstPoints = pointsToF(dstFace.points);
	Rect dstRect = boundingRect(dstPoints);
	for (auto &p : dstPoints) {
		p.x -= dstRect.x;
		p.y -= dstRect.y;
	}

	// Compute perspective transformation from source points to destination points
	// Points are obtained from dlib, which guarantees there are always 68 corresponding points
	Mat homography = findHomography(srcPoints, dstPoints);

	// Extract source face into separate subimage
	Mat face1, mask1;
	extractFace(src, face1, mask1, srcFace);

	// Transform extracted face into another separate subimage
	Mat face2, mask2;
	warpPerspective(face1, face2, homography, Size(dstRect.width, dstRect.height));

	// Find mask of transformed face in the subimage (the same as shifted hull in destination face)
	mask2 = Mat::zeros(face2.size(), CV_8U);
	vector<Point> maskPoints = dstFace.hullPoints;
	for (auto &p : maskPoints) {
		p.x -= dstRect.x;
		p.y -= dstRect.y;
	}
	fillConvexPoly(mask2, maskPoints, Scalar(255));

	// Seamlessly clone (performs color blending) transformed face into destination image
	seamlessClone(face2, dst, mask2, pointsCenter(rectToPoints(dstRect)), dst, NORMAL_CLONE);

}

/**
 * Copy of the image frame (so that we could perform swap after one face was replaced by another)
 */
Mat repl;

Mat process(Mat& img) {

	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);

	// Find all the faces in the image
	auto faces = findFaces(img);
	img.copyTo(repl);

	for (uint i = 0; i < faces.size(); i++) {
		uint j = (i + 1) % faces.size();
		if (i == j) {
			break;
		}

		// Perform swap with the next face in the circular manner (1<->2, 2<->3, 3<->1)
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
