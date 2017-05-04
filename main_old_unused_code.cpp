#include <opencv2/opencv.hpp>

#include "face.hpp"
#include "func.h"

using namespace cv;
using namespace std;

Face findFace(Mat img) {
	return findFaces(img)[0];
}

// Number of frames to call full face finding algorithm
const int NUM_FRAMES_TO_REFRESH = 30;

// Face from the camera
Face camFace;
int camFaceAge = 0;

// Image of swap face and its information
Mat swapFaceImg;
Face swapFace;

/**
 * Function to find face and write it into global variable
 */
void findFace(Mat& img, Mat& gray) {
        camFace = findFace(img);
        camFaceAge = 0;
}

Mat oldGray;
/**
 * Refreshes face position
 */
bool refreshFace(Mat& img, Mat& gray) {
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

	// Parameters were used the same as in the example
	TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
	// Find corner points' positions in the new image by optical flow (to speed up)
	calcOpticalFlowPyrLK(oldGray, gray, oldPoints, newPoints, status, err, Size(10, 10), 3, termcrit, 0, 0.001);

	// This means optical flow hasn't found a correspondence
	if (newPoints.size() != oldPoints.size()) {
			return false;
	}
	// Save new points
	camFace.points = pointsFToI(newPoints);

	gray.copyTo(oldGray);
	camFaceAge++;
	return true;
}

/**
 * Function to find Delaunay triangulation of the face points
 */
vector<Vec6f> triangluateHull(Mat& img, Face& face) {

	Rect rect = Rect(0, 0, img.cols, img.rows);
	// Initialize triangulation class
	Subdiv2D subdiv(rect);
	// Insert all points
	for (auto &p : face.points) {
			subdiv.insert(Point2f(p.x, p.y));
	}

	vector<Vec6f> triangleList;
	vector<Vec6f> ret;
	// Find triangles
	subdiv.getTriangleList(triangleList);
	vector<Point> p(3);

	// Accumulate triangles not outside the image into the array
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

/**
 * Copies triangle from the source image into the destination image
 */
void copyTriangle(Mat& src, vector<Point2f> srcTr, Mat& dst, vector<Point2f> dstTr) {

	// Finds bounding boxes of both triangles
	Rect srcR = boundingRect(srcTr);
	Rect dstR = boundingRect(dstTr);

	// Shift triangle points relative to their raectangles
	for (auto &p : srcTr) {
			p.x -= srcR.x;
			p.y -= srcR.y;
	}
	for (auto &p : dstTr) {
			p.x -= dstR.x;
			p.y -= dstR.y;
	}

	// Copy rectangle containing source triangle into separate subimage
	Mat small;
	src(srcR).copyTo(small);

	// Compute transformation from source to destination triangle
	Mat transform = getAffineTransform(srcTr, dstTr);
	// Prepare image for the transformed triangle
	Mat transformed = Mat::zeros(Size(dstR.width, dstR.height), dst.type());
	// Transform the triangle into the prepared subimage
	warpAffine(small, transformed, transform, transformed.size());

	// Find mask of triangle inside the transformed image (the same as destination triangle
	// shifted relative to its rectangle)
	Mat srcMask = Mat::zeros(transformed.size(), CV_8U);
	vector<Point> srcMaskTr = pointsFToI(dstTr);
	fillConvexPoly(srcMask, srcMaskTr, Scalar(255));

	// Copy transformed triangle into the destination part
	transformed.copyTo(dst(dstR), srcMask);

}

void doSwapByTriangluation(Mat& src, Face& srcFace, Mat& dst, Face& dstFace) {
	// Find triangulation of source face
	auto srcTriangles = triangluateHull(src, srcFace);

	// Find triangulation of destination face
	auto dstTriangles = triangluateHull(dst, dstFace);
	Rect dstRect = boundingRect(dstFace.hullPoints);

	// Prepare image of new face (source transformed into shape of destination face)
	// The face is aggregated and only then copied into the destination image at once,
	// since seamless cloning is slow
	Size newFaceS = Size(dstRect.width, dstRect.height);
	Mat newFace = Mat::zeros(newFaceS, dst.type());
	Mat newFaceMask = Mat::zeros(newFaceS, CV_8U);

	uint l = min(srcTriangles.size(), dstTriangles.size());
	// Iterate over all triangles
	for (uint i = 0; i < l; i++) {

		auto st = srcTriangles[i];
		auto dt = dstTriangles[i];

		// Shift points relative to the bounding box
		vector<Point2f> srcPoints = {Point2f(st[0], st[1]), Point2f(st[2], st[3]), Point2f(st[4], st[5])};
		vector<Point2f> dstPoints = {Point2f(dt[0], dt[1]), Point2f(dt[2], dt[3]), Point2f(dt[4], dt[5])};
		for (auto &p : dstPoints) {
			p.x -= max(0, dstRect.x);
			p.y -= max(0, dstRect.y);
		}
		// Copy triangle from source image into the new face image
		copyTriangle(src, srcPoints, newFace, dstPoints);
	}

	// Find mask of face inside the new face image (the same as destination face points shifted
	// relative to their boundiing box)
	vector<Point> newFacePoints = dstFace.hullPoints;
	for (auto &p : newFacePoints) {
			p.x -= dstRect.x;
			p.y -= dstRect.y;
	}
	fillConvexPoly(newFaceMask, newFacePoints, Scalar(255));

	// Seamlessly clone (performs color blending) new face into the destination
	seamlessClone(newFace, dst, newFaceMask, pointsCenter(rectToPoints(dstRect)), dst, NORMAL_CLONE);

}

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
bool tryRefresh;


/**
 * Finds face points based on color histograms
 */
vector<Point> estimateFacePoints(Mat& img, Mat& gray, Rect inside) {

	// Number of histogram bins
	int histSize = 16;
	vector<Mat> planes;
	Mat hsv;
	// Convert image into HSV
	cvtColor(img, hsv, CV_BGR2HSV);
	split(hsv, planes);
	float range[] = {0, 256};
	const float* histRange = {range};

	vector<Mat> hists = {Mat(), Mat(), Mat()};
	// Prepare mask (inside of head rectangle)
	Mat mask = Mat::zeros(img.size(), CV_8U);
	mask(inside) = 255;

	// Calculate and normalize histogram for all the HSV planes
	for (int i = 0; i < 3; i++) {
		calcHist(&planes[i], 1, 0, mask, hists[i], 1, &histSize, &histRange, true, false);
		normalize(hists[i], hists[i], 0, 1, NORM_MINMAX);
	}

	// Prepare face mask
	Mat faceMask = Mat::zeros(mask.size(), mask.type());
	// Iterate all the pixels inside the region of interest
	for (int row  = inside.y; row < inside.br().y; row++) {
		for (int col = inside.x; col < inside.br().x; col++) {
			bool yes = true;
			float sum = 0;
			// Only H and S planes will be used (to minimize an impact of lightness)
			for (int i = 0; i < 2; i++) {
				// Find histogram index
				int index = planes[i].at<uchar>(row, col) / 16;
				sum += hists[i].at<float>(index, 0);
			}
			// If the sum of two histogram values is higher than 1.3, it's considerd to be a face
			yes = yes && sum > 1.3;
			faceMask.at<uchar>(row, col) = yes ? 255 : 0;
		}
	}

	vector<vector<Point>> contours;
	vector<Point> facePoints;
	vector<Vec4i> hierarchy;
	// Find contours of face mask
	findContours(faceMask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	if (contours.size() > 0) {
		// Find the contour with the most points
		vector<Point> contour = *max_element(contours.begin(), contours.end(), [](vector<Point> v1, vector<Point> v2) {
			return v1.size() < v2.size();
		});
		// Find convex hull of contour - the hull points will be face points
		convexHull(Mat(contour), facePoints);
	} else {
		// If no contour is found, rectangle will be used
		facePoints = rectToPoints(inside);
	}

	return facePoints;

}

/**
 * Finds head inside the image
 */
Head findHead(Mat& img, Mat& gray) {

	Head head;

	vector<Rect> faces;

	// Detect faces by Haar cascade detector
	faceDet.detectMultiScale(gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

	// Iterate all found faces
	for (uint i = 0; i < faces.size(); i++) {

		Mat faceROI = gray(faces[i]);
		std::vector<Rect> eyes;

		// Detect eyes by Haar cascade detector
		eyeDet.detectMultiScale(faceROI, eyes, 1.1, 2,0 | CASCADE_SCALE_IMAGE, Size(30, 30));

		// If there are less than 1 or more than 3 eyes, it's not considered a face
		// 1 and 3 eyes are accepted because of less than perfect detection rate
		if (eyes.size() < 1 || eyes.size() > 3) {
			continue;
		}

		// Fill structure with found data
		head.rect = faces[i];

		head.eye1 = eyes[0];
		if (eyes.size() >= 2) {
			head.eye2 = eyes[1];
		}

		int x1 = head.rect.x;
		int x2 = x1 + head.rect.width;
		int y1 = head.rect.y;
		int y2 = y1 + head.rect.height;

		// Shrink the rectangle slightly
		int xx = (x2 - x1) / 12;
		int yy = (y2 - y1) / 12;
		x1 += xx;
		x2 -= xx;
		y1 += yy;

		head.face = Rect(x1, y1, x2 - x1, y2 - y1);
		head.faceRectPoints = rectToPoints(head.face);
		head.faceHullPoints = estimateFacePoints(img, gray, head.face);
		head.faceTriangles = triangluateHull(img, gray, head);// Triangluate hull changed signature since

		break;
	}

	return head;
}

/**
 * Finds features to track
 */
vector<Point2f> findFeatures(Mat& img, Mat& gray, Rect inside = Rect(0, 0, 0, 0)) {

	vector<Point2f> corners;
	// Prepare mask with region of interest
	Mat mask;
	if (inside.width > 0) {
		mask = Mat::zeros(img.size(), CV_8U);
		mask(inside) = 1;
	}

	// Find corners
	goodFeaturesToTrack(gray, corners, 0, 0.01, 6, mask, 3, false, 0.04);

	vector<int> indices;
	vector<Point2f> hullPoints;
	// Find hull of corner points
	convexHull(corners, indices, false, false);
	for (auto &i : indices) {
		hullPoints.push_back(corners.at(i));
	}

	return hullPoints;
}

/**
 * Finds feature points' positions in new frame by optical flow
 */
vector<Point2f> computeFlow(Mat& img, Mat& gray, vector<Point2f> oldFeatures) {

	vector<uchar> status;
	Mat err;
	vector<Point2f> features;

	TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
	calcOpticalFlowPyrLK(oldGray, gray, oldFeatures, features, status, err, Size(10, 10), 3, termcrit, 0, 0.001);

	return features;
}

/**
 * Processing function
 */
Mat process(Mat& img) {

	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);
	vector<Point> facePoints;

	Head _head;
	// Find head if enough frames have elapsed
	if (tryRefresh) {
		_head = findHead(img, gray);
	}

	// Head found
	if (!_head.faceRectPoints.empty()) {
		head = _head;
		// Do not refresh for several frames
		tryRefresh = false;
		// Find features inside the head rectangle
		detectedFeatures = findFeatures(img, gray, head.rect);
		oldFeatures = detectedFeatures;
		facePoints = head.faceRectPoints;
	} else {
		// If head was not found (or wasn't attempted to find due to low number of frames having elapsed),
		// the position will be determined by optical flow
		oldFeatures = computeFlow(img, gray, oldFeatures);
		if (!head.faceRectPoints.empty()) {
			// Compute the transform between the original found face and current face obtained by optical flow
			headTransform = estimateRigidTransform(detectedFeatures, oldFeatures, true);
			if (!headTransform.empty()) {
				// If transform was successfully computed, the transformed rectangle will be used later
				transform(head.faceRectPoints, facePoints, headTransform);
			} else {
				// Otherwise the original rectangle will be used
				facePoints = head.faceRectPoints;
			}
		} else {
			// If no head is saved, we will refresh in the next frame
			tryRefresh = true;
		}
    }

    if (!facePoints.empty() && !replHead.empty()) {
		Mat trReplHead;
		vector<Point2f> replPoints = matToPoints(replHead);
		vector<Point2f> trReplPoints;

		// Find perspective transformation between replacing face and face in the image
		Mat replTransformPers = findHomography(replPoints, facePoints, CV_RANSAC);
		vector<Point2f> _facePoints = pointsToF(facePoints);

		// Find affine transformation between replacing face and face in the image
		Mat replTransform = estimateRigidTransform(replPoints, _facePoints, true);

		// Affinely transform hull points of replacing face
		// Those points will be used to find the size of the transformed image
		vector<Point2f> replHullPoints = pointsToF(replHeadH.faceHullPoints);
		for (auto &point : replHullPoints) {
			point.x -= replHeadH.face.x;
			point.y -= replHeadH.face.y;
		}
		transform(replHullPoints, trReplPoints, replTransform);

		// Prepare the image for transformed face
		Size trSize = pointsMax(pointsFToI(trReplPoints));
		Mat trMask = Mat::zeros(trSize, CV_8U);
		// Create mask containing the transformed face
		auto trReplPointsI = pointsFToI(trReplPoints);
		fillConvexPoly(trMask, trReplPointsI, Scalar(255));

		// Perspectively transform replacing head by calculated transform
		warpPerspective(replHead, trReplHead, replTransformPers, trSize);
		Point2f center = pointsCenter(facePoints);
		// Seamlessly clone the transformed face into the destination image
		seamlessClone(trReplHead, img, trMask, Point2i(center.x, center.y), img, NORMAL_CLONE);
    }

    gray.copyTo(oldGray);

    return img;
}

/**
 * Returns subimage containing only head
 */
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
			cap >> frame;

			imshow(WIN, process(frame));

			int key = waitKey(1);

			if (key == 27) {// ESC
				break;
			}
			// Set no replacement mode
			else if (key == 48 || key == 176) {// 0
				replHead.release();
				replHead = Mat();
			}
			// Set replacement mode - load face from file with the suffix of the pressed digit
			else if ((key >= 49 && key <= 56) || (key >= 177 && key <= 183)) {// 1 - 7
				int num = key - 48;
				if (num < 1 || num > 7) {
					num = key - 176;
				}
				if (num < 1 || num > 7) {
					num = 1;
				}

				Mat face = imread(format("./img/face%d.jpg", num));
				replHead = cropHead(face, replHeadH);
			}
	}

	return 0;
}







