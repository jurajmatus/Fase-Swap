#include "func.h"

vector<Point2f> matToPoints(Mat img) {
	vector<Point2f> points;
	points.push_back(Point2f(0, 0));
	points.push_back(Point2f(img.size().width, 0));
	points.push_back(Point2f(img.size().width, img.size().height));
	points.push_back(Point2f(0, img.size().height));
	return points;
}

vector<Point2f> rectToPoints(Rect rect) {
	vector<Point2f> points;
	points.push_back(Point2f(rect.x, rect.y));
	points.push_back(Point2f(rect.x + rect.width, rect.y));
	points.push_back(Point2f(rect.x + rect.width, rect.y + rect.height));
	points.push_back(Point2f(rect.x, rect.y + rect.height));
	return points;
}

vector<Point2i> pointsFToI(vector<Point2f> points) {
	vector<Point2i> ret;
	for (Point2f point : points) {
		ret.push_back(Point2i(point.x, point.y));
	}
	return ret;
}

Point pointsCenter(vector<Point2f> points) {
	vector<float> xs, ys;
	xs.resize(points.size());
	std::transform(points.begin(), points.end(), xs.begin(), [](Point p) {
		return p.x;
	});
	ys.resize(points.size());
	std::transform(points.begin(), points.end(), ys.begin(), [](Point p) {
		return p.y;
	});
	float maxX = *max_element(xs.begin(), xs.end());
	float maxY = *max_element(ys.begin(), ys.end());
	float minX = *min_element(xs.begin(), xs.end());
	float minY = *min_element(ys.begin(), ys.end());

	return Point((minX + maxX) / 2, (minY + maxY) / 2);
}

Size pointsMax(vector<Point2f> points) {
	vector<float> xs, ys;
	xs.resize(points.size());
	std::transform(points.begin(), points.end(), xs.begin(), [](Point p) {
		return p.x;
	});
	ys.resize(points.size());
	std::transform(points.begin(), points.end(), ys.begin(), [](Point p) {
		return p.y;
	});
	float maxX = *max_element(xs.begin(), xs.end());
	float maxY = *max_element(ys.begin(), ys.end());

	return Size(maxX, maxY);
}
