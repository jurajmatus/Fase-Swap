#include "func.h"

/**
 * Finds corner points of Mat
 */
vector<Point2f> matToPoints(Mat img) {
	vector<Point2f> points;
	points.push_back(Point2f(0, 0));
	points.push_back(Point2f(img.size().width, 0));
	points.push_back(Point2f(img.size().width, img.size().height));
	points.push_back(Point2f(0, img.size().height));
	return points;
}

/**
 * Finds corner points of Rect
 */
vector<Point> rectToPoints(Rect rect) {
	vector<Point> points;
	points.push_back(Point(rect.x, rect.y));
	points.push_back(Point(rect.x + rect.width, rect.y));
	points.push_back(Point(rect.x + rect.width, rect.y + rect.height));
	points.push_back(Point(rect.x, rect.y + rect.height));
	return points;
}

/**
 * Changes points array from float to int precision
 */
vector<Point2i> pointsFToI(vector<Point2f> points) {
	vector<Point2i> ret;
	for (Point2f point : points) {
		ret.push_back(Point2i(point.x, point.y));
	}
	return ret;
}

/**
 * Changes points array from default to int precision
 */
vector<Point2i> pointsToI(vector<Point> points) {
	vector<Point2i> ret;
	for (Point point : points) {
		ret.push_back(Point2i(point.x, point.y));
	}
	return ret;
}

/**
 * Changes points array from default to float precision
 */
vector<Point2f> pointsToF(vector<Point> points) {
	vector<Point2f> ret;
	for (Point point : points) {
		ret.push_back(Point2f(point.x, point.y));
	}
	return ret;
}

/**
 * Changes points array from int to float precision
 */
vector<Point2f> pointsItoF(vector<Point> points) {
	vector<Point2f> ret;
	for (Point point : points) {
		ret.push_back(Point2f(point.x, point.y));
	}
	return ret;
}

/**
 * Finds the center of many points
 */
Point pointsCenter(vector<Point> points) {
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

/**
 * Finds the Size that is the same as the beggest point coordinates in the points array
 */
Size pointsMax(vector<Point> points) {
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

/**
 * Finds boundig box of hull
 */
Rect hullToRect(vector<Point> hull) {
	vector<float> xs, ys;
	xs.resize(hull.size());
	std::transform(hull.begin(), hull.end(), xs.begin(), [](Point p) {
		return p.x;
	});
	ys.resize(hull.size());
	std::transform(hull.begin(), hull.end(), ys.begin(), [](Point p) {
		return p.y;
	});
	float maxX = *max_element(xs.begin(), xs.end());
	float maxY = *max_element(ys.begin(), ys.end());
	float minX = *min_element(xs.begin(), xs.end());
	float minY = *min_element(ys.begin(), ys.end());

	return Rect(minX, minY, maxX - minX, maxY - minY);
}

/**
 * Finds hull
 */
vector<Point> pointsToHull(vector<Point> points) {
	vector<Point> hullPoints;
	convexHull(Mat(points), hullPoints);
	return hullPoints;
}
