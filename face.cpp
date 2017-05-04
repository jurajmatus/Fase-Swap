#include "face.hpp"

/**
 * File with face features
 * Not commited in repository, needs to be downloaded
 */
static const string datFile = "data/face_features.dat";
static dlib::shape_predictor sp;
bool spInitialized = false;

/**
 * Function that finds all the faces in the image
 */
vector<Face> findFaces(Mat& _img) {

	// Lazy initialization of shape predictor
	if (!spInitialized) {
		dlib::deserialize(datFile) >> sp;
		spInitialized = true;
	}

	// Variables initialization
	dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
	dlib::array2d <dlib::rgb_pixel> img;

	// Conversion of OpenCV image into dlib image
	dlib::assign_image(img, dlib::cv_image<dlib::bgr_pixel>(_img));

	vector<dlib::rectangle> dets = detector(img);

	vector<dlib::full_object_detection> shapes;
	// Detection of face renctangles
	for (unsigned long j = 0; j < dets.size(); ++j) {
		dlib::full_object_detection shape = sp(img, dets[j]);
		shapes.push_back(shape);
	}

	vector<Face> faces;

	// Detection of face points
	for (uint i = 0; i < min(shapes.size(), dets.size()); i++) {
		auto shape = shapes[i];
		auto det = dets[i];
		Face face;
		face.rect = Rect(det.left(), det.top(), det.width(), det.height());

		for (uint i = 0; i < shape.num_parts(); i++) {
			auto p = shape.part(i);
			face.points.push_back(Point(p.x(), p.y()));
		}
		face.hullPoints = pointsToHull(face.points);

		faces.push_back(face);
	}

	return faces;

}
