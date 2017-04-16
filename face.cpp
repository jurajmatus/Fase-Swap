#include "face.hpp"

static const string datFile = "data/face_features.dat";

Face findFace(Mat _img) {

	dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
	dlib::shape_predictor sp;
	dlib::deserialize(datFile) >> sp;

	dlib::array2d <dlib::rgb_pixel> img;
	dlib::assign_image(img, dlib::cv_image<dlib::bgr_pixel>(_img));
	dlib::pyramid_up(img);

	vector<dlib::rectangle> dets = detector(img);

	vector<dlib::full_object_detection> shapes;
	for (unsigned long j = 0; j < dets.size(); ++j) {
		dlib::full_object_detection shape = sp(img, dets[j]);
		shapes.push_back(shape);
	}

	Face face;

	if (shapes.size() > 0) {
		face.rect = Rect(dets[0].left(), dets[0].top(), dets[0].width(), dets[0].height());
		auto shape = shapes[0];
		for (uint i = 0; i < shape.num_parts(); i++) {
			auto p = shape.part(0);
			face.points.push_back(Point(p.x(), p.y()));
		}
	}

	return face;

}
