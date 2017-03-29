// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

 This example program shows how to find frontal human faces in an image and
 estimate their pose.  The pose takes the form of 68 landmarks.  These are
 points on the face such as the corners of the mouth, along the eyebrows, on
 the eyes, and so forth.



 This face detector is made using the classic Histogram of Oriented
 Gradients (HOG) feature combined with a linear classifier, an image pyramid,
 and sliding window detection scheme.  The pose estimator was created by
 using dlib's implementation of the paper:
 One Millisecond Face Alignment with an Ensemble of Regression Trees by
 Vahid Kazemi and Josephine Sullivan, CVPR 2014
 and was trained on the iBUG 300-W face landmark dataset.

 Also, note that you can train your own models using dlib's machine learning
 tools.  See train_shape_predictor_ex.cpp to see an example.




 Finally, note that the face detector is fastest when compiled with at least
 SSE2 instructions enabled.  So if you are using a PC with an Intel or AMD
 chip then you should enable at least SSE2 instructions.  If you are using
 cmake to compile this program you can enable them by using one of the
 following commands when you create the build project:
 cmake path_to_dlib_root/examples -DUSE_SSE2_INSTRUCTIONS=ON
 cmake path_to_dlib_root/examples -DUSE_SSE4_INSTRUCTIONS=ON
 cmake path_to_dlib_root/examples -DUSE_AVX_INSTRUCTIONS=ON
 This will set the appropriate compiler options for GCC, clang, Visual
 Studio, or the Intel compiler.  If you are using another compiler then you
 need to consult your compiler's manual to determine how to enable these
 instructions.  Note that AVX is the fastest but requires a CPU from at least
 2011.  SSE4 is the next fastest and is supported by most current machines.
 */

#include "dlib_test.hpp"

// ----------------------------------------------------------------------------------------

int executeDlib(string datFile, vector<string> imgFiles) {
	try {

		// We need a face detector.  We will use this to get bounding boxes for
		// each face in an image.
		dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
		// And we also need a shape_predictor.  This is the tool that will predict face
		// landmark positions given an image and face bounding box.  Here we are just
		// loading the model from the shape_predictor_68_face_landmarks.dat file you gave
		// as a command line argument.
		dlib::shape_predictor sp;
		dlib::deserialize(datFile) >> sp;

		dlib::image_window win, win_faces;
		// Loop over all the images provided on the command line.
		for (string fileName : imgFiles) {
			cout << "processing image " << fileName << endl;
			dlib::array2d < dlib::rgb_pixel > img;
			dlib::load_image(img, fileName);
			// Make the image larger so we can detect small faces.
			dlib::pyramid_up (img);

			// Now tell the face detector to give us a list of bounding boxes
			// around all the faces in the image.
			std::vector<dlib::rectangle> dets = detector(img);
			cout << "Number of faces detected: " << dets.size() << endl;

			// Now we will go ask the shape_predictor to tell us the pose of
			// each face we detected.
			std::vector<dlib::full_object_detection> shapes;
			for (unsigned long j = 0; j < dets.size(); ++j) {
				dlib::full_object_detection shape = sp(img, dets[j]);
				cout << "number of parts: " << shape.num_parts() << endl;
				cout << "pixel position of first part:  " << shape.part(0)
						<< endl;
				cout << "pixel position of second part: " << shape.part(1)
						<< endl;
				// You get the idea, you can get all the face part locations if
				// you want them.  Here we just store them in shapes so we can
				// put them on the screen.
				shapes.push_back(shape);
			}

			// Now let's view our face poses on the screen.
			win.clear_overlay();
			win.set_image(img);
			win.add_overlay(render_face_detections(shapes));

			// We can also extract copies of each face that are cropped, rotated upright,
			// and scaled to a standard size as shown here:
			dlib::array<dlib::array2d<dlib::rgb_pixel> > face_chips;
			dlib::extract_image_chips(img, get_face_chip_details(shapes), face_chips);
			win_faces.set_image(tile_images(face_chips));

			cout << "Hit enter to process the next image..." << endl;
			cin.get();
		}
	} catch (exception& e) {
		cout << "\nexception thrown!" << endl;
		cout << e.what() << endl;
	}

	return 0;
}

// ----------------------------------------------------------------------------------------
