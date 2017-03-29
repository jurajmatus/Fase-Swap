#ifndef DLIB_TEST_HPP_
#define DLIB_TEST_HPP_

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <iostream>

using namespace std;

int executeDlib(string datFile, vector<string> imgFiles);

#endif
