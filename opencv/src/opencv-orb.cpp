/*

Author: Florian Lier [flier AT techfak.uni-bielefeld DOT de]

Starting Basis: OpenCV 3.1.0 CPP Examples

By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install, copy or use the software.

                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Copyright (C) 2000-2016, Intel Corporation, all rights reserved.
Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
Copyright (C) 2009-2016, NVIDIA Corporation, all rights reserved.
Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
Copyright (C) 2015-2016, OpenCV Foundation, all rights reserved.
Copyright (C) 2015-2016, Itseez Inc., all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall copyright holders or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.

*/


// STD
#include <vector>
#include <time.h>
#include <string>
#include <iostream>

// OPENCV
#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>

// CUDA
#include <opencv2/cudafeatures2d.hpp>

//BOOST
#include "boost/date_time/posix_time/posix_time.hpp"

using namespace std;
using namespace cv;

static void help() {
    cout << ">>>  ./otc-opencv-orb /path/to/config.yaml" << endl;
}

vector<Scalar> color_mix() {
    vector<Scalar> colormix;
    // Blue
    colormix.push_back(Scalar(219, 152, 52));
    // Cyan
    colormix.push_back(Scalar(173, 68, 142));
    // Orange
    colormix.push_back(Scalar(34, 126, 230));
    // Turquoise
    colormix.push_back(Scalar(156, 188, 26));
    // Pomgranate
    colormix.push_back(Scalar(43, 57, 192));
    // Asbestos
    colormix.push_back(Scalar(141, 140, 127));
    // Emerald
    colormix.push_back(Scalar(113, 204, 46));
    // White
    colormix.push_back(Scalar(241, 240, 236));
    // Green Sea
    colormix.push_back(Scalar(133, 160, 22));
    // Black
    colormix.push_back(Scalar(0, 0, 0));

    return colormix;
}

float euclideanDist(Point& p, Point& q) {
    Point diff = p - q;
    return cv::sqrt(diff.x*diff.x + diff.y*diff.y);
}

vector<Rect> pre_process_contours(Mat input_image, int thresh, RNG rng)
{
    Mat threshold_output;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    Mat src_gray;
    vector<Rect> rects;

    // Convert image to gray and blur it
    cvtColor(input_image, src_gray, CV_BGR2GRAY);
    blur(src_gray, src_gray, Size(3,3));

    // Detect edges using Threshold
    threshold(src_gray, threshold_output, thresh, 255, THRESH_BINARY );
    // Find contours
    findContours(threshold_output, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

    // Approximate contours to polygons + get bounding rects and circles
    vector<vector<Point>> contours_poly(contours.size());
    vector<Rect> boundRect(contours.size());

    for( int i = 0; i < contours.size(); i++ )
    {
        approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
        boundRect[i] = boundingRect(Mat(contours_poly[i]));
    }

    // Draw polygonal contour + bonding rects + circles
    Mat drawing = Mat::zeros( threshold_output.size(), CV_8UC3 );

    // cout << contours.size() << endl;
    for( int i = 0; i < contours.size(); i++ )
    {
        if (hierarchy[i][3] > -1) {
            if (boundRect[i].area() > 10000) {
                Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255));
                drawContours(drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point());
                rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );
                rects.push_back(boundRect[i]);
            }
        }
    }

    namedWindow(":: OTC Contours ::", CV_WINDOW_AUTOSIZE );
    imshow(":: OTC Contours ::", drawing );

    return rects;
}


int main(int argc, char *argv[])
{

    cout << ">>> CUDA Enabled Devices --> " << cuda::getCudaEnabledDeviceCount() << endl;

    if (cuda::getCudaEnabledDeviceCount() == 0)
    {
        cout << "E >>> No CUDA Enabled Devices" << endl;
        return -1;
    } else {
        cuda::printShortCudaDeviceInfo(cuda::getDevice());
    }

    vector<Scalar> colors;
    vector<String> target_paths;
    vector<String> target_labels;
    vector<Mat> target_images;
    // Ptr<Feature2D> orb;
    vector<vector<KeyPoint>> keys_current_target;
    // vector<Mat> desc_current_target_image;
    // vector<cuda::GpuMat> cuda_keys_current_target;
    vector<cuda::GpuMat> cuda_desc_current_target_image;

    const int fontFace = FONT_HERSHEY_PLAIN;
    const double fontScale = 1;
    const int num_to_skip = 60;
    const int text_origin = 10;
    int max_keypoints = 0;
    int max_number_matching_points = 0;
    int frame_num = 0;
    int text_offset_y = 20;
    int detection_threshold = 0;
    int res_x = 640;
    int res_y = 480;
    bool do_not_draw = false;
    bool toggle_homography = false;

    String type_descriptor;
    String point_matcher;
    String draw_homography;

    Mat camera_image, frame;
    VideoCapture cap(0);

    // Ptr<DescriptorMatcher> descriptorMatcher;
    // FlannBasedMatcher flann_matcher(new cv::flann::LshIndexParams(5,20,2));
    CommandLineParser parser(argc, argv, "{@config |<none>| yaml config file}" "{help h ||}");

    if (parser.has("help"))
    {
        help();
        return 0;
    }

    FileStorage fs(parser.get<string>(0), FileStorage::READ);

    if (fs.isOpened()) {\
        fs["keypointalgo"] >> type_descriptor;
        cout << ">>> Keypoint Descriptor --> " << type_descriptor  << endl;

        fs["matcher"] >> point_matcher;
        cout << ">>> Matching Algorithm --> " << point_matcher  << endl;

        fs["maxkeypoints"] >> max_keypoints;
        cout << ">>> Keypoints: --> " << max_keypoints  << endl;

        max_number_matching_points = fs["maxmatches"];
        cout << ">>> Maximum number of matching points --> " << max_number_matching_points << endl;

        detection_threshold = fs["detectionthreshold"];
        cout << ">>> Detection Threshold --> " << detection_threshold << endl;

        res_x = fs["resolutionx"];
        cout << ">>> Sensor X resolution --> " << res_x << endl;

        res_y = fs["resolutiony"];
        cout << ">>> Sensor Y resolution --> " << res_y << endl;

        fs["homography"] >> draw_homography;
        cout << ">>> Draw Homography --> " << draw_homography << endl;

        if (draw_homography == "true") {
            toggle_homography = true;
        }

        FileNode targets = fs["targets"];
        FileNodeIterator it = targets.begin(), it_end = targets.end();

        int idx = 0;

        for( ; it != it_end; ++it, idx++ ) {
            cout << ">>> Target " << idx << " --> ";
            cout << (String)(*it) << endl;
            target_paths.push_back((String)(*it));
        }

        if(idx > 5) {
            cout << "E >>> Sorry, only 5 targets are allowed (for now)" << endl;
            return -1;
        }

        FileNode labels = fs["labels"];
        FileNodeIterator it2 = labels.begin(), it_end2 = labels.end();

        int idy = 0;

        for( ; it2 != it_end2; ++it2, idy++ ) {
            cout << ">>> Label  " << idy << " --> ";
            cout << (String)(*it2) << endl;
            target_labels.push_back((String)(*it2));
        }
    }

    fs.release();

    colors = color_mix();

    cap.set(CV_CAP_PROP_FRAME_WIDTH, res_x);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, res_y);

    // descriptorMatcher = DescriptorMatcher::create(point_matcher);
    // orb = ORB::create(max_keypoints, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);

    Ptr<cuda::ORB> cuda_orb = cuda::ORB::create(max_keypoints, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);
    Ptr<cuda::DescriptorMatcher> cuda_bf_matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);

    // Compute Keypoints and Descriptors for all target images
    for(int i=0; i < target_paths.size(); i++) {

        Mat tmp_img = imread(target_paths[i], IMREAD_GRAYSCALE);
        cuda::GpuMat cuda_tmp_img(tmp_img);

        if (tmp_img.rows*tmp_img.cols <= 0) {
            cout << "E >>> Image " << target_paths[i] << " is empty or cannot be found" << endl;
            return -1;
        }

        target_images.push_back(tmp_img);

        vector<KeyPoint> tmp_kp;
        // Mat tmp_dc;

        // cuda::GpuMat tmp_cuda_kp;
        cuda::GpuMat tmp_cuda_dc;

        // orb->detect(tmp_img, tmp_kp, Mat());
        // orb->compute(tmp_img, tmp_kp, tmp_dc);

        cuda_orb->detectAndCompute(cuda_tmp_img, cuda::GpuMat(), tmp_kp, tmp_cuda_dc);

        if(point_matcher == "FlannBased") {
            // tmp_dc.convertTo(tmp_dc, CV_8U);
            cout << "E >>> Flann is not implemented with GPU support" << endl;
            return -1;
        }

        keys_current_target.push_back(tmp_kp);
        // desc_current_target_image.push_back(tmp_dc);

        // cuda_keys_current_target.push_back(tmp_cuda_kp);
        cuda_desc_current_target_image.push_back(tmp_cuda_dc);

    }

    if(!cap.isOpened()) {
        cout << "E >>> Camera cannot be opened" << endl;
        return -1;
    }

    // Keypoint for current_target_image and camera_image
    vector<KeyPoint> keys_camera_image;
    // cuda::GpuMat cuda_keys_camera_image;

    // Descriptor for current_target_image and camera_image
    // Mat desc_camera_image;
    cuda::GpuMat cuda_desc_camera_image;

    for(;;) {

        boost::posix_time::ptime start_all = boost::posix_time::microsec_clock::local_time();

        boost::posix_time::ptime start_cap = boost::posix_time::microsec_clock::local_time();

        cap >> frame;

        boost::posix_time::ptime end_cap = boost::posix_time::microsec_clock::local_time();

        // Check frames
        if (frame.rows*frame.cols <= 0) {
            cout << "E >>> Camera Image " << " is NULL\n";
            continue;
        }

        // Convert to grey
        cvtColor (frame, camera_image, COLOR_BGR2GRAY);

        // Skip first 30 frames in order to compensate color/brightness correction
        if ( ++frame_num < num_to_skip ) {
            continue;
        }

        boost::posix_time::ptime start_detect = boost::posix_time::microsec_clock::local_time();

        cuda::GpuMat cuda_tmp_img(camera_image);

        try {
            // orb->detectAndCompute(camera_image, Mat(), keys_camera_image, desc_camera_image, false);
            cuda_orb->detectAndCompute(cuda_tmp_img, cuda::GpuMat(), keys_camera_image, cuda_desc_camera_image);
        }
        catch (Exception& e) {
            cout << "E >>> ORB fail O_O" << "\n";
            continue;
        }

        boost::posix_time::ptime end_detect = boost::posix_time::microsec_clock::local_time();

        if (keys_camera_image.empty()) {
            cout << "E >>> Could not derive enough keypoints: " << endl;
            continue;
        }

        vector<double> cum_distance;
        vector<vector<DMatch>> cum_matches;

        boost::posix_time::ptime start_match = boost::posix_time::microsec_clock::local_time();

        for(int i=0; i < target_images.size(); i++) {
            vector<DMatch> matches;
            try {

                try {

                    if(!cuda_desc_current_target_image[i].empty() && !cuda_desc_camera_image.empty()) {

                        if(point_matcher == "FlannBased") {
                            // desc_camera_image.convertTo(desc_camera_image, CV_8U);
                            // flann_matcher.match(cuda_desc_current_target_image[i], desc_camera_image, matches);
                        } else {
                            // descriptorMatcher->match(cuda_desc_current_target_image[i], desc_camera_image, matches, Mat());
                            cuda_bf_matcher->match(cuda_desc_current_target_image[i], cuda_desc_camera_image, matches);
                        }

                        // Keep best matches only to have a nice drawing.
                        // We sort distance between descriptor matches
                        if (matches.size() <= max_number_matching_points) {
                            cout << "E >>> Not enough matches: " << matches.size() << endl;
                            continue;
                        }

                        Mat index;
                        int nbMatch=int(matches.size());

                        Mat tab(nbMatch, 1, CV_32F);

                        for (int i = 0; i<nbMatch; i++)
                        {
                            tab.at<float>(i, 0) = matches[i].distance;
                        }

                        sortIdx(tab, index, SORT_EVERY_COLUMN + SORT_ASCENDING);

                        vector<DMatch> bestMatches;

                        for (int i = 0; i<max_number_matching_points; i++)
                        {
                            if (matches[index.at<int>(i, 0)].distance <= detection_threshold*1.5) {
                                bestMatches.push_back(matches[index.at<int>(i, 0)]);
                            }
                        }

                        cum_matches.push_back(bestMatches);

                        vector<DMatch>::iterator it;
                        double raw_distance_sum = 0;

                        for (it = bestMatches.begin(); it != bestMatches.end(); it++) {
                            raw_distance_sum = raw_distance_sum + it->distance;
                        }

                        double mean_distance = raw_distance_sum/max_number_matching_points;
                        cum_distance.push_back(mean_distance);

                    } else {
                        do_not_draw = true;
                        cout << "E >>> Descriptors are empty" << endl;
                    }
                }
                catch (Exception& e) {
                    cout << "E >>> Cumulative distance cannot be computed, next iteration" << endl;
                    continue;
                }
            }
            catch (Exception& e) {
                cout << "E >>> Matcher is wating for input..." << endl;
                continue;
            }
        }

        boost::posix_time::ptime end_match = boost::posix_time::microsec_clock::local_time();

        text_offset_y = 20;

        if (do_not_draw == false) {

            for (int i=0; i < target_images.size(); i++) {
                try {
                    vector<DMatch>::iterator it;
                    vector<int> point_list_x;
                    vector<int> point_list_y;

                    for (it = cum_matches[i].begin(); it != cum_matches[i].end(); it++) {
                        // Point2d k_t = keys_current_target[i][it->queryIdx].pt;
                        Point2d c_t = keys_camera_image[it->trainIdx].pt;

                        point_list_x.push_back(c_t.x);
                        point_list_y.push_back(c_t.y);

                        Point2d current_point(c_t.x, c_t.y );
                        circle(frame, current_point, 3.0, colors[i], 1, 1 );
                    }

                    nth_element(point_list_x.begin(), point_list_x.begin() + point_list_x.size()/2, point_list_x.end());
                    nth_element(point_list_y.begin(), point_list_y.begin() + point_list_y.size()/2, point_list_y.end());

                    if (!point_list_x.empty() && !point_list_y.empty()) {
                        int median_x =  point_list_x[point_list_x.size()/2];
                        int median_y = point_list_y[point_list_y.size()/2];

                        Point2d location = Point2d(median_x, median_y);

                        if (cum_distance[i] <= detection_threshold) {
                            putText(frame, target_labels[i], location, fontFace, fontScale,
                                    colors[i], 2, LINE_AA);

                        }

                        string label = target_labels[i]+": ";
                        string distance_raw = to_string(cum_distance[i]);

                        putText(frame, label+distance_raw, Point2d(text_origin, text_offset_y), fontFace, fontScale, colors[i], 1, LINE_AA);
                        text_offset_y = text_offset_y+15;
                    }
                } catch (Exception& e) {
                    cout << "E >>> Could not derive median" << endl;
                    continue;
                }
            }
        }

        for (int i=0; i < target_images.size(); i++) {

            if (cum_distance[i] <= detection_threshold && toggle_homography) {
                try {
                    //-- localize the object
                    vector<Point2d> obj;
                    vector<Point2d> scene;

                    vector<DMatch>::iterator it;
                    for (it = cum_matches[i].begin(); it != cum_matches[i].end(); it++) {
                        obj.push_back(keys_current_target[i][it->queryIdx].pt);
                        scene.push_back(keys_camera_image[it->trainIdx].pt);
                    }

                    if( !obj.empty() && !scene.empty() && cum_matches[i].size() >= 4) {

                        Mat H = findHomography(obj, scene, cv::RANSAC);

                        //-- get the corners from the object to be detected
                        vector<cv::Point2d> obj_corners(4);
                        obj_corners[0] = Point(0, 0);
                        obj_corners[1] = Point(target_images[i].cols, 0);
                        obj_corners[2] = Point(target_images[i].cols, target_images[i].rows);
                        obj_corners[3] = Point(0, target_images[i].rows);

                        vector<Point2d> scene_corners(4);
                        vector<Point2f> scene_corners_f(4);

                        perspectiveTransform(obj_corners, scene_corners, H);

                        for (int i=0; i < scene_corners.size(); i++) {
                            scene_corners_f[i] = Point2f(scene_corners[i]);
                        }

                        cv::TermCriteria termCriteria = cv::TermCriteria(cv::TermCriteria::MAX_ITER| cv::TermCriteria::EPS, 20, 0.01);
                        cornerSubPix(camera_image, scene_corners_f, Size(10,10), Size(-1,-1), termCriteria);

                        //-- Draw lines between the corners (the mapped object in the scene - image_2 )
                        line(frame, scene_corners[0], scene_corners[1], Scalar(0,255,0), 4 );
                        line(frame, scene_corners[1], scene_corners[2], Scalar(0,255,0), 4 );
                        line(frame, scene_corners[2], scene_corners[3], Scalar(0,255,0), 4 );
                        line(frame, scene_corners[3], scene_corners[0], Scalar(0,255,0), 4 );
                    }

                } catch (Exception& e) {
                    cout << "E >>> Could not derive transform" << endl;
                }
            }
        }

        boost::posix_time::ptime end_all = boost::posix_time::microsec_clock::local_time();

        boost::posix_time::time_duration diff_all = end_all - start_all;
        boost::posix_time::time_duration diff_detect = end_detect - start_detect;
        boost::posix_time::time_duration diff_match = end_match - start_match;
        boost::posix_time::time_duration diff_cap = end_cap - start_cap;

        string string_time_all = to_string(diff_all.total_milliseconds());
        string string_time_detect = to_string(diff_detect.total_milliseconds());
        string string_time_match = to_string(diff_match.total_milliseconds());
        string string_time_cap = to_string(diff_cap.total_milliseconds());

        putText(frame, "Delta T (Capture): "+string_time_cap+" ms", Point2d(frame.cols-220, 20),
                fontFace, fontScale, Scalar(156, 188, 26), 1, LINE_AA);

        putText(frame, "Delta T (Detect): "+string_time_detect+" ms", Point2d(frame.cols-220, 40),
                fontFace, fontScale, Scalar(255,255,255), 1, LINE_AA);

        putText(frame, "Delta T (Match): "+string_time_match+" ms", Point2d(frame.cols-220, 60),
                fontFace, fontScale, Scalar(34, 126, 230), 1, LINE_AA);

        putText(frame, "Delta T (Full): "+string_time_all+" ms", Point2d(frame.cols-220, 80),
                fontFace, fontScale, Scalar(219, 152, 52), 1, LINE_AA);

        namedWindow(":: OTC ORB Detection ::", WINDOW_AUTOSIZE);
        imshow(":: OTC ORB Detection ::", frame);

        if(waitKey(1) >= 0) {
            destroyAllWindows();
            cap.release();
            return 0;
        }
    }

    return 0;
}
