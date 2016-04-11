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

#include <opencv2/core/utility.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <string>
#include <algorithm>
#include <functional>

using namespace std;
using namespace cv;

static void help() {
    cout << "  ./otc-opencv-orb /path/to/config.yaml" << endl;
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


int main(int argc, char *argv[])
{
    vector<Scalar> colors;
    vector<int> winner;
    vector<Point2d> last_winner;
    vector<vector<int>> train_list;
    vector<String> target_paths;
    vector<String> target_labels;

    int max_keypoints;
    int max_number_matching_points;
    const int fontFace = FONT_HERSHEY_PLAIN;
    const double fontScale = 1;
    int frame_num = 0;
    const int num_to_skip = 60;
    const int text_origin = 10;
    int text_offset_y = 20;
    int detection_threshold = 0;
    int res_x = 640;
    int res_y = 480;

    String type_descriptor;
    String point_matcher;

    // Descriptor Matcher
    Ptr<DescriptorMatcher> descriptorMatcher;
    // FlannBased
    // FlannBasedMatcher descriptorMatcherFlann(new flann::LshIndexParams(20,10,2));
    // FlannBasedMatcher descriptorMatcherFlann(new flann::LshIndexParams(5,24,2));

    cv::CommandLineParser parser(argc, argv, "{@config |<none>| yaml config file}" "{help h ||}");

    if (parser.has("help"))
    {
        help();
        return 0;
    }

    FileStorage fs(parser.get<string>(0), FileStorage::READ);

    if (fs.isOpened()) {\
        fs["keypointalgo"] >> type_descriptor;
        cout << "Keypoint Descriptor --> " << type_descriptor  << endl;

        fs["matcher"] >> point_matcher;
        cout << "Matching Algorithm --> " << point_matcher  << endl;

        fs["maxkeypoints"] >> max_keypoints;
        cout << "Keypoints: --> " << max_keypoints  << endl;

        max_number_matching_points = fs["maxmatches"];
        cout << "Maximum number of matching points --> " << max_number_matching_points << endl;

        detection_threshold = fs["detectionthreshold"];
        cout << "Detection Threshold --> " << detection_threshold << endl;

        res_x = fs["resolutionx"];
        cout << "Sensor X resolution --> " << res_x << endl;

        res_y = fs["resolutiony"];
        cout << "Sensor Y resolution --> " << res_y << endl;

        FileNode targets = fs["targets"];
        FileNodeIterator it = targets.begin(), it_end = targets.end();
        int idx = 0;

        for( ; it != it_end; ++it, idx++ )
        {
            cout << "Target " << idx << " --> ";
            cout << (String)(*it) << endl;
            target_paths.push_back((String)(*it));
        }

        if(idx > 5) {
            cout << "Sorry, only 5 targets are allowed (for now)" << endl;
            return 0;
        }

        FileNode labels = fs["labels"];
        FileNodeIterator it2 = labels.begin(), it_end2 = labels.end();
        int idx2 = 0;

        for( ; it2 != it_end2; ++it2, idx2++ )
        {
            cout << "Label  " << idx2 << " --> ";
            cout << (String)(*it2) << endl;
            target_labels.push_back((String)(*it2));
        }
    }

    fs.release();

    colors = color_mix();

    // Fill training list
    for(int i=0; i < target_paths.size(); i++) {
        train_list.push_back({0,0});
        winner.push_back(0);
        last_winner.push_back(Point2d(0,0));
    }

    vector<Mat> target_images;
    Ptr<Feature2D> b;
    // This is the standard for OpenCV
    // b = ORB::create(500, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);
    // b = ORB::create();
    b = ORB::create(max_keypoints, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);

    vector<vector<KeyPoint>> keys_current_target;
    vector<Mat> desc_current_target_image;

    // Compute Keypoints and Descriptors for all target images
    for(int i=0; i < target_paths.size(); i++){
        Mat tmp_img = imread(target_paths[i], IMREAD_GRAYSCALE);

        if (tmp_img.rows*tmp_img.cols <= 0) {
            cout << "Image " << target_paths[i] << " is empty or cannot be found\n";
            return 0;
        }

        target_images.push_back(tmp_img);

        vector<KeyPoint> tmp_kp;
        Mat tmp_dc;

        b->detect(tmp_img, tmp_kp, Mat());
        b->compute(tmp_img, tmp_kp, tmp_dc);

        keys_current_target.push_back(tmp_kp);
        desc_current_target_image.push_back(tmp_dc);
    }

    Mat camera_image, frame;
    VideoCapture cap(0);
    cap.set(CV_CAP_PROP_FRAME_WIDTH, res_x);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, res_y);

    if(!cap.isOpened()) {
        cout << "Camera cannot be opened" << endl;
        return -1;
    }

    descriptorMatcher = DescriptorMatcher::create(point_matcher);
    // Keypoint for current_target_image and camera_image
    vector<KeyPoint> keys_camera_image;
    // Descriptor for current_target_image and camera_image
    Mat desc_camera_image;

    for(;;) {
        cap >> frame;

        // Check frames
        if (frame.rows*frame.cols <= 0) {
            cout << "Camera Image " << " is NULL\n";
            continue;
        }

        // Convert to grey
        cvtColor(frame, camera_image, COLOR_BGR2GRAY);

        // Skip first 30 frames in order to compensate color/brightness correction
        if ( ++frame_num < num_to_skip )
            continue;

        try {
            b->detectAndCompute(camera_image, Mat(), keys_camera_image, desc_camera_image, false);
        }
        catch (Exception& e) {
            continue;
        }

        if (desc_camera_image.rows < 1 || desc_camera_image.cols < 1) {
            continue;
        }

        vector<double> cum_distance;
        vector<vector<DMatch>> cum_matches;

        for(int i=0; i < target_images.size(); i++) {
            vector<DMatch> matches;

            try {
                if ((point_matcher == "BruteForce-Hamming" || point_matcher == "BruteForce-Hamming(2)") && (b->descriptorType() == CV_32F || b->defaultNorm() <= NORM_L2SQR))
                {
                    cout << "**************************************************************************\n";
                    cout << "It's strange. You should use Hamming distance only for a binary descriptor\n";
                    cout << "**************************************************************************\n";
                }
                if (point_matcher == "FlannBased") {
                    if(desc_current_target_image[i].type() != CV_32F) {
                        desc_current_target_image[i].convertTo(desc_current_target_image[i], CV_32F);
                    }
                    if(desc_camera_image.type()!= CV_32F) {
                        desc_camera_image.convertTo(desc_camera_image, CV_32F);
                    }
                }

                try {
                    if (point_matcher == "FlannBased") {
                        // descriptorMatcherFlann.match(desc_current_target_image[i], desc_camera_image, matches);
                        cout << "Flann support is disbaled." << endl;
                        return 0;
                    } else {
                        descriptorMatcher->match(desc_current_target_image[i], desc_camera_image, matches, Mat());
                    }

                    // Keep best matches only to have a nice drawing.
                    // We sort distance between descriptor matches
                    if (int(matches.size()) <= max_number_matching_points) {
                        cout << "Not enough matches: " << matches.size() << endl;
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
                        bestMatches.push_back(matches[index.at<int>(i, 0)]);
                    }

                    cum_matches.push_back(bestMatches);

                    vector<DMatch>::iterator it;
                    double raw_distance_sum = 0;

                    for (it = bestMatches.begin(); it != bestMatches.end(); it++) {
                        raw_distance_sum = raw_distance_sum + it->distance;
                    }

                    double mean_distance = raw_distance_sum/max_number_matching_points;
                    cum_distance.push_back(mean_distance);
                }
                catch (Exception& e) {
                    cout << "Cumulative distance cannot be computed, next iteration" << endl;
                }
            }
            catch (Exception& e) {
                cout << "Matcher is wating for input..." << endl;
            }
        }

        text_offset_y = 20;

        for(int i=0; i < target_images.size(); i++) {
            vector<DMatch>::iterator it;
            vector<int> point_list_x;
            vector<int> point_list_y;

            for (it = cum_matches[i].begin(); it != cum_matches[i].end(); it++) {
                // Point2d k_t = keys_current_target[i][it->queryIdx].pt;
                Point2d c_t = keys_camera_image[it->trainIdx].pt;

                vector<int>::iterator it2;
                it2 = find(train_list[i].begin(), train_list[i].end(), it->trainIdx);
                if (it2 != train_list[i].end()) {
                    winner.at(i) = it->trainIdx;
                    train_list[i].clear();
                } else {
                    train_list[i].push_back(it->trainIdx);
                }

                point_list_x.push_back(c_t.x);
                point_list_y.push_back(c_t.y);
                circle(frame, c_t, 3.0, colors[i], 1, 1 );
            }

            nth_element(point_list_x.begin(), point_list_x.begin() + point_list_x.size()/2, point_list_x.end());
            nth_element(point_list_y.begin(), point_list_y.begin() + point_list_y.size()/2, point_list_y.end());

            int median_x =  point_list_x[point_list_x.size()/2];
            int median_y = point_list_y[point_list_y.size()/2];

            Point2d location = Point2d(median_x, median_y);

            if (cum_distance[i] <= detection_threshold) {
                putText(frame, target_labels[i], location, fontFace, fontScale, colors[i], 2, LINE_AA);

            }

            string label = target_labels[i]+": ";
            string distance_raw = to_string(cum_distance[i]);

            putText(frame, label+distance_raw, Point2d(text_origin, text_offset_y), fontFace, fontScale, colors[i], 1, LINE_AA);
            text_offset_y = text_offset_y+15;
        }

        namedWindow(":: OTC ORB ::", WINDOW_AUTOSIZE);
        imshow(":: OTC ORB ::", frame);

        if(waitKey(1) >= 0) {
            return 0;
        }
    }

    return 0;
}
