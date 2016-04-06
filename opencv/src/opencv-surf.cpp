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


#include <stdio.h>
#include <iostream>
#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
void readme();

int main( int argc, char** argv )
{

  if( argc != 2 ) {
    readme();
    return -1;
  }

  VideoCapture cap(0);

  if(!cap.isOpened()) {
    return -1;
  }

  namedWindow(":: OTC FLANN SURF MATCHES ::", 1);

  Mat img_1 = imread( argv[1], IMREAD_GRAYSCALE );
  Mat img_2;
  Mat frame;

  //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
  int minHessian = 400;
  Ptr<SURF> detector = SURF::create();
  detector->setHessianThreshold(minHessian);
  Mat descriptors_1;
  std::vector<KeyPoint> keypoints_1, keypoints_2;
  detector->detectAndCompute( img_1, Mat(), keypoints_1, descriptors_1 );
  FlannBasedMatcher matcher;

  for(;;) {

    Mat descriptors_2;
    std::vector<DMatch> matches;
    std::vector<DMatch> good_matches;

    cap >> frame;

    cvtColor(frame, img_2, COLOR_BGR2GRAY);

    if( !img_1.data || !img_2.data ) {
        std::cout<< " --(!) Error reading images " << std::endl;
        return -1;
    }

    detector->detectAndCompute(img_2, Mat(), keypoints_2, descriptors_2);

    //-- Step 2: Matching descriptor vectors using FLANN matcher
    matcher.match(descriptors_1, descriptors_2, matches);

    double max_dist = 0.3; double min_dist = 0.05;

    //-- Quick calculation of max and min distances between keypoints
    for( int i = 0; i < descriptors_1.rows; i++ ) {
        double dist = matches[i].distance;
        if( dist < min_dist ) {
            min_dist = dist;
        }
        if( dist > max_dist ) {
            max_dist = dist;
        }
    }

    printf("-- Max dist : %f \n", max_dist );
    printf("-- Min dist : %f \n", min_dist );

    //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
    //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
    //-- small)
    //-- PS.- radiusMatch can also be used here.
    for( int i = 0; i < descriptors_1.rows; i++ ) {
        if(matches[i].distance <= max(2*min_dist, 0.01) ) {
            good_matches.push_back(matches[i]);
        }
    }

    //-- Draw only "good" matches
    Mat img_matches;
    try {
        drawMatches( img_1, keypoints_1, img_2, keypoints_2,
                    good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                    vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        cout << "-- Good Matches : " << good_matches.size() << endl;
    } catch(...) {
        continue;
    }

    //-- Show detected matches
    imshow( ":: OTC FLANN SURF MATCHES ::", img_matches );

    /* for( int i = 0; i < (int)good_matches.size(); i++ ) {
        printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx );
    } */

    if(waitKey(30) >= 0) {
        break;
    }

  }

  return 0;

}

void readme() {
    std::cout << "-- Usage: ./otc-opencv-surf <img1>" << std::endl;
}
