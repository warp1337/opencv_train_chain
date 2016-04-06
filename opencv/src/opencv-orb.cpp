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

#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;

static void help()
{
    cout << "  ./otc-opencv-orb /path/to/config.yaml" << endl;
}



int main(int argc, char *argv[])
{
    vector<String> typeDesc;
    vector<String> typeAlgoMatch;
    vector<String> fileName;
    std::vector<String> target_paths;
    std::vector<String> target_labels;

    int number_of_points = 10;

    typeDesc.push_back("ORB");
    // typeDesc.push_back("BRISK");
    // typeAlgoMatch.push_back("BruteForce");
    // typeAlgoMatch.push_back("BruteForce-L1");
    typeAlgoMatch.push_back("BruteForce-Hamming");
    // typeAlgoMatch.push_back("BruteForce-Hamming(2)");

    cv::CommandLineParser parser(argc, argv, "{@config |<none>| yaml config file}" "{help h ||}");

    if (parser.has("help"))
    {
        help();
        return 0;
    }

    FileStorage fs(parser.get<string>(0), FileStorage::READ);
    if (fs.isOpened()) {
        cout << "Reading config file..." << endl;
        number_of_points = fs["maxmatches"];
        cout << "Maximum number of matching points --> " << number_of_points << endl;

        FileNode targets = fs["targets"];
        FileNodeIterator it = targets.begin(), it_end = targets.end();
        int idx = 0;

        for( ; it != it_end; ++it, idx++ )
        {
            cout << "Target " << idx << " --> ";
            cout << (String)(*it) << endl;
            target_paths.push_back((String)(*it));
        }

        FileNode labels = fs["labels"];
        FileNodeIterator it2 = labels.begin(), it_end2 = labels.end();
        int idx2 = 0;

        for( ; it2 != it_end2; ++it2, idx2++ )
        {
            cout << "Label " << idx2 << " --> ";
            cout << (String)(*it2) << endl;
            target_labels.push_back((String)(*it2));
        }
    }
    else {
        fs.release();
    }

    fs.release();

    vector<Mat> target_images;

    for(int i=0; i < target_paths.size(); i++){
       Mat tmp_img = imread(target_paths[i], IMREAD_GRAYSCALE);
       if (tmp_img.rows*tmp_img.cols <= 0) {
           cout << "Image " << target_paths[i] << " is empty or cannot be found\n";
           return 0;
       }
       target_images.push_back(tmp_img);
    }

    Mat camera_image, frame;

    VideoCapture cap(0);

    if(!cap.isOpened()) {
      return -1;
    }

    for(;;) {

        cap >> frame;

        cvtColor(frame, camera_image, COLOR_BGR2GRAY);

        if (camera_image.rows*camera_image.cols <= 0) {
            cout << "Camera Image " << " is NULL\n";
            continue;
        }

        vector<double> desMethCmp;
        Ptr<Feature2D> b;
        vector<String>::iterator itDesc;

        for (itDesc = typeDesc.begin(); itDesc != typeDesc.end(); itDesc++) {

            for(int i=0; i < target_images.size(); i++) {

                // Current Image
                Mat current_target_image = target_images[i];
                Ptr<DescriptorMatcher> descriptorMatcher;
                // Match between current_target_image and camera_image
                vector<DMatch> matches;
                // keypoint  for current_target_image and camera_image
                vector<KeyPoint> keys_current_target, keys_camera_image;
                // Descriptor for current_target_image and camera_image
                Mat desc_current_target_image, desc_camera_image;
                vector<String>::iterator itMatcher = typeAlgoMatch.end();
                if (*itDesc == "AKAZE-DESCRIPTOR_KAZE_UPRIGHT") {
                    b = AKAZE::create(AKAZE::DESCRIPTOR_KAZE_UPRIGHT);
                }
                if (*itDesc == "AKAZE") {
                    b = AKAZE::create();
                }
                if (*itDesc == "ORB"){
                    b = ORB::create();
                }
                else if (*itDesc == "BRISK") {
                    b = BRISK::create();
                }

                try {
                    // We can detect keypoint with detect method
                    b->detect(current_target_image, keys_current_target, Mat());
                    // and compute their descriptors with method  compute
                    b->compute(current_target_image, keys_current_target, desc_current_target_image);
                    // or detect and compute descriptors in one step
                    b->detectAndCompute(camera_image, Mat(), keys_camera_image, desc_camera_image, false);
                    // Match method loop
                    for (itMatcher = typeAlgoMatch.begin(); itMatcher != typeAlgoMatch.end(); itMatcher++) {

                        descriptorMatcher = DescriptorMatcher::create(*itMatcher);
                        if ((*itMatcher == "BruteForce-Hamming" || *itMatcher == "BruteForce-Hamming(2)") && (b->descriptorType() == CV_32F || b->defaultNorm() <= NORM_L2SQR))
                        {
                            cout << "**************************************************************************\n";
                            cout << "It's strange. You should use Hamming distance only for a binary descriptor\n";
                            cout << "**************************************************************************\n";
                        }
                        if ((*itMatcher == "BruteForce" || *itMatcher == "BruteForce-L1") && (b->defaultNorm() >= NORM_HAMMING))
                        {
                            cout << "**************************************************************************\n";
                            cout << "It's strange. You shouldn't use L1 or L2 distance for a binary descriptor\n";
                            cout << "**************************************************************************\n";
                        }
                        try {

                            descriptorMatcher->match(desc_current_target_image, desc_camera_image, matches, Mat());

                            // Keep best matches only to have a nice drawing.
                            // We sort distance between descriptor matches
                            if (matches.size() <= 1) {
                                cout << "Not enough Matches: " << matches.size() << endl;
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

                            for (int i = 0; i<number_of_points; i++)
                            {
                                bestMatches.push_back(matches[index.at<int>(i, 0)]);
                            }

                            Mat result;
                            drawMatches(current_target_image, keys_current_target, camera_image, keys_camera_image, bestMatches, result);

                            namedWindow(*itDesc+": "+*itMatcher, WINDOW_AUTOSIZE);
                            imshow(*itDesc + ": " + *itMatcher, result);

                            vector<DMatch>::iterator it;

                            // cout<<"**********Match results**********\n";
                            // cout << "Index \tIndex \tdistance\n";

                            // Use to compute distance between keyPoint matches and to evaluate match algorithm
                            // double cumSumDist2=0;

                            int idx = 0;
                            double raw_distance_sum = 0;

                            for (it = bestMatches.begin(); it != bestMatches.end(); it++) {
                                cout << "Point --> " << idx << "\t" << it->queryIdx << "\t" <<  it->trainIdx << "\t"  <<  it->distance << "\n";
                                // Point2d p=keys_current_target[it->queryIdx].pt-keys_camera_image[it->trainIdx].pt;
                                // cumSumDist2=p.x*p.x+p.y*p.y;
                                // cout << "Sum of first 30 best matches (x+y): " << cumSumDist2 << endl;
                                idx++;
                                raw_distance_sum = raw_distance_sum + it->distance;
                            }

                            double mean_distance = raw_distance_sum/number_of_points;
                            cout << "Mean of the first " << number_of_points << " 'best' matches (distance): " << mean_distance << endl;

                            // desMethCmp.push_back(cumSumDist2);

                            if(waitKey(30) >= 0) {
                                return 0;
                            }
                        }
                        catch (Exception& e) {
                            cout << e.msg << endl;
                            cout << "Cumulative distance cannot be computed." << endl;
                            desMethCmp.push_back(-1);
                        }
                    }
                }
                catch (Exception& e) {
                    cout << "Feature : " << *itDesc << "\n";
                    if (itMatcher != typeAlgoMatch.end()) {
                        cout << "Matcher : " << *itMatcher << "\n";
                    }
                    cout << e.msg << endl;
                }
            }
        }
    }

    /*
    int i=0;
    cout << "Cumulative distance between keypoint match for different algorithm and feature detector \n\t";
    cout << "We cannot say which is the best but we can say results are differents! \n\t";
    for (vector<String>::iterator itMatcher = typeAlgoMatch.begin(); itMatcher != typeAlgoMatch.end(); itMatcher++)
    {
        cout<<*itMatcher<<"\t";
    }
    cout << "\n";
    for (itDesc = typeDesc.begin(); itDesc != typeDesc.end(); itDesc++)
    {
        cout << *itDesc << "\t";
        for (vector<String>::iterator itMatcher = typeAlgoMatch.begin(); itMatcher != typeAlgoMatch.end(); itMatcher++, i++)
        {
            cout << desMethCmp[i]<<"\t";
        }
        cout<<"\n";
    }
    */

    return 0;
}
