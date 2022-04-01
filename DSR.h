#pragma once
#include <iostream>
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include<opencv2/features2d.hpp>

using namespace cv;
using namespace std;

Mat get_Hs(Mat singleH, int h, int w);
template <typename T>
double get_align_rate(Mat Hy, T points1, T points2, int threshold);
Mat get_Hy_single_trial(vector<Point2f> pts1, vector<Point2f> pts2);
Mat get_Hy(vector<Point2f>points1, vector<Point2f>points2, int num_trials, int num_samples, int threshold, float score);

Mat findHomographyDSR(vector<Point2f>points1, vector<Point2f>points2, int num_trials, int num_samples, int calib_height, int calib_width, int threshold, float score);

