#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include<opencv2/xfeatures2d.hpp>
#include "DSR.h"

using namespace std;
using namespace cv;
using namespace xfeatures2d;

int main(int* arg, int** argv)
{
	Mat imgL, imgR;
	vector<Point2f> cornerPoints[2];
	imgL = imread("../data/image0_s.png");
	imgR = imread("../data/image1_s.png");

	/******************SIFT keypoints Detect and Match ************************/
	int numFeatures = 500;
	//create detector
	Ptr<SIFT> detector = SIFT::create(numFeatures);
	vector<KeyPoint> keypoints1, keypoints2;
	detector->detect(imgL, keypoints1);
	detector->detect(imgR, keypoints2);

	cout << "Keypoints1:" << keypoints1.size() << endl;
	cout << "Keypoints2:" << keypoints2.size() << endl;

	Mat drawimgL, drawimgR;
	drawKeypoints(imgL, keypoints1, drawimgL);
	drawKeypoints(imgR, keypoints2, drawimgR);

	Mat Descriptor1, Descriptor2;
	Ptr<SiftDescriptorExtractor> descriptor = SiftDescriptorExtractor::create();
	descriptor->compute(imgL, keypoints1, Descriptor1);
	descriptor->compute(imgR, keypoints2, Descriptor2);

	BFMatcher matcher(NORM_L2);
	vector<DMatch> matches;
	matcher.match(Descriptor1, Descriptor2, matches);

	double max_dist = 0;
	double min_dist = 1000;
	for (int i = 1; i < Descriptor1.rows; ++i)
	{
		double dist = matches[i].distance;
		if (dist > max_dist)
			max_dist = dist;
		if (dist < min_dist)
			min_dist = dist;
	}
	cout << "min_dist=" << min_dist << endl;
	cout << "max_dist=" << max_dist << endl;
 
	vector<DMatch> goodMatches;
	for (int i = 0; i < matches.size(); ++i)
	{
		double dist = matches[i].distance;
		if (dist < 4 * min_dist)
			goodMatches.push_back(matches[i]);
	}
	cout << "goodMatches:" << goodMatches.size() << endl;


	Mat result;
	drawMatches(imgL, keypoints1, imgR, keypoints2, goodMatches, result,
		Scalar(0, 255, 255), Scalar::all(-1));

	vector<KeyPoint> R_keypoint01, R_keypoint02;
	for (int i = 0; i < goodMatches.size(); i++)
	{
		R_keypoint01.push_back(keypoints1[goodMatches[i].queryIdx]);
		R_keypoint02.push_back(keypoints2[goodMatches[i].trainIdx]);
	}

	for (int i = 0; i < goodMatches.size();i++)
	{
		cornerPoints[0].push_back(R_keypoint01[i].pt);
		cornerPoints[1].push_back(R_keypoint02[i].pt);
	}

	/****************** findHomographyDSR ***************************/

	int num_trials = 200;
	int point_pick = 20;  //number of trialsand number of points picked for RANSAC
	int calib_height = imgL.rows;
	int calib_width = imgL.cols;
	int threshold = 1;
	float score = 0;

	Mat H_DSR = findHomographyDSR(cornerPoints[0], cornerPoints[1], num_trials, point_pick, calib_height, calib_width, threshold, score);

	cout << "findHomographyDSR Successful" << endl;
	/*************** Warp source image based on homography **********/

	Mat imgR_warped;
	warpPerspective(imgR, imgR_warped, H_DSR, imgR.size());

	imwrite("../data/imgR_warped.jpg", imgR_warped);
	imwrite("../data/imgL.jpg", imgL);

	/****************** StereoBM ***************************/

	Ptr<StereoBM> sgbm=cv::StereoBM::create();
	Mat left_disp;
	Mat im_outDSRGary, imgLGary;
	cvtColor(imgL, imgLGary, COLOR_BGR2GRAY);
	cvtColor(imgR_warped, im_outDSRGary, COLOR_BGR2GRAY);

	sgbm->compute(imgLGary, im_outDSRGary, left_disp);
	imwrite("../data/left_disp.jpg", left_disp);

	cout << "StereoBM Successful" << endl;

	return 0;
}