#include "DSR.h"

template <typename T>
Mat get_Hk(T points1, T points2, Mat singleH)
{
	Mat Hk = (Mat_<double>(3, 3) <<
		1, 0, 0,
		0, 1, 0, 
		0, 0, 1);
	int num = points2.size();
	Mat pts2_t_norm;
	Mat X = Mat::ones(3, num, CV_64FC1);

	for (size_t i = 0; i < num; i++)
	{
		X.ptr<double>(0)[i] = points2[i].x;
		X.ptr<double>(1)[i] = points2[i].y;
	}

	Mat pts2_t = singleH * X; //apply tranform

	repeat(pts2_t.row(2), 3, 1, pts2_t_norm);

	pts2_t = pts2_t / pts2_t_norm;

	//The implementation tried to eliminate the effect of outliers
	vector<double>	errors;
	for (size_t i = 0; i < num; i++)
	{
		if (abs(pts2_t.ptr<double>(1)[i] - points1[i].y)<1)
		{
			errors.push_back((points1[i].x - pts2_t.ptr<double>(0)[i] ));

		}
	}
	sort(errors.begin(),errors.end()); 

	double K = 0;

	// Not consider outlier on x - axis
	K = errors[0];
	Hk.ptr<double>(0)[2] = K;

	return Hk;
}

Mat get_Hs(Mat singleH, int h, int w)
{
	Mat Hs = Mat::eye(3, 3, CV_64F);
	Mat vertex = (Mat_<double>(3, 4) <<
		w / 2., double(w), w / 2.,	0,
		0.0, h / 2., double(h), h / 2.,
		1., 1., 1., 1.);
	Mat vertex_t_norm;

	Mat vertex_t= singleH * vertex;

	repeat(vertex_t.row(2), 3, 1, vertex_t_norm);

	vertex_t = vertex_t / vertex_t_norm;

	double ux, uy, vx, vy;
	double sa, sb;
	ux = vertex_t.ptr<double>(0)[1] - vertex_t.ptr<double>(0)[3];
	uy = vertex_t.ptr<double>(1)[1] - vertex_t.ptr<double>(1)[3];

	vx = vertex_t.ptr<double>(0)[0] - vertex_t.ptr<double>(0)[2];
	vy = vertex_t.ptr<double>(1)[0] - vertex_t.ptr<double>(1)[2];

	sa = (h * h * uy * uy + w * w * vy *vy) / (h * w * (uy * vx - ux * vy));
	sb = (h * h * ux * uy + w * w * vx * vy) / (h * w * (ux * vy - uy * vx));

	if (sa < 0)
	{
		sa = -sa;
		sb = -sb;
	}

	Hs.ptr<double>(0)[0] = sa;
	Hs.ptr<double>(0)[1] = sb;

	return Hs;
}

template <typename T>
double get_align_rate(Mat Hy, T points1, T points2, int threshold)
{
	//Calculate alignment ratios for RANSAC
	int num = points2.size();
	Mat Y_norm;
	Mat X = Mat::ones(3, num, CV_64FC1);
	double  error = 0;
	int count = 0;
	double align_rate = 0;

	for (size_t i = 0; i < num; i++)
	{
		X.ptr<double>(0)[i] = points2[i].x;
		X.ptr<double>(1)[i] = points2[i].y;
	}

	Mat Y = Hy * X; //apply tranform

	repeat(Y.row(2), 3, 1, Y_norm);

	Y = Y / Y_norm;

	for (int i = 0; i < num; i++)
	{
		error = abs(Y.ptr<double>(1)[i] - points1[i].y);
		if (error<= threshold)
		{
			count++;
		}

	}

	align_rate = double(count) / double(num);

	return align_rate;
}
Mat get_Hy_single_trial(vector<Point2f> pts1, vector<Point2f> pts2)
{
	Mat Hy = Mat::eye(3, 3, CV_64F);
	int nums = pts1.size();
	Mat A = Mat::zeros(nums, 5, CV_64F);
	for (size_t i = 0; i < nums; i++)
	{
		A.ptr<double>(i)[0] = pts2[i].x;
		A.ptr<double>(i)[1] = pts2[i].y;
		A.ptr<double>(i)[2] = 1;
		A.ptr<double>(i)[3] = -1 * pts2[i].x * pts1[i].y;
		A.ptr<double>(i)[4] = -1 * pts2[i].y * pts1[i].y;
	  
	}

	Mat A_pinv;
	invert(A, A_pinv, DECOMP_SVD);

	Mat y=Mat::zeros(nums, 1, CV_64F);
	for (size_t i = 0; i < nums; i++)
	{
		y.ptr<double>(i)[0] = pts1[i].y;
	}

	Mat H_params = A_pinv * y;

	Hy.ptr<double>(1)[0] = H_params.ptr<double>(0)[0];
	Hy.ptr<double>(1)[1] = H_params.ptr<double>(1)[0];
	Hy.ptr<double>(1)[2] = H_params.ptr<double>(2)[0];
	Hy.ptr<double>(2)[0] = H_params.ptr<double>(3)[0];
	Hy.ptr<double>(2)[1] = H_params.ptr<double>(4)[0];

	return Hy;

}

//template <typename T>
Mat get_Hy(vector<Point2f>points1, vector<Point2f>points2, int num_trials, int num_samples, int threshold, float score)
{
	Mat Hy_tmp = Mat::eye(3, 3, CV_64F);
	Mat best_Hy;
	unsigned int n = points1.size();
	int ran_samples_idx = 0;
	RNG rng;
	vector<Point2f>ran_samples1, ran_samples2;
	double max_align_rate = 0;
	double align_rate_earlystop = 0.99;

	for (int i = 0; i < num_trials; i++)
	{
		for (int i = 0; i < num_samples; i++)
		{
			ran_samples_idx = rng(n);
			ran_samples1.push_back(points1[ran_samples_idx]);
			ran_samples2.push_back(points2[ran_samples_idx]);
		}

		Hy_tmp = get_Hy_single_trial(ran_samples1, ran_samples2);
		double align_rate = get_align_rate(Hy_tmp, points1, points2, threshold);

		if (align_rate > max_align_rate)
		{
			max_align_rate = align_rate;
			best_Hy = Hy_tmp;
		}

		if (align_rate > align_rate_earlystop)
			break;

		ran_samples1.clear();
		ran_samples2.clear();
	}

	return best_Hy;
}
/**
 * .
 * 
 * @param points1
 * @param points2
 * @param num_trials
 * @param num_samples
 * @param calib_height
 * @param calib_width
 * @param threshold
 * @param score
 * @return MatH
 */
Mat findHomographyDSR(vector<Point2f>points1, vector<Point2f>points2, int num_trials, int num_samples, int calib_height, int calib_width, int threshold,float score)
{
	Mat singleH =Mat::eye(3, 3, CV_64F);
	//Step 1, get Hy
	Mat Hy= get_Hy(points1, points2, num_trials, num_samples, threshold, score);
	singleH = Hy * singleH;

	//Step 2, get Hs
	Mat Hs = get_Hs(singleH, calib_height, calib_width);
	singleH = Hs * singleH;

	//Step 3, get Hk
	Mat Hk = get_Hk(points1, points2, singleH);
	singleH = Hk * singleH;

	return singleH;
}