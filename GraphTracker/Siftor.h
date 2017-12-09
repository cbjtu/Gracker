#pragma once

#include "common.h"


struct SIFT_D {
	int desc[128];
};

class Siftor
{
public:
	Siftor();
	~Siftor();

	int CalcPointSift(const cv::Mat& img, const vector<cv::Point2i>& vecPoints, vector<cv::KeyPoint>& keypoints, vector<SIFT_D>& descriptors);
	int CalcPointSift(const cv::Mat& img, const cv::Rect rect, vector<vector<KeyPoint>>& keypoints, vector<vector<SIFT_D>>& descriptors);
	int CalcRotationSift(const cv::Mat& img, const cv::Rect rect, int bins, vector<vector<SIFT_D>> desccriptors[]);

	int CalcMaximumSift(const cv::Mat& img, vector<cv::Point2i>& vecPoints, int max_radius, vector<cv::KeyPoint>& keypoints, vector<SIFT_D>& descriptors);
	int CalcMaximumSift(const cv::Mat& img, cv::Rect rect, int nGrid, vector<cv::KeyPoint>& keypoints, vector<SIFT_D>& descriptors);


	static bool getRotInvariant() { return m_bRotInvariant; }

protected:

	int CalcPointMag(const cv::Mat& img, cv::Point2i pt, double & mag, int & ori);
	int CalcPointDesc(const cv::Mat& img, cv::KeyPoint& kp, int* desc);
	int CalcBlockDesc(const cv::Mat& img, cv::KeyPoint& kp, cv::Point2i left_top, double* desc);

	cv::Mat rotateImage(cv::Mat& inputImg, cv::Point2f center, double angle);

protected:
	static const bool m_bRotInvariant = false;

};

