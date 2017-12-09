#pragma once
#include "common.h"

#define WARPPINGMODE_AFFINE			0		// affine transformation
#define WARPPINGMODE_PERSPECTIVE	1		// perspective transformation
#define WARPPINGMODE_SCALE_ONLY		2		// scale only
#define WARPPINGMODE_SCALE_ROT		3		// scale and rotation

class Warpping
{
public:
	Warpping(int nMode);
	~Warpping();

	Warpping(const Warpping& w);
	const Warpping& operator = (const Warpping& w);


	void initialize();
	void initialize(double x, double y);

	void zero();

	bool isZero();

	int computeWarpping(vector<cv::Point2f>& src, vector<cv::Point2f>& dst, int method = 0);

	int computeWarpping(vector<cv::Point2f>& src, vector<cv::Point2f>& dst, vector<cv::Point2f>& pred, double dPredWeight);

	cv::Point2f doWarpping(const cv::Point2f pt);
	cv::Point2f invWarpping(const cv::Point2f pt);

	vector<cv::Point2f> doWarpping(const vector<cv::Point2f>& pts);
	vector<cv::Point2f> invWarpping(const vector<cv::Point2f>& pts);

	static cv::Point2f getCenter(vector<cv::Point2f>& pts);

	Warpping inv();


	cv::Mat getH()			{ return H.clone(); }

	void setH(const cv::Mat& H_new);
	void setWarppingMode(int mode)	{ m_nMode = mode; }
	static void setPredictShape(bool bPredict)	{ m_bPredictShape = bPredict; }

public:

	Warpping operator + (const Warpping& w);
	Warpping operator - (const Warpping& w);
	Warpping operator * (double scale);

protected:
	cv::Mat H;
	cv::Mat H_;

	static bool m_bPredictShape;

	int m_nMode = WARPPINGMODE_PERSPECTIVE;

};

ostream& operator << (ostream& out, Warpping& w);

