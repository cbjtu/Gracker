#include "Warpping.h"
#include <iostream>
bool Warpping::m_bPredictShape = false;

ostream& operator << (ostream& out, Warpping& w)
{
	out << w.getH() << endl;
	return out;
}


Warpping::Warpping(int nMode)
{
	m_nMode = nMode;

	zero();
}

Warpping::Warpping(const Warpping& w)
{
	m_nMode = w.m_nMode;

	H = w.H.clone();
	H_ = w.H_.clone();
}

Warpping::~Warpping()
{
}

const Warpping& Warpping::operator = (const Warpping& w) 
{
	m_nMode = w.m_nMode;

	H = w.H.clone();
	H_ = w.H_.clone();

	return *this;
}

void Warpping::initialize()
{
	int size = 3;

	H = cv::Mat::eye(size, size, CV_64FC1);
	H_ = cv::Mat::eye(size, size, CV_64FC1);

}

void Warpping::initialize(double x, double y)
{
	int size = 3;
	H = cv::Mat::eye(size, size, CV_64FC1);
	H.at<double>(0, 2) = x;
	H.at<double>(1, 2) = y;

	H_ = H.inv();
}


void Warpping::zero()
{
	int size = 3;

	H = cv::Mat::zeros(size, size, CV_64FC1);
	H_ = cv::Mat::zeros(size, size, CV_64FC1);

}

bool Warpping::isZero()
{
	return (cv::norm(H) < 1e-6);
}


Warpping Warpping::inv()
{
	Warpping w(this->m_nMode);
	w.H = this->H_.clone();
	w.H_ = this->H.clone();

	return w;
}

void  Warpping::setH(const cv::Mat& H_new) 
{
	H_new.copyTo(H);
	H_ = H.inv();
}


int Warpping::computeWarpping(vector<cv::Point2f>& src, vector<cv::Point2f>& dst, int method)
{
	if (m_nMode == WARPPINGMODE_PERSPECTIVE) {
		H = cv::findHomography(src, dst, 0); 

	/*	cv::Mat X = cv::Mat::zeros(3, src.size(), CV_64FC1);
		cv::Mat Y = cv::Mat::zeros(3, dst.size(), CV_64FC1);
		for (int i = 0; i < src.size(); i++) {
			X.at<double>(0, i) = src[i].x;
			X.at<double>(1, i) = src[i].y;
			X.at<double>(2, i) = 1.0;

			Y.at<double>(0, i) = dst[i].x;
			Y.at<double>(1, i) = dst[i].y;
			Y.at<double>(2, i) = 1.0;
		}

		H = (Y * X.t()) * (X * X.t()).inv();
	*/
	}
	else if (m_nMode == WARPPINGMODE_AFFINE){
		cv::Mat X = cv::Mat::zeros(3, src.size(), CV_64FC1);
		cv::Mat Y = cv::Mat::zeros(2, dst.size(), CV_64FC1);
		for (int i = 0; i < src.size(); i++) {
			X.at<double>(0, i) = src[i].x;
			X.at<double>(1, i) = src[i].y;
			X.at<double>(2, i) = 1.0;

			Y.at<double>(0, i) = dst[i].x;
			Y.at<double>(1, i) = dst[i].y;
		}

		cv::Mat A = (Y * X.t()) * (X * X.t()).inv();

		H = cv::Mat::eye(3, 3, CV_64FC1);
		H.at<double>(0, 0) = A.at<double>(0, 0);
		H.at<double>(0, 1) = A.at<double>(0, 1);
		H.at<double>(0, 2) = A.at<double>(0, 2);
		H.at<double>(1, 0) = A.at<double>(1, 0);
		H.at<double>(1, 1) = A.at<double>(1, 1);
		H.at<double>(1, 2) = A.at<double>(1, 2);
	}
	else if (m_nMode == WARPPINGMODE_SCALE_ONLY) {
		cv::Mat X = cv::Mat::zeros(3, src.size(), CV_64FC1);
		cv::Mat Y = cv::Mat::zeros(2, dst.size(), CV_64FC1);
		for (int i = 0; i < src.size(); i++) {
			X.at<double>(0, i) = src[i].x;
			X.at<double>(1, i) = src[i].y;
			X.at<double>(2, i) = 1.0;

			Y.at<double>(0, i) = dst[i].x;
			Y.at<double>(1, i) = dst[i].y;
		}

		cv::Mat A = (Y * X.t()) * (X * X.t()).inv();

		H = cv::Mat::eye(3, 3, CV_64FC1);
		H.at<double>(0, 0) = A.at<double>(0, 0);
		H.at<double>(0, 2) = A.at<double>(0, 2);
		H.at<double>(1, 1) = A.at<double>(1, 1);
		H.at<double>(1, 2) = A.at<double>(1, 2);

	}

	H_ = H.inv();

	return 1;
}

int Warpping::computeWarpping(vector<cv::Point2f>& src, vector<cv::Point2f>& dst, vector<cv::Point2f>& pred, double dPredWeight)
{
	cv::Point2f sc = getCenter(src);
	cv::Point2f dc = getCenter(dst);
	cv::Point2f pc = getCenter(pred);

	vector<cv::Point2f> scene;
	for (int i = 0; i < src.size(); i++) {
		float x = dc.x + (1.0 - dPredWeight) * (dst[i].x - dc.x) + dPredWeight * (pred[i].x - pc.x);
		float y = dc.y + (1.0 - dPredWeight) * (dst[i].y - dc.y) + dPredWeight * (pred[i].y - pc.y);
		scene.push_back(cv::Point2f(x, y));
	}

	return computeWarpping(src, scene);

//	H = cv::findHomography(src, scene, 0);
//	H_ = H.inv();
//	return 1;
}


cv::Point2f Warpping::doWarpping(const cv::Point2f pt)
{
	vector<cv::Point2f> obj;
	vector<cv::Point2f> scene;

	obj.push_back(pt);
	perspectiveTransform(obj, scene, H);

	return scene[0];
}

vector<cv::Point2f> Warpping::doWarpping(const vector<cv::Point2f>& pts)
{
	vector<cv::Point2f> obj;
	vector<cv::Point2f> scene;

	obj.assign(pts.begin(), pts.end());
	perspectiveTransform(obj, scene, H);

	return scene;
}



cv::Point2f Warpping::invWarpping(const cv::Point2f pt)
{
	vector<cv::Point2f> obj;
	vector<cv::Point2f> scene;

	obj.push_back(pt);
	perspectiveTransform(obj, scene, H_);

	return scene[0];
}

vector<cv::Point2f> Warpping::invWarpping(const vector<cv::Point2f>& pts)
{
	vector<cv::Point2f> obj;
	vector<cv::Point2f> scene;

	obj.assign(pts.begin(), pts.end());
	perspectiveTransform(obj, scene, H_);
	
	return scene;
}


Warpping Warpping::operator + (const Warpping& w)
{
	Warpping sum = *this;

	if (m_bPredictShape) {
		sum.H = H + w.H;
	}
	else{
		sum.H = H.clone();
		sum.H.at<double>(0, 2) += w.H.at<double>(0, 2);
		sum.H.at<double>(1, 2) += w.H.at<double>(1, 2);
	}

	sum.H_ = sum.H.inv();

	return sum;
}

Warpping Warpping::operator - (const Warpping& w)
{
	Warpping sum = *this;

	if (m_bPredictShape) {
		sum.H = H - w.H;
	}
	else{
		sum.H = H.clone();

		sum.H.at<double>(0, 2) -= w.H.at<double>(0, 2);
		sum.H.at<double>(1, 2) -= w.H.at<double>(1, 2);
	}

	sum.H_ = sum.H.inv();

	return sum;
}

Warpping Warpping::operator * (double scale)
{
	Warpping sum = *this;

	if (m_bPredictShape) {
		sum.H = H * scale;
	}
	else{
		sum.H.at<double>(0, 2) *= scale;
		sum.H.at<double>(1, 2) *= scale;
	}

	sum.H_ = sum.H.inv();

	return sum;
}

cv::Point2f Warpping::getCenter(vector<cv::Point2f>& pts)
{
	double sx = 0.0, sy = 0.0;
	for (int i = 0; i < pts.size(); i++) {
		sx += pts[i].x;
		sy += pts[i].y;
	}
	sx /= pts.size();
	sy /= pts.size();
	return cv::Point2f((float)sx, (float)sy);
}