#include "Siftor.h"
#include "Polygon.h"
#include "Utils.h"
#include <iostream>

static const int BLOCK_SIZE = 4;

Siftor::Siftor()
{
}


Siftor::~Siftor()
{

}


int CountBlockMag(double* mag, int* ori, int len, int div, double* magcounts)
{
	for (int i = 0; i < div; i++)
		magcounts[i] = 0;

	for (int i = 0; i < len; i++) {
		int ind = div * ori[i] / 360;
		magcounts[ind] += mag[i];
	}

	return 1;
}

/**************************
* calculate sift with maximum response around given points
* input: 
*     img:	       image data
*     vecPoints:   coordinates of given points
*     max_radius:  maximum radius of the search region
* output:
*     keypoints:   keypoints[i] store the keypoint with maximum response around vecPoints[i] in distance < radius.
*     descriptors: descriptors of the keypoints
***************************/
int Siftor::CalcMaximumSift(const cv::Mat& img, vector<cv::Point2i>& vecPoints, int max_radius, vector<cv::KeyPoint>& keypoints, vector<SIFT_D>& descriptors)
{
	if (img.type() != CV_8UC1)
		return 0;

	if (vecPoints.size() <= 0)
		return 0;

	// for each point, calc minimum distance from other points
	vector<double> minDist(vecPoints.size(), 1e6);
	for (int i = 0; i < vecPoints.size(); i++) {
		for (int k = 0; k < vecPoints.size(); k++) {
			if (i == k)
				continue;
			double dist = sqrt((vecPoints[i].x - vecPoints[k].x) * (vecPoints[i].x - vecPoints[k].x) + (vecPoints[i].y - vecPoints[k].y) * (vecPoints[i].y - vecPoints[k].y));
			if (dist < minDist[i]) {
				minDist[i] = dist;
			}
		}
	}

	keypoints.clear();
	descriptors.clear();

	for (int kpi = 0; kpi < vecPoints.size(); kpi++) {
		double max_mag = 0.0;
		int max_ori = 0;
		cv::Point2i max_pt;

		int radius = min((int)(minDist[kpi] / 2.0), max_radius);
		int left = max(0, vecPoints[kpi].x - radius);
		int right = min(img.cols - 1, vecPoints[kpi].x + radius);
		int top = max(0, vecPoints[kpi].y - radius);
		int bottom = min(img.rows - 1, vecPoints[kpi].y + radius);

		for (int x = left; x <= right; x++) {
			for (int y = top; y <= bottom; y++) {
				double mag = 0.0;
				int ori = 0;
				CalcPointMag(img, cv::Point2i(x, y), mag, ori);
				if (mag > max_mag) {
					max_mag = mag;
					max_ori = ori;
					max_pt = cv::Point2i(x, y);
				}
			}
		}

		cv::KeyPoint kp;
		kp.pt.x = (float)max_pt.x;
		kp.pt.y = (float)max_pt.y;
		kp.angle = (float)max_ori;
		kp.response = (float)max_mag;
		kp.class_id = 0;
		kp.octave = 0;
		kp.size = 1;
		keypoints.push_back(kp);
	}

	sort(keypoints.begin(), keypoints.end(), CompareKeypoints);

	// filter too close points
	int index = 0;
	double dMinDist = 1.0;
	while (index < keypoints.size()) {
		double x = keypoints[index].pt.x;
		double y = keypoints[index].pt.y;
		vector<cv::KeyPoint>::iterator it = remove_if(keypoints.begin() + index + 1, keypoints.end(), KeypointClose(x, y, dMinDist));
		keypoints.resize(it - keypoints.begin());
		index++;
	}

	for (int i = 0; i < keypoints.size(); i++) {
		// generate keypoint descriptors
		SIFT_D sift_d;
		CalcPointDesc(img, keypoints[i], sift_d.desc);
		descriptors.push_back(sift_d);
	}

	return (int)keypoints.size();
}


/**************************
* calculate sift with maximum response around given points
* input:
*     img:	  image data
*     rect:   rect of interest
*     nGrid:  number of grid
* output:
*     keypoints:   keypoints[i] store the keypoint with maximum response around vecPoints[i] in distance < radius.
*     descriptors: descriptors of the keypoints
***************************/
int Siftor::CalcMaximumSift(const cv::Mat& img, cv::Rect rect, int nGrid, vector<cv::KeyPoint>& keypoints, vector<SIFT_D>& descriptors)
{
	if (img.type() != CV_8UC1)
		return 0;

	double xlen = max(1.0, (double)rect.width / (double)nGrid);
	double ylen = max(1.0, (double)rect.height / (double)nGrid);

	keypoints.clear();
	descriptors.clear();

	for (int kpi = 0; kpi < nGrid * nGrid; kpi++) {
		int row = kpi / nGrid;
		int col = kpi % nGrid;

		int left = max(0, rect.x + (int)(col * xlen));
		int right = min(img.cols - 1, rect.x + (int)((col + 1) * xlen) - 1);
		int top = max(0, rect.y + (int)(row * ylen));
		int bottom = min(img.rows - 1, rect.y + (int)((row + 1) * ylen) - 1);

		if (right >= rect.x + rect.width || bottom >= rect.y + rect.height)
			continue;

		double max_mag = 0.0;
		int max_ori = 0;
		cv::Point2i max_pt;

		for (int x = left; x <= right; x++) {
			for (int y = top; y <= bottom; y++) {
				double mag = 0.0;
				int ori = 0;
				CalcPointMag(img, cv::Point2i(x, y), mag, ori);
				if (mag >= max_mag) {
					max_mag = mag;
					max_ori = ori;
					max_pt = cv::Point2i(x, y);
				}
			}
		}

		cv::KeyPoint kp;
		kp.pt.x = (float)max_pt.x;
		kp.pt.y = (float)max_pt.y;
		kp.angle = (float)max_ori;
		kp.response = (float)max_mag;
		kp.class_id = 0;
		kp.octave = 0;
		kp.size = 1;
		keypoints.push_back(kp);
	}

	for (int i = 0; i < keypoints.size(); i++) {
		// generate keypoint descriptors
		SIFT_D sift_d;
		CalcPointDesc(img, keypoints[i], sift_d.desc);
		descriptors.push_back(sift_d);
	}

	return (int)keypoints.size();
}


int Siftor::CalcPointSift(const cv::Mat& img, const vector<cv::Point2i>& vecPoints, vector<cv::KeyPoint>& keypoints, vector<SIFT_D>& descriptors)
{
	if (img.type() != CV_8UC1)
		return 0;

	if (vecPoints.size() <= 0)
		return 0;

	keypoints.clear();
	descriptors.clear();

	for (int kpi = 0; kpi < vecPoints.size(); kpi++) {
		double mag = 0.0;
		int ori = 0;
		CalcPointMag(img, vecPoints[kpi], mag, ori);

		cv::KeyPoint kp;
		kp.pt.x = (float)vecPoints[kpi].x;
		kp.pt.y = (float)vecPoints[kpi].y;
		kp.angle = (float)ori;
		kp.response = (float)mag;
		kp.class_id = 0;
		kp.octave = 0;
		kp.size = 1;
		keypoints.push_back(kp);


		// generate keypoint descriptors
		SIFT_D sift_d;
		CalcPointDesc(img, kp, sift_d.desc);
		descriptors.push_back(sift_d);
	}

	return (int)keypoints.size();
}


int Siftor::CalcPointMag(const cv::Mat& img, cv::Point2i pt, double & mag, int & ori)
{
	double mag_kp[5][5];
	int    ori_kp[5][5];
	memset(mag_kp, 0, sizeof(mag_kp));
	memset(ori_kp, 0, sizeof(ori_kp));

	for (int r = 0; r < 5; r++) {
		int imgR = r - 2 + pt.y;
		if (imgR < 0 || imgR > img.rows - 2)
			continue;

		for (int c = 0; c < 5; c++) {
			int imgC = c - 2 + pt.x;
			if (imgC < 0 || imgC > img.cols - 2)
				continue;

			int cur = img.at<unsigned char>(imgR, imgC);
			int right = img.at<unsigned char>(imgR, imgC + 1);
			int bottom = img.at<unsigned char>(imgR + 1, imgC);
			mag_kp[r][c] = sqrt((right - cur) * (right - cur) + (bottom - cur) * (bottom - cur));
			ori_kp[r][c] = (int)cv::fastAtan2((float)(bottom - cur), (float)(right - cur));
		}
	}

	// count the maximum orientation and magnitude of the keypoint
	double magcounts[36], maxvm = 0.0;
	int maxvp = 0;
	CountBlockMag((double*)mag_kp, (int*)ori_kp, 25, 36, magcounts);
	for (int k = 0; k < 36; k++) {
		if (magcounts[k] > maxvm) {
			maxvm = magcounts[k];
			maxvp = k;
		}
	}

	ori = (int)((maxvp + 0.5) * 10);
	mag = maxvm;

	return 1;
}


int Siftor::CalcBlockDesc(const cv::Mat& img, cv::KeyPoint& kp, cv::Point2i left_top, double* desc)
{
	double mag_block[4][4];
	int ori_block[4][4];
	memset(mag_block, 0, sizeof(mag_block));
	memset(ori_block, 0, sizeof(ori_block));

	int main_angle = (int)kp.angle;

	for (int y = left_top.y; y < left_top.y + 4; y++){
		if (y < 0 || y > img.rows - 2)
			continue;

		for (int x = left_top.x; x < left_top.x + 4; x++) {
			if (x < 0 || x > img.cols - 2)
				continue;

			int cur = img.at<unsigned char>(y, x);
			int right = img.at<unsigned char>(y, x + 1);
			int bottom = img.at<unsigned char>(y + 1, x);
			
			mag_block[y - left_top.y][x - left_top.x] = sqrt((right - cur) * (right - cur) + (bottom - cur) * (bottom - cur));
			ori_block[y - left_top.y][x - left_top.x] = (int)(cv::fastAtan2((float)(bottom - cur), (float)(right - cur)) + 360 - main_angle) % 360;
		}
	}

	CountBlockMag((double*)mag_block, (int*)ori_block, 16, 8, desc);

	return 1;
}



int Siftor::CalcPointDesc(const cv::Mat& img, cv::KeyPoint& kp, int* desc)
{
	int x = (int)kp.pt.x;
	int y = (int)kp.pt.y;


	double mag_count[128];
	int bi = 0;

	for (int ri = 0; ri < 4; ri++) {
		for (int ci = 0; ci < 4; ci++) {
			cv::Point2i left_top(x - 7 + ci * 4, y - 7 + ri * 4);
			if (m_bRotInvariant) {
				cv::Point2f block_center(x - 7 + ci * 4 + 1, y - 7 + ri * 4 + 1);
				cv::Point2f roted_center = planimetry::rotatePoint(block_center, cv::Point2f(x, y), kp.angle);
				left_top = cv::Point2i(roted_center.x - 1, roted_center.y - 1);
			}

			CalcBlockDesc(img, kp, left_top,  mag_count + (bi * 8));
			bi++;
		}
	}


	for (int i = 0; i < 128; i++)
		desc[i] = (int)mag_count[i];

	return 1;
}



int Siftor::CalcPointSift(const cv::Mat& img, const cv::Rect rect, vector<vector<KeyPoint>>& keypoints, vector<vector<SIFT_D>>& descriptors)
{
	if (img.type() != CV_8UC1)
		return 0;

	if (rect.width <= 0 || rect.height <= 0)
		return 0;
	if (rect.x < 0 || rect.x + rect.width > img.cols || rect.y < 0 || rect.y + rect.height > img.rows)
		return 0;

	// generate descriptors for each point int the region of interest
	int kpNum = rect.width * rect.height;

	vector<KeyPoint> vecKpRow;
	vector<SIFT_D>   vecDescRow;
	vecKpRow.assign(rect.width, KeyPoint());
	vecDescRow.assign(rect.width, SIFT_D());

	keypoints.clear();
	descriptors.clear();
	keypoints.assign(rect.height, vecKpRow);
	descriptors.assign(rect.height, vecDescRow);

	for (int y = rect.y; y < rect.y + rect.height; y++) {
		for (int x = rect.x; x < rect.x + rect.width; x++) {
			double mag = 0.0;
			int ori = 0;
			CalcPointMag(img, cv::Point2i(x, y), mag, ori);

			cv::KeyPoint kp;
			kp.pt.x = (float)x;
			kp.pt.y = (float)y;
			kp.angle = (float)ori;
			kp.response = (float)mag;
			kp.class_id = 0;
			kp.octave = 0;
			kp.size = 1;

			//	keypoints.push_back(kp);
			keypoints[y - rect.y][x - rect.x] = kp;

			// generate keypoint descriptors
			SIFT_D sift_d;
			CalcPointDesc(img, kp, sift_d.desc);
			descriptors[y - rect.y][x - rect.x] = sift_d;
		}
	}

	return kpNum;
}


int Siftor::CalcRotationSift(const cv::Mat& img, const cv::Rect rect, int bins, vector<vector<SIFT_D>> descriptors[])
{
	cv::Mat srcImg = img.clone();

	vector<SIFT_D>   vecDescRow;
	vecDescRow.assign(rect.width, SIFT_D());

	cv::Point2f center(rect.x + rect.width / 2.0f, rect.y + rect.height / 2.0f);

	for (int bi = 0; bi < bins; bi++) {
		// rotate image
		double angle = 360.0 * bi / bins;   
		double radian = angle / 180.0 * CV_PI;
		cv::Mat rotImg = rotateImage(srcImg, center, angle);

		double sinv = sin(radian);
		double cosv = cos(radian);
		double cx1 = srcImg.cols / 2.0;
		double cy1 = srcImg.rows / 2.0;
		double cx2 = rotImg.cols / 2.0;
		double cy2 = rotImg.rows / 2.0;

		// assign rotated descriptors to the original points
		descriptors[bi].clear();
		descriptors[bi].assign(rect.height, vecDescRow);

		cv::Mat_<double> M = getRotationMatrix2D(center, angle, 1.0);

		for (int y = rect.y; y < rect.y + rect.height; y++) {
			for (int x = rect.x; x < rect.x + rect.width; x++) {
				int rx = M(0, 0) * x + M(0, 1) * y + M(0, 2);
				int ry = M(1, 0) * x + M(1, 1) * y + M(1, 2);

				if (rx >= 0 && rx < rotImg.cols && ry >= 0 && ry < rotImg.rows) {
					double mag = 0.0;
					int ori = 0;
					CalcPointMag(rotImg, cv::Point2i(rx, ry), mag, ori);
				//	CalcPointMag2(rotImg, cv::Point2i(rx, ry), mag, ori);

					cv::KeyPoint kp;
					kp.pt.x = (float)rx;
					kp.pt.y = (float)ry;
					kp.angle = (float)ori;
					kp.response = (float)mag;
					kp.class_id = 0;
					kp.octave = 0;
					kp.size = 1;

					SIFT_D sift_d;
					CalcPointDesc(rotImg, kp, sift_d.desc);
				//	CalcPointDesc2(rotImg, kp, sift_d.desc);

					descriptors[bi][y - rect.y][x - rect.x] = sift_d;
				}
				else {
					descriptors[bi][y - rect.y][x - rect.x] = descriptors[0][y - rect.y][x - rect.x];
				}
			}
		}
	
	}

	return 1;
}

cv::Mat Siftor::rotateImage(cv::Mat& inputImg, cv::Point2f center, double angle)
{

	double radian = angle / 180.0 * CV_PI;

	//Ìî³äÍ¼ÏñÊ¹Æä·ûºÏÐý×ªÒªÇó
	int uniSize = (int)(max(inputImg.cols, inputImg.rows)* 1.414);
	int dx = (int)(uniSize - inputImg.cols) / 2;
	int dy = (int)(uniSize - inputImg.rows) / 2;

//	cv::Mat  tempImg;
//	cv::copyMakeBorder(inputImg, tempImg, dy, dy, dx, dx, BORDER_CONSTANT);
//	imwrite("img_border.jpg", tempImg);

	//ÐýÞD¾ØÕó
	cv::Mat affine_matrix = getRotationMatrix2D(center, angle, 1.0);

	//ÐýÞD
	cv::Mat  tempImg;
	warpAffine(inputImg, tempImg, affine_matrix, inputImg.size());

	return tempImg;
}
