#include "Brisker.h"


const float Brisker::basicSize_ = 12.0;
const unsigned int Brisker::scales_ = 64;
const float Brisker::scalerange_ = 30;        // 40->4 Octaves - else, this needs to be adjusted...
const unsigned int Brisker::n_rot_ = 1024;	 // discretization of the rotation look-up

//const float BriskScaleSpace::safetyFactor_ = 1.0;
//const float BriskScaleSpace::basicSize_ = 12.0;


Brisker::Brisker(bool rotationInvariant, bool scaleInvariant, float patternScale)
{
	std::vector<float> rList;
	std::vector<int> nList;

	// this is the standard pattern found to be suitable also
	rList.resize(5);
	nList.resize(5);
	const double f = 0.85*patternScale;

	rList[0] = f * 0;
	rList[1] = f*2.9;
	rList[2] = f*4.9;
	rList[3] = f*7.4;
	rList[4] = f*10.8;

	nList[0] = 1;
	nList[1] = 10;
	nList[2] = 14;
	nList[3] = 15;
	nList[4] = 20;

	rotationInvariance = rotationInvariant;
	scaleInvariance = scaleInvariant;
	generateKernel(rList, nList, 5.85*patternScale, 8.2*patternScale);

}


Brisker::~Brisker()
{
	delete[] patternPoints_;
	delete[] shortPairs_;
	delete[] longPairs_;
	delete[] scaleList_;
	delete[] sizeList_;
}

bool Brisker::RoiPredicate(const float minX, const float minY,
	const float maxX, const float maxY, const KeyPoint& keyPt)
{
	const Point2f& pt = keyPt.pt;
	return (pt.x < minX) || (pt.x >= maxX) || (pt.y < minY) || (pt.y >= maxY);
}


void Brisker::generateKernel(std::vector<float> &radiusList, std::vector<int> &numberList, 
	float dMax, float dMin,	std::vector<int> indexChange)
{

	dMax_ = dMax;
	dMin_ = dMin;

	// get the total number of points
	const int rings = radiusList.size();
	assert(radiusList.size() != 0 && radiusList.size() == numberList.size());
	points_ = 0; // remember the total number of points
	for (int ring = 0; ring<rings; ring++){
		points_ += numberList[ring];
	}
	// set up the patterns
	patternPoints_ = new BriskPatternPoint[points_*scales_*n_rot_];
	BriskPatternPoint* patternIterator = patternPoints_;

	// define the scale discretization:
	static const float lb_scale = log(scalerange_) / log(2.0);
	static const float lb_scale_step = lb_scale / (scales_);

	scaleList_ = new float[scales_];
	sizeList_ = new unsigned int[scales_];

	const float sigma_scale = 1.3;

	for (unsigned int scale = 0; scale <scales_; ++scale){
		scaleList_[scale] = pow((double)2.0, (double)(scale*lb_scale_step));
		sizeList_[scale] = 0;

		// generate the pattern points look-up
		double alpha, theta;
		for (size_t rot = 0; rot<n_rot_; ++rot){
			theta = double(rot) * 2 * CV_PI / double(n_rot_); // this is the rotation of the feature
			for (int ring = 0; ring<rings; ++ring){
				for (int num = 0; num<numberList[ring]; ++num){
					// the actual coordinates on the circle
					alpha = (double(num)) * 2 * CV_PI / double(numberList[ring]);
					patternIterator->x = scaleList_[scale] * radiusList[ring] * cos(alpha + theta); // feature rotation plus angle of the point
					patternIterator->y = scaleList_[scale] * radiusList[ring] * sin(alpha + theta);
					// and the gaussian kernel sigma
					if (ring == 0){
						patternIterator->sigma = sigma_scale*scaleList_[scale] * 0.5;
					}
					else{
						patternIterator->sigma = sigma_scale*scaleList_[scale] * (double(radiusList[ring]))*sin(CV_PI / numberList[ring]);
					}
					// adapt the sizeList if necessary
					const unsigned int size = ceil(((scaleList_[scale] * radiusList[ring]) + patternIterator->sigma)) + 1;
					if (sizeList_[scale]<size){
						sizeList_[scale] = size;
					}

					// increment the iterator
					++patternIterator;
				}
			}
		}
	}

	// now also generate pairings
	shortPairs_ = new BriskShortPair[points_*(points_ - 1) / 2];
	longPairs_ = new BriskLongPair[points_*(points_ - 1) / 2];
	noShortPairs_ = 0;
	noLongPairs_ = 0;

	// fill indexChange with 0..n if empty
	unsigned int indSize = indexChange.size();
	if (indSize == 0) {
		indexChange.resize(points_*(points_ - 1) / 2);
		indSize = indexChange.size();
	}
	for (unsigned int i = 0; i<indSize; i++){
		indexChange[i] = i;
	}
	const float dMin_sq = dMin_*dMin_;
	const float dMax_sq = dMax_*dMax_;
	for (unsigned int i = 1; i<points_; i++){
		for (unsigned int j = 0; j<i; j++){ //(find all the pairs)
			// point pair distance:
			const float dx = patternPoints_[j].x - patternPoints_[i].x;
			const float dy = patternPoints_[j].y - patternPoints_[i].y;
			const float norm_sq = (dx*dx + dy*dy);
			if (norm_sq>dMin_sq){
				// save to long pairs
				BriskLongPair& longPair = longPairs_[noLongPairs_];
				longPair.weighted_dx = int((dx / (norm_sq))*2048.0 + 0.5);
				longPair.weighted_dy = int((dy / (norm_sq))*2048.0 + 0.5);
				longPair.i = i;
				longPair.j = j;
				++noLongPairs_;
			}
			else if (norm_sq<dMax_sq){
				// save to short pairs
				assert(noShortPairs_<indSize); // make sure the user passes something sensible
				BriskShortPair& shortPair = shortPairs_[indexChange[noShortPairs_]];
				shortPair.j = j;
				shortPair.i = i;
				++noShortPairs_;
			}
		}
	}

	// no bits:
	strings_ = (int)ceil((float(noShortPairs_)) / 128.0) * 4 * 4;
}


// simple alternative:
inline int Brisker::smoothedIntensity(const cv::Mat& image,
	const cv::Mat& integral, const float key_x,
	const float key_y, const unsigned int scale,
	const unsigned int rot, const unsigned int point) const{

	// get the float position
	const BriskPatternPoint& briskPoint = patternPoints_[scale*n_rot_*points_ + rot*points_ + point];
	const float xf = briskPoint.x + key_x;
	const float yf = briskPoint.y + key_y;
	const int x = int(xf);
	const int y = int(yf);
	const int& imagecols = image.cols;

	// get the sigma:
	const float sigma_half = briskPoint.sigma;
	const float area = 4.0*sigma_half*sigma_half;

	// calculate output:
	int ret_val;
	if (sigma_half<0.5){
		//interpolation multipliers:
		const int r_x = (xf - x) * 1024;
		const int r_y = (yf - y) * 1024;
		const int r_x_1 = (1024 - r_x);
		const int r_y_1 = (1024 - r_y);
		uchar* ptr = image.data + x + y*imagecols;
		// just interpolate:
		ret_val = (r_x_1*r_y_1*int(*ptr));
		ptr++;
		ret_val += (r_x*r_y_1*int(*ptr));
		ptr += imagecols;
		ret_val += (r_x*r_y*int(*ptr));
		ptr--;
		ret_val += (r_x_1*r_y*int(*ptr));
		return (ret_val + 512) / 1024;
	}

	// this is the standard case (simple, not speed optimized yet):

	// scaling:
	const int scaling = 4194304.0 / area;
	const int scaling2 = float(scaling)*area / 1024.0;

	// the integral image is larger:
	const int integralcols = imagecols + 1;

	// calculate borders
	const float x_1 = xf - sigma_half;
	const float x1 = xf + sigma_half;
	const float y_1 = yf - sigma_half;
	const float y1 = yf + sigma_half;

	const int x_left = int(x_1 + 0.5);
	const int y_top = int(y_1 + 0.5);
	const int x_right = int(x1 + 0.5);
	const int y_bottom = int(y1 + 0.5);

	// overlap area - multiplication factors:
	const float r_x_1 = float(x_left) - x_1 + 0.5;
	const float r_y_1 = float(y_top) - y_1 + 0.5;
	const float r_x1 = x1 - float(x_right) + 0.5;
	const float r_y1 = y1 - float(y_bottom) + 0.5;
	const int dx = x_right - x_left - 1;
	const int dy = y_bottom - y_top - 1;
	const int A = (r_x_1*r_y_1)*scaling;
	const int B = (r_x1*r_y_1)*scaling;
	const int C = (r_x1*r_y1)*scaling;
	const int D = (r_x_1*r_y1)*scaling;
	const int r_x_1_i = r_x_1*scaling;
	const int r_y_1_i = r_y_1*scaling;
	const int r_x1_i = r_x1*scaling;
	const int r_y1_i = r_y1*scaling;

	if (dx + dy>2){
		// now the calculation:
		uchar* ptr = image.data + x_left + imagecols*y_top;
		// first the corners:
		ret_val = A*int(*ptr);
		ptr += dx + 1;
		ret_val += B*int(*ptr);
		ptr += dy*imagecols + 1;
		ret_val += C*int(*ptr);
		ptr -= dx + 1;
		ret_val += D*int(*ptr);

		// next the edges:
		int* ptr_integral = (int*)integral.data + x_left + integralcols*y_top + 1;
		// find a simple path through the different surface corners
		const int tmp1 = (*ptr_integral);
		ptr_integral += dx;
		const int tmp2 = (*ptr_integral);
		ptr_integral += integralcols;
		const int tmp3 = (*ptr_integral);
		ptr_integral++;
		const int tmp4 = (*ptr_integral);
		ptr_integral += dy*integralcols;
		const int tmp5 = (*ptr_integral);
		ptr_integral--;
		const int tmp6 = (*ptr_integral);
		ptr_integral += integralcols;
		const int tmp7 = (*ptr_integral);
		ptr_integral -= dx;
		const int tmp8 = (*ptr_integral);
		ptr_integral -= integralcols;
		const int tmp9 = (*ptr_integral);
		ptr_integral--;
		const int tmp10 = (*ptr_integral);
		ptr_integral -= dy*integralcols;
		const int tmp11 = (*ptr_integral);
		ptr_integral++;
		const int tmp12 = (*ptr_integral);

		// assign the weighted surface integrals:
		const int upper = (tmp3 - tmp2 + tmp1 - tmp12)*r_y_1_i;
		const int middle = (tmp6 - tmp3 + tmp12 - tmp9)*scaling;
		const int left = (tmp9 - tmp12 + tmp11 - tmp10)*r_x_1_i;
		const int right = (tmp5 - tmp4 + tmp3 - tmp6)*r_x1_i;
		const int bottom = (tmp7 - tmp6 + tmp9 - tmp8)*r_y1_i;

		return (ret_val + upper + middle + left + right + bottom + scaling2 / 2) / scaling2;
	}

	// now the calculation:
	uchar* ptr = image.data + x_left + imagecols*y_top;
	// first row:
	ret_val = A*int(*ptr);
	ptr++;
	const uchar* end1 = ptr + dx;
	for (; ptr<end1; ptr++){
		ret_val += r_y_1_i*int(*ptr);
	}
	ret_val += B*int(*ptr);
	// middle ones:
	ptr += imagecols - dx - 1;
	uchar* end_j = ptr + dy*imagecols;
	for (; ptr<end_j; ptr += imagecols - dx - 1){
		ret_val += r_x_1_i*int(*ptr);
		ptr++;
		const uchar* end2 = ptr + dx;
		for (; ptr<end2; ptr++){
			ret_val += int(*ptr)*scaling;
		}
		ret_val += r_x1_i*int(*ptr);
	}
	// last row:
	ret_val += D*int(*ptr);
	ptr++;
	const uchar* end3 = ptr + dx;
	for (; ptr<end3; ptr++){
		ret_val += r_y1_i*int(*ptr);
	}
	ret_val += C*int(*ptr);

	return (ret_val + scaling2 / 2) / scaling2;
}

void Brisker::countBlockMag(double* mag, int* ori, int len, int div, double* magcounts)
{
	for (int i = 0; i < div; i++)
		magcounts[i] = 0;

	for (int i = 0; i < len; i++) {
		int ind = div * ori[i] / 360;
		magcounts[ind] += mag[i];
	}
}

void Brisker::calcPointDesc(const Mat& image, const Mat& _integral, cv::KeyPoint& kp, uchar desc[])
{
	static const float log2 = 0.693147180559945;
	static const float lb_scalerange = log(scalerange_) / (log2);
	static const float basicSize06 = basicSize_*0.6;
	unsigned int basicscale = 0;
	unsigned int scale = 0;
	if (scaleInvariance) {
		scale = std::max((int)(scales_ / lb_scalerange*(log(kp.size / (basicSize06)) / log2) + 0.5), 0);
		// saturate
		if (scale >= scales_) 
			scale = scales_ - 1;
	}
	else{
		basicscale = std::max((int)(scales_ / lb_scalerange*(log(1.45*basicSize_ / (basicSize06)) / log2) + 0.5), 0);
		scale = basicscale;
	}

	// return if the point is in the border of the image
	const int border = sizeList_[scale];
	const int border_x = image.cols - border;
	const int border_y = image.rows - border;
	if (RoiPredicate(border, border, border_x, border_y, kp))
		return;


	uchar* ptr = desc;
	int theta = 0;

	int* _values = new int[points_]; // for temporary use
	int* pvalues = _values;


	// temporary variables containing gray values at sample points:
	int t1 = 0, t2 = 0;	
	int direction0 = 0, direction1 = 0;		// the feature orientation

	const float& x = kp.pt.x;
	const float& y = kp.pt.y;

	if (rotationInvariance)
	{
		// get the gray values in the unrotated pattern
		for (unsigned int i = 0; i< points_; i++){
			*(pvalues++) = smoothedIntensity(image, _integral, x, y, scale, 0, i);
		}

		direction0 = 0;
		direction1 = 0;
		// now iterate through the long pairings
		const BriskLongPair* max = longPairs_ + noLongPairs_;
		for (BriskLongPair* iter = longPairs_; iter<max; ++iter)
		{
			t1 = *(_values + iter->i);
			t2 = *(_values + iter->j);
			const int delta_t = (t1 - t2);
		
			// update the direction:
			const int tmp0 = delta_t*(iter->weighted_dx) / 1024;
			const int tmp1 = delta_t*(iter->weighted_dy) / 1024;
			direction0 += tmp0;
			direction1 += tmp1;
		}
		
		kp.angle = atan2((float)direction1, (float)direction0) / CV_PI*180.0;
		theta = int((n_rot_*kp.angle) / (360.0) + 0.5);
		if (theta<0)
			theta += n_rot_;
		if (theta >= int(n_rot_))
			theta -= n_rot_;
	}

	// now also extract the stuff for the actual direction:
	// let us compute the smoothed values
	int shifter = 0;

	//unsigned int mean=0;
	pvalues = _values;
	// get the gray values in the rotated pattern
	for (unsigned int i = 0; i<points_; i++){
		*(pvalues++) = smoothedIntensity(image, _integral, x,
			y, scale, theta, i);
	}

	// now iterate through all the pairings
	UINT32_ALIAS* ptr2 = (UINT32_ALIAS*)ptr;
	const BriskShortPair* max = shortPairs_ + noShortPairs_;
	for (BriskShortPair* iter = shortPairs_; iter<max; ++iter){
		t1 = *(_values + iter->i);
		t2 = *(_values + iter->j);
		if (t1>t2){
			*ptr2 |= ((1) << shifter);

		} // else already initialized with zero
		// take care of the iterators:
		++shifter;
		if (shifter == 32){
			shifter = 0;
			++ptr2;
		}
	}
	
	delete[] _values;
}

void Brisker::calcPointMag(const cv::Mat& img, cv::Point2i pt, double & mag, int & ori)
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
	countBlockMag((double*)mag_kp, (int*)ori_kp, 25, 36, magcounts);
	for (int k = 0; k < 36; k++) {
		if (magcounts[k] > maxvm) {
			maxvm = magcounts[k];
			maxvp = k;
		}
	}

	ori = (int)((maxvp + 0.5) * 10);
	mag = maxvm;

}

int Brisker::calcPointBrisk(const Mat& image, const vector<cv::Point2i>& vecPoints, std::vector<KeyPoint>& keypoints,
	std::vector<BRISK_D>& descriptors)
{
	if (image.type() != CV_8UC1)
		return 0;

	keypoints.clear();
	descriptors.clear();

	// first, calculate the integral image over the whole image:
	// current integral image
	cv::Mat _integral; // the integral image
	cv::integral(image, _integral);

	for (int i = 0; i < vecPoints.size(); i++) {
		double mag = 0.0;
		int ori = 0;
		calcPointMag(image, vecPoints[i], mag, ori);

		cv::KeyPoint kp;
		kp.pt.x = (float)vecPoints[i].x;
		kp.pt.y = (float)vecPoints[i].y;
		kp.angle = (float)ori;
		kp.response = (float)mag;
		kp.class_id = 0;
		kp.octave = 0;
		kp.size = 1;

		keypoints.push_back(kp);

		BRISK_D brisk_d;
		memset(brisk_d.desc, 0, sizeof(brisk_d.desc));
		calcPointDesc(image, _integral, kp, brisk_d.desc);
		descriptors.push_back(brisk_d);
	}

	return 1;
}


int Brisker::calcPointBrisk(const cv::Mat& image, const cv::Rect rect, vector<vector<cv::KeyPoint>>& keypoints, vector<vector<BRISK_D>>& descriptors)
{
	if (image.type() != CV_8UC1)
		return 0;

	if (rect.width <= 0 || rect.height <= 0)
		return 0;
	if (rect.x < 0 || rect.x + rect.width > image.cols || rect.y < 0 || rect.y + rect.height > image.rows)
		return 0;

	vector<cv::KeyPoint> vecKpRow;
	vector<BRISK_D>   vecDescRow;
	vecKpRow.assign(rect.width, cv::KeyPoint());
	vecDescRow.assign(rect.width, BRISK_D());

	keypoints.clear();
	descriptors.clear();
	keypoints.assign(rect.height, vecKpRow);
	descriptors.assign(rect.height, vecDescRow);

	// first, calculate the integral image over the whole image:
	// current integral image
	cv::Mat _integral; // the integral image
	cv::integral(image, _integral);


	for (int y = rect.y; y < rect.y + rect.height; y++) {
		for (int x = rect.x; x < rect.x + rect.width; x++) {

			double mag = 0.0;
			int ori = 0;
			calcPointMag(image, cv::Point2i(x, y), mag, ori);

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
			BRISK_D brisk_d;
			memset(brisk_d.desc, 0, sizeof(brisk_d.desc));
			calcPointDesc(image, _integral, kp, brisk_d.desc);

			descriptors[y - rect.y][x - rect.x] = brisk_d;
		}
	}

	return 1;
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
int Brisker::CalcMaximumBrisk(const cv::Mat& img, cv::Rect rect, int nGrid, vector<cv::KeyPoint>& keypoints, vector<BRISK_D>& descriptors)
{
	if (img.type() != CV_8UC1)
		return 0;

	// first, calculate the integral image over the whole image:
	// current integral image
	cv::Mat _integral; // the integral image
	cv::integral(img, _integral);


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
				calcPointMag(img, cv::Point2i(x, y), mag, ori);
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
		BRISK_D brisk_d;
		memset(brisk_d.desc, 0, sizeof(brisk_d.desc));
		calcPointDesc(img, _integral, keypoints[i], brisk_d.desc);
		descriptors.push_back(brisk_d);
	}

	return (int)keypoints.size();
}
