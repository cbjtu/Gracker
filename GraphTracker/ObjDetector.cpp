
#include "ObjDetector.h"
#include "Utils.h"

ObjDetector::ObjDetector()
{
	m_pFeatureDetector = new SiftFeatureDetector();
	m_pDescriptorExtractor = new SiftDescriptorExtractor();

	m_hogWindow = cvSize(128, 128);
	m_pHogExtractor = new HOGDescriptor(m_hogWindow, cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 9);
}


ObjDetector::~ObjDetector()
{
	delete m_pFeatureDetector;
	delete m_pDescriptorExtractor;
	delete m_pHogExtractor;
}


bool ObjDetector::setModel(const cv::Mat& model)
{
	if (model.type() != CV_8UC1)
		return false;

	m_imgModel.release();
	m_imgModel = model.clone();

	m_vecModelPoint.clear();
	m_matModelDesc.release();
	
	m_pFeatureDetector->detect(model, m_vecModelPoint);

	sort(m_vecModelPoint.begin(), m_vecModelPoint.end(), CompareKeypoints);

	int index = 0;
	double dMinDist = (double)m_nMinKeypointDist;
	while (index < m_vecModelPoint.size()) {
		double x = m_vecModelPoint[index].pt.x;
		double y = m_vecModelPoint[index].pt.y;
		vector<cv::KeyPoint>::iterator it = remove_if(m_vecModelPoint.begin() + index + 1, m_vecModelPoint.end(), KeypointClose(x, y, dMinDist));
		m_vecModelPoint.resize(it - m_vecModelPoint.begin());
		index++;
	}

	int num = min((int)m_vecModelPoint.size(), m_nMaxModelPoints);
	m_vecModelPoint.resize(num);

	m_pDescriptorExtractor->compute(model, m_vecModelPoint, m_matModelDesc);


	return true;
}


cv::Rect ObjDetector::boxing(const cv::Mat& frame)
{
	int W = m_imgModel.cols;
	int H = m_imgModel.rows;

	double max_sim = 0.0;
	cv::Rect max_rect;

	cv::Mat model;
	cv::resize(m_imgModel, model, m_hogWindow);
//	vector<float> model_desc;
//	m_pHogExtractor->compute(model, model_desc, cvSize(1, 1), cvSize(0, 0));
	vector<int> model_desc;
	calcColorHist(model, model_desc);

	cv::Mat fea1(model_desc);

	double scale = 0.25;
	while (scale <= 2.0) {
		int width = W * scale;
		int height = H * scale;
		int stride = max(8, max(width, height) / 8);

		for (int y = 0; y < frame.rows - height; y += stride) {
			for (int x = 0; x < frame.cols - width; x += stride) {
				cv::Mat sample;
				cv::Rect rect(x, y, width, height);
				cv::resize(frame(rect), sample, m_hogWindow);
			//	vector<float> descriptor;
			//	m_pHogExtractor->compute(sample, descriptor, cvSize(1, 1), cvSize(0, 0));
				vector<int> descriptor;
				calcColorHist(sample, descriptor);

				cv::Mat fea2(descriptor);

				double sim_app = fea1.dot(fea2) / sqrt(fea1.dot(fea1) * fea2.dot(fea2));
				double sim_area = max(1.0, (double)(rect.width * rect.height) / (double)(m_imgModel.rows * m_imgModel.cols));
				double sim = sim_app + sim_area;
				if (sim > max_sim) {
					max_sim = sim;
					max_rect = rect;
				}
			}
		}

		scale *= 2;
	}

	return max_rect;
}

void ObjDetector::calcColorHist(cv::Mat& img, vector<int>& hist)
{
	int nBlock = 8;
	int nBin = 8;

	hist.clear();
	hist.assign(nBlock * nBlock * nBin, 0);

	int W = img.cols;
	int H = img.rows;
	int blockW = W / nBlock;
	int blockH = H / nBlock;

	for (int row = 0; row < nBlock; row++) {
		for (int col = 0; col < nBlock; col++) {
			int block_ind = row * nBlock + col;
			int hist_ind = block_ind * nBin;
			for (int y = row * blockH; y < (row + 1) * blockH; y++) {
				for (int x = col * blockW; x < (col + 1) * blockW; x++) {
					unsigned char grey = img.at<unsigned char>(y, x);
					hist[hist_ind + nBin * grey / 256]++;
				}
			}
		}
	}
}

bool ObjDetector::templateMatch(const cv::Mat& frame, Warpping& w)
{
	int result_cols = frame.cols - m_imgModel.cols + 1;
	int result_rows = frame.rows - m_imgModel.rows + 1;
	cv::Mat result = cv::Mat::zeros(result_cols, result_rows, CV_32FC1);

	int match_method = CV_TM_SQDIFF_NORMED;
	matchTemplate(frame, m_imgModel, result, match_method);

	// 找到最佳匹配位置  
	double minVal;
	double maxVal;
	Point minLoc;
	Point maxLoc;
	Point matchLoc;
	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());   // 寻找result中的最大和最小值，以及它们所处的像素位置  

	// 使用SQDIFF和SQDIFF_NORMED方法时：值越小代表越相似  
	// 使用其他方法时：值越大代表越相似  
	if (match_method == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED)
	{
		matchLoc = minLoc;
	}
	else
	{
		matchLoc = maxLoc;
	}

	w.initialize(matchLoc.x, matchLoc.y);

	return true;
}

bool ObjDetector::detect(const cv::Mat& frame, Warpping& w)
{
	if (frame.type() != CV_8UC1)
		return false;

	if (m_vecModelPoint.size() <= 0)
		return false;

	if (m_vecModelPoint.size() <= 20)
		return templateMatch(frame, w);

//	m_objRect = boxing(frame);
//	cv::Mat mask = cv::Mat_<unsigned char>::zeros(frame.size());
//	mask(m_objRect) = 1;

	// extract keypoints
	vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;
//	m_pFeatureDetector->detect(frame, keypoints, mask);
	m_pFeatureDetector->detect(frame, keypoints);

	sort(keypoints.begin(), keypoints.end(), CompareKeypoints);

	int index = 0;
	double dMinDist = (double)m_nMinKeypointDist;
	while (index < keypoints.size()) {
		double x = keypoints[index].pt.x;
		double y = keypoints[index].pt.y;
		vector<cv::KeyPoint>::iterator it = remove_if(keypoints.begin() + index + 1, keypoints.end(), KeypointClose(x, y, dMinDist));
		keypoints.resize(it - keypoints.begin());
		index++;
	}


	m_pDescriptorExtractor->compute(frame, keypoints, descriptors);

	// generate affinity matrix
	vector<MatchInfo> matches;
	cv::Mat_<double> affinity, group1, group2;
	genAffinityMatrix(keypoints, descriptors, matches, group1, group2, affinity);

	// graph matching 
	GMA gma;
	cv::Mat_<double> x;

	if (matches.size() > m_vecModelPoint.size()) {
		//	x = gma.PSM(affinity, matches, group1, group2);
		//	x = gma.SM(affinity);
		x = gma.IPFP(affinity, matches, group1, group2);
		//  x = gma.GNCCP_APE(affinity, matches, group1, group2);
		x = gma.greedyMapping(x, matches);
	}
	
	// compute object warping function
	vector<Point2f> obj;
	vector<Point2f> scene;
	for (int i = 0; i < x.rows; i++){
		if (x(i, 0) > 0.0) {
			obj.push_back(m_vecModelPoint[matches[i].idx_g1].pt);
			scene.push_back(keypoints[matches[i].idx_g2].pt);
		}
	}


//	vector<Point2f> inliersObj(obj);
//	vector<Point2f> inliersScene(scene);
//	aff = computeAffine(inliersObj, inliersScene);

//	drawDebugImage(frame, aff, obj, scene, inliersObj, inliersScene);


	if (obj.size() < 10)
		return false;


	int num = (int)obj.size();
	for (int iter = 0; iter < num / 2; iter++) {
		w.computeWarpping(obj, scene);

		vector<cv::Point2f> warpPts = w.doWarpping(obj);

		// repeatedly remove the match with maximum error
		vector<double> errs(obj.size(), 0.0);
		double max_err = 0.0;
		int max_idx = 0;
		for (int i = 0; i < obj.size(); i++) {
			double dx = scene[i].x - warpPts[i].x;
			double dy = scene[i].y - warpPts[i].y;
			errs[i] = sqrt(dx * dx + dy * dy);

			if (errs[i] > max_err) {
				max_err = errs[i];
				max_idx = i;
			}
		}

		obj.erase(obj.begin() + max_idx);
		scene.erase(scene.begin() + max_idx);
	}

	return true;

}


/*
AffineT ObjDetector::computeAffine(vector<cv::Point2f>& src, vector<cv::Point2f>& dst)
{
	AffineT affine;
	affine.computeAffine(src, dst);

	int num = (int)src.size();
	for (int iter = 0; iter < num / 2; iter++) {
		// repeatedly remove the match with maximum error
		vector<double> errs(num, 0.0);
		double max_err = 0.0;
		int max_idx = 0;
		for (int i = 0; i < src.size(); i++) {
			cv::Point2f pt = affine.doAffine(src[i]);
			double dx = dst[i].x - pt.x;
			double dy = dst[i].y - pt.y;
			errs[i] = sqrt(dx * dx + dy * dy);

			if (errs[i] > max_err) {
				max_err = errs[i];
				max_idx = i;
			}
		}

		src.erase(src.begin() + max_idx);
		dst.erase(dst.begin() + max_idx);

		// then recompute the affine transformation
		affine.computeAffine(src, dst);
	}

	return affine;

}
*/

/*
void ObjDetector::drawDebugImage(const cv::Mat& frame, AffineT& aff, vector<Point2f>& obj, vector<Point2f>& scene, vector<Point2f>& inliersObj, vector<Point2f>& inliersScene)
{
	cv::Mat outImg;
	vector<KeyPoint> keypoints1, keypoints2;
	vector<cv::DMatch> matches;
	cv::drawMatches(m_imgModel, keypoints1, frame, keypoints2, matches, outImg);


	// draw the estimated bounding box
	cv::Point2f opt1(m_imgModel.cols + m_objRect.x, m_objRect.y);
	cv::Point2f opt2(m_imgModel.cols + m_objRect.x + m_objRect.width, m_objRect.y);
	cv::Point2f opt3(m_imgModel.cols + m_objRect.x + m_objRect.width, m_objRect.y + m_objRect.height);
	cv::Point2f opt4(m_imgModel.cols + m_objRect.x, m_objRect.y + m_objRect.height);
	cv::Scalar red(0, 0, 255);
	cv::line(outImg, opt1, opt2, red, 2);
	cv::line(outImg, opt2, opt3, red, 2);
	cv::line(outImg, opt3, opt4, red, 2);
	cv::line(outImg, opt4, opt1, red, 2);



	cv::Scalar yellow(0, 255, 255);
	for (int i = 0; i < obj.size(); i++) {
		cv::Point2f pt1 = obj[i];
		cv::Point2f pt2 = scene[i];
		pt2.x += m_imgModel.cols;

		cv::line(outImg, pt1, pt2, yellow, 2);
	}

	cv::Scalar blue(255, 0, 0);
	for (int i = 0; i < inliersObj.size(); i++) {
		cv::Point2f pt1 = inliersObj[i];
		cv::Point2f pt2 = inliersScene[i];
		pt2.x += m_imgModel.cols;

		cv::line(outImg, pt1, pt2, blue, 2);
	}


	float w = m_imgModel.cols;
	float h = m_imgModel.rows;

	vector<Point2f> corners;
	Point2f pt1 = aff.doAffine(Point2f(0.f, 0.f));
	Point2f pt2 = aff.doAffine(Point2f(w, 0.f));
	Point2f pt3 = aff.doAffine(Point2f(w, h));
	Point2f pt4 = aff.doAffine(Point2f(0.f, h));

	pt1.x += w;
	pt2.x += w;
	pt3.x += w;
	pt4.x += w;

	cv::Scalar green(0, 255, 0);
	line(outImg, pt1, pt2, green, 2);
	line(outImg, pt2, pt3, green, 2);
	line(outImg, pt3, pt4, green, 2);
	line(outImg, pt4, pt1, green, 2);

	imshow("object_detector", outImg);
}
*/

int ObjDetector::genCandidateSet(const vector<cv::KeyPoint>& keypoints, const cv::Mat& descriptors, 
	vector<MatchInfo>& matches, cv::Mat_<double>& group1, cv::Mat_<double>& group2)
{
	vector<int> usedIndex;	// record indexs of points used to form candidate matches 
	int usedCount = 0;
	usedIndex.assign(keypoints.size(), -1);

	matches.clear();
	for (int i = 0; i < m_vecModelPoint.size(); i++){
		cv::Mat fea1 = m_matModelDesc.row(i).clone();

		// filter points under appearance constraints
		vector<cv::KeyPoint> candidates;

		for (int k = 0; k < keypoints.size(); k++) {
			cv::Mat fea2 = descriptors.row(k).clone();

			double sim = fea1.dot(fea2) / sqrt(fea1.dot(fea1) * fea2.dot(fea2));

			if (sim >= m_dAppTol) {
				cv::KeyPoint kp = keypoints[k];
				kp.class_id = k;
				kp.response = (float)sim;
				candidates.push_back(kp);
			}

		}
		
		// select points with maximum similarity
		sort(candidates.begin(), candidates.end(), CompareKeypoints);
		int size = min((int)candidates.size(), m_nMaxCandidates);
		candidates.resize(size);
		for (int j = 0; j < candidates.size(); j++) {
			MatchInfo info;
			info.idx_g1 = i;
			info.idx_g2 = candidates[j].class_id;
			info.idx_match = (int)matches.size();
			info.score = candidates[j].response;
			matches.push_back(info);

			if (usedIndex[info.idx_g2] < 0) {
				usedIndex[info.idx_g2] = usedCount;
				usedCount++;
			}
		}
	}

	// generate groups information
	group1 = cv::Mat_<double>::zeros((int)matches.size(), (int)m_vecModelPoint.size());
	group2 = cv::Mat_<double>::zeros((int)matches.size(), usedCount);
	for (int i = 0; i < matches.size(); i++) {
		group1.at<double>(i, matches[i].idx_g1) = 1.0;
		group2.at<double>(i, usedIndex[matches[i].idx_g2]) = 1.0;
	}

	return 1;
}

int ObjDetector::genAffinityMatrix(const vector<cv::KeyPoint>& keypoints, const cv::Mat& descriptors, 
	vector<MatchInfo>& matches, cv::Mat_<double>& group1, cv::Mat_<double>& group2, cv::Mat_<double>& affinity)
{
	genCandidateSet(keypoints, descriptors, matches, group1, group2);

	if (matches.size() <= 0)
		return 0;

	affinity = cv::Mat_<double>::zeros((int)matches.size(), (int)matches.size());
	for (int i = 0; i < matches.size(); i++) {
		int p1 = matches[i].idx_g1;
		int p2 = matches[i].idx_g2;

		double sim_p = matches[i].score;

		double px1 = m_vecModelPoint[p1].pt.x;
		double py1 = m_vecModelPoint[p1].pt.y;
		double px2 = keypoints[p2].pt.x;
		double py2 = keypoints[p2].pt.y;

		for (int k = 0; k < matches.size(); k++) {
			int q1 = matches[k].idx_g1;
			int q2 = matches[k].idx_g2;

			double sim_q = matches[k].score;

			double qx1 = m_vecModelPoint[q1].pt.x;
			double qy1 = m_vecModelPoint[q1].pt.y;
			double qx2 = keypoints[q2].pt.x;
			double qy2 = keypoints[q2].pt.y;


			if (p1 == q1 && p2 == q2) {		// node affinity
				affinity(i, k) = sim_p;
			}
			else if (p1 != q1 && p2 != q2) {	// edge affinity
				double dist1 = sqrt((px1 - qx1) * (px1 - qx1) + (py1 - qy1) * (py1 - qy1));
				double dist2 = sqrt((px2 - qx2) * (px2 - qx2) + (py2 - qy2) * (py2 - qy2));

				double dx = abs((qx1 - px1) - (qx2 - px2));
				double dy = abs((qy1 - py1) - (qy2 - py2));
				double dist3 = sqrt((dx * dx) + (dy * dy));

				affinity(i, k) = m_dGeoWeight * (max(0.0, 1.0 - dist3 / (max(dist1, dist2))));

			//	affinity(i, k) = m_dGeoWeight * (1.0 - abs(dist1 - dist2) / (dist1 + dist2));
				affinity(i, k) = affinity(i, k) + (sim_p + sim_q) / 2;
			}
			else {
				affinity(i, k) = 0.0;
			}

		}
	}

	return matches.size();
}


/*
int ObjDetector::genCandidateSet2(const vector<cv::KeyPoint>& keypoints, const vector<DescType>& descriptors,
	vector<MatchInfo>& matches, cv::Mat_<double>& group1, cv::Mat_<double>& group2)
{
	const vector<cv::KeyPoint>& vecModelPoint = m_pModel->getModelPoint();
	const vector<DescType>& vecModelDesc = m_pModel->getModelDesc();

	vector<int> usedIndex;	// record indexs of points used to form candidate matches 
	int usedCount = 0;
	usedIndex.assign(keypoints.size(), -1);

	matches.clear();
	for (int i = 0; i < vecModelPoint.size(); i++){
		const DescType& fea1 = vecModelDesc[i];

		// filter points under appearance constraints
		vector<cv::KeyPoint> candidates;

		for (int k = 0; k < keypoints.size(); k++) {
			const DescType& fea2 = descriptors[k];

			double sim = ObjModel::feaSimilarity(fea1, fea2);

			if (sim >= m_dAppTol) {
				cv::KeyPoint kp = keypoints[k];
				kp.class_id = k;
				kp.response = (float)sim;
				candidates.push_back(kp);
			}

		}

		// select points with maximum similarity
		sort(candidates.begin(), candidates.end(), CompareKeypoints);
		int size = min((int)candidates.size(), m_nMaxCandidates);
		candidates.resize(size);
		for (int j = 0; j < candidates.size(); j++) {
			MatchInfo info;
			info.idx_g1 = i;
			info.idx_g2 = candidates[j].class_id;
			info.idx_match = (int)matches.size();
			info.score = candidates[j].response;
			matches.push_back(info);

			if (usedIndex[info.idx_g2] < 0) {
				usedIndex[info.idx_g2] = usedCount;
				usedCount++;
			}
		}
	}

	// generate groups information
	group1 = cv::Mat_<double>::zeros((int)matches.size(), (int)vecModelPoint.size());
	group2 = cv::Mat_<double>::zeros((int)matches.size(), usedCount);
	for (int i = 0; i < matches.size(); i++) {
		group1.at<double>(i, matches[i].idx_g1) = 1.0;
		group2.at<double>(i, usedIndex[matches[i].idx_g2]) = 1.0;
	}

	return 1;
}

int ObjDetector::genAffinityMatrix2(const vector<cv::KeyPoint>& keypoints, const vector<DescType>& descriptors,
	vector<MatchInfo>& matches, cv::Mat_<double>& group1, cv::Mat_<double>& group2, cv::Mat_<double>& affinity)
{
	// generate candidate match set
	genCandidateSet2(keypoints, descriptors, matches, group1, group2);
	if (matches.size() <= 0)
		return 0;


	const vector<cv::KeyPoint>& vecModelPoint = m_pModel->getModelPoint();
	const vector<DescType>& vecModelDesc = m_pModel->getModelDesc();


	affinity = cv::Mat_<double>::zeros((int)matches.size(), (int)matches.size());
	for (int i = 0; i < matches.size(); i++) {
		int p1 = matches[i].idx_g1;
		int p2 = matches[i].idx_g2;

		double sim_p = matches[i].score;

		double px1 = vecModelPoint[p1].pt.x;
		double py1 = vecModelPoint[p1].pt.y;
		double px2 = keypoints[p2].pt.x;
		double py2 = keypoints[p2].pt.y;

		for (int k = 0; k < matches.size(); k++) {
			int q1 = matches[k].idx_g1;
			int q2 = matches[k].idx_g2;

			double sim_q = matches[k].score;

			double qx1 = vecModelPoint[q1].pt.x;
			double qy1 = vecModelPoint[q1].pt.y;
			double qx2 = keypoints[q2].pt.x;
			double qy2 = keypoints[q2].pt.y;


			if (p1 == q1 && p2 == q2) {		// node affinity
				affinity(i, k) = sim_p;
			}
			else if (p1 != q1 && p2 != q2) {	// edge affinity
				double dist1 = sqrt((px1 - qx1) * (px1 - qx1) + (py1 - qy1) * (py1 - qy1));
				double dist2 = sqrt((px2 - qx2) * (px2 - qx2) + (py2 - qy2) * (py2 - qy2));

				double dx = abs((qx1 - px1) - (qx2 - px2));
				double dy = abs((qy1 - py1) - (qy2 - py2));
				double dist3 = sqrt((dx * dx) + (dy * dy));

				affinity(i, k) = m_dGeoWeight * (max(0.0, 1.0 - dist3 / (max(dist1, dist2))));

				//	affinity(i, k) = m_dGeoWeight * (1.0 - abs(dist1 - dist2) / (dist1 + dist2));
				affinity(i, k) = affinity(i, k) + (sim_p + sim_q) / 2;
			}
			else {
				affinity(i, k) = 0.0;
			}

		}
	}

	return matches.size();
}
*/