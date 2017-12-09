#include "Gracker.h"
#include "Utils.h"
#include <iostream>
#include <fstream>

static const char* kWarpedWindow = "warped";


ostream& operator << (ostream& out, GraConfig& grac)
{
	out << endl;
	out << "GraConfig {" << endl;
	out << "\tnWarppingMode = " << ((grac.nWarppingMode == WARPPINGMODE_AFFINE) ? "AFFINE" : "PERSPECTIVE") << endl;
	out << "\tbPredictShape = " << grac.bPredictShape << endl;
	out << "\tdPointAppTol = " << grac.dPointAppTol << endl;
	out << "\tdDetectAppTol = " << grac.dDetectAppTol << endl;
	out << "\tdPredWeight = " << grac.dPredWeight << endl;
	out << "};" << endl;
	out << endl;

	return out;
}


Gracker::Gracker(const GraConfig& conf) : 
	m_nWarppingMode(conf.nWarppingMode), 
	m_nDescType(conf.nDescType),
	m_nModelGrids(conf.nModelGrids),
	m_nMaxIter(conf.nMaxIter),
	m_bPredictShape(conf.bPredictShape),
//	m_bUpdateDesc(conf.bUpdateDesc),
	m_dPointAppTol(conf.dPointAppTol),
	m_dAppTolUpdate(conf.dAppTolUpdate),
	m_dPredWeight(conf.dPredWeight),
	m_curWarpping(conf.nWarppingMode)
{
	Warpping::setPredictShape(m_bPredictShape);

	m_detector.setAppTol(conf.dDetectAppTol);

	m_nQuietMode = 1;
}


Gracker::~Gracker()
{
}


int Gracker::initModel(cv::Mat& frame, vector<cv::Point2f>& initPoints)
{
	if (frame.type() != CV_8UC1)
		return 0;

	int left = 100000, top = 100000;
	int right = 0, bottom = 0;
	for (int i = 0; i < initPoints.size(); i++) {
		left = min(left, (int)initPoints[i].x);
		right = max(right, (int)initPoints[i].x);
		top = min(top, (int)initPoints[i].y);
		bottom = max(bottom, (int)initPoints[i].y);
	}
	
	cv::Rect rect = cvRect(left, top, right - left + 1, bottom - top + 1);

	m_initPolygon.clear();
	m_initCenter.x = m_initCenter.y = 0;
	for (int i = 0; i < initPoints.size(); i++) {
		m_initPolygon.push_back(cv::Point2f(initPoints[i].x - left, initPoints[i].y - top));

		m_initCenter.x += initPoints[i].x - left;
		m_initCenter.y += initPoints[i].y - top;
	}
	//几何中心
	m_initCenter.x /= (float)initPoints.size();
	m_initCenter.y /= (float)initPoints.size();


	m_nPointGeoTol = (int)sqrt(rect.width * rect.width + rect.height * rect.height) / 10;//匹配的最大容忍偏差
	m_nPointGeoTol = max(20, min(40, m_nPointGeoTol));//X>=20 || X<=40

	// initialize warpping function
	m_curWarpping.initialize(rect.x, rect.y);

	// initial graph model
	m_initGraph.setDescType(m_nDescType);
	m_initGraph.setGrids(50);
//	m_initGraph.setGrids(m_nModelGrids);
	m_initGraph.construct(frame, rect);
	m_initGraph.transferNodeLoc(m_curWarpping.inv());

	m_frmRect = cv::Rect(0, 0, frame.cols, frame.rows);

	m_vecWarpHist.clear();
	recordWarpping(m_vecWarpHist, m_curWarpping);

	m_lWarpTime = 0;
	m_lConstructTime = 0;
	m_lMatrixTime = 0;
	m_lMatchingTime = 0;
	m_lAffineTime = 0;
	m_lTotalTime = 0;

	m_imgInit = frame.clone();
	m_rctInit = rect;

	m_detector.setModel(m_initGraph.getImage());

	return 1;
}


Warpping Gracker::processFrame(cv::Mat& frame)
{
	if (frame.type() != CV_8UC1) {
		m_curWarpping.zero();
		return m_curWarpping;
	}

	int iter = 0;

	GMA gma;
	cv::Mat_<double> x;
	vector<MatchInfo> matches;



	Warpping predWarp = predictWarpping(m_vecWarpHist);
	if (!isWarppingValid(predWarp)) {//非法变形
		bool bRet = m_detector.detect(frame, predWarp);//重新检测
		if (bRet) {
			cout << " <<<<<<<<<<<<<<<<<  detect object in the frame <<<<<<<<<<<<<<<<<< " << endl;
		//	cout << predAffine << endl;
		}
		else {
			cout << " !!!!!!!!!!!!!!!!!!!! detect object in the frame failed !!!!!!!!!!!!!!!!!!!!" << endl;
			m_curWarpping.zero();
			return m_curWarpping;
		} 

		m_dPointAppTol = m_dInitPointAppTol;
	}


	// parameters for warpping frame back to initial pos
	//初始化位置
	int width = m_initGraph.getWidth();
	int height = m_initGraph.getHeight();
	int xshift = (width < 64) ? 8 : 0;//移动而来为8，否则是重新侦测的
	int yshift = (height < 64) ? 8 : 0;



	Warpping initWarp = predWarp;
	while (1) {
		// initialize affine transformation as that in the last frame
		//初始化上一帧的仿射转换
		clock_t t0 = clock();

		// warp current frame back to initial pos
		//把当前帧移动到初始位置
		cv::Mat H_ = initWarp.getH().inv();
		H_.at<double>(0, 2) += xshift;
		H_.at<double>(1, 2) += yshift;
		cv::Mat warped; 
		cv::warpPerspective(frame, warped, H_, cv::Size(width + 2 * xshift, height + 2 * yshift));//透视转换
		m_lWarpTime = m_lWarpTime + clock() - t0;//时间修正

		
		t0 = clock();
		// construct graph model构造图像模型
		cv::Rect rect(xshift, yshift, width, height);
		m_dstG.setDescType(m_nDescType);
		m_dstG.setGrids(m_nModelGrids);
		m_dstG.construct(warped, rect);

		// transfer dest node location 
		//转换目标位置
		Warpping w(m_nWarppingMode);
		w.setH(H_.inv());
		m_dstG.transferNodeLoc(w);
		m_lConstructTime = m_lConstructTime + clock() - t0;


		// compute affinity matrix for graph matching
		t0 = clock();
		cv::Mat_<double> affinity, group1, group2;
		genAffinityMatrix(m_initGraph, m_dstG, initWarp, matches, group1, group2, affinity);
		m_lMatrixTime = m_lMatrixTime + clock() - t0;

		t0 = clock();
		if (matches.size() > 0) {
		//	x = gma.RRWM(affinity, matches, group1, group2);
			x = gma.IPFP(affinity, matches, group1, group2);
			x = gma.greedyMapping(x, matches);
		}
		else{
			x = cv::Mat_<double>::zeros(1, 1);
		}
		m_lMatchingTime = m_lMatchingTime + clock() - t0;

		if (cv::norm(x, CV_L1) < 10) {	
			// no enough matches to compute affine trasform, then set affine to zero
			cout << "matches: " << matches.size() << ", x_norm: " << cv::norm(x, CV_L1) << endl;
			m_curWarpping.zero();
			break;
		}

		t0 = clock();
		Warpping dstW(m_nWarppingMode);
		dstW.setH(H_.inv());
		m_curWarpping = computeWarpping(m_initGraph, m_dstG, matches, x, predWarp);

		m_lAffineTime = m_lAffineTime + clock() - t0;

		// for next iteration
		initWarp = m_curWarpping;	

		if (++iter >= m_nMaxIter)//m_nMaxIter
			break;
	}


	if (isWarppingValid(m_curWarpping)){
		recordWarpping(m_vecWarpHist, m_curWarpping);
	}
	else {
		m_vecWarpHist.clear();
	}

	if (m_nQuietMode) {
		drawDebugImg(frame, m_initGraph, matches, x, m_curWarpping);
	}


	cout << "nodes: " << m_dstG.getNodeNum() << ", matches: " << matches.size() << endl;
	cout << "Warp image: " << m_lWarpTime * 1.0 / CLOCKS_PER_SEC << endl;
	cout << "Construct graph: " << m_lConstructTime * 1.0 / CLOCKS_PER_SEC << endl;
	cout << "Build matrix    : " << m_lMatrixTime * 1.0 / CLOCKS_PER_SEC << endl;
	cout << "Graph matching  ：" << m_lMatchingTime * 1.0 / CLOCKS_PER_SEC << endl;
	cout << "Affine Transform: " << m_lAffineTime * 1.0 / CLOCKS_PER_SEC << endl;


	return m_curWarpping;
}

void Gracker::drawDebugImg(cv::Mat& frame, GraphModel& srcG, vector<MatchInfo>& matches, cv::Mat_<double>& x, Warpping& w)
{
	vector<cv::KeyPoint> vecSrcPoint;
	vector<cv::KeyPoint> vecMatchPoint;
	vector<cv::DMatch> DMatches;

	planimetry::Polygon border(m_initPolygon);

/*	int index = 0;
	for (int i = 0; i < srcG.getNodeNum(); i++) {
		const GraphNode& node = srcG.getNode(i);
		if (!border.isPointInPolygon(planimetry::Vector2d(node.kp.pt.x, node.kp.pt.y)))
			continue;

		vecSrcPoint.push_back(node.kp);

		cv::KeyPoint kp;
		kp.pt = w.doWarpping(node.kp.pt);
		vecMatchPoint.push_back(kp);

		cv::DMatch m;
		m.distance = 0.0;
		m.imgIdx = 0;
		m.queryIdx = index;
		m.trainIdx = index;
		DMatches.push_back(m);

		index++;
	}
*/
	cv::drawMatches(srcG.getImage(), vecSrcPoint, frame, vecMatchPoint, DMatches, m_imgDebug, cv::Scalar(255, 255, 0));

	drawWarppingRect(m_imgDebug, m_initPolygon, w, cv::Scalar(0, 255, 0));
}


void Gracker::drawWarppingRect(cv::Mat& img, vector<cv::Point2f>& points, Warpping& w, cv::Scalar color)
{
	vector<cv::Point2f> dstPoints;
	for (int i = 0; i < points.size(); i++) {
		cv::Point2f ap = w.doWarpping(points[i]);
		ap.x += m_initGraph.getWidth();

		ap.x *= 16.0;
		ap.y *= 16.0;

		dstPoints.push_back(ap);
	}

	int width = 2;
	for (int i = 0; i < points.size(); i++) {
		cv::line(img, dstPoints[i], dstPoints[(i + 1) % dstPoints.size()], color, width, CV_AA, 4);
	}

}


vector<cv::Point2f> Gracker::calcModelLoc(const cv::Mat& img, const vector<cv::Point2i> & loc, Warpping& w)
{
	vector<cv::Point2f> vecPoints;
	for (int i = 0; i < loc.size(); i++) {
		cv::Point2f pt = w.doWarpping(loc[i]);

	//	// remove points out of region
	//	if (pt.x < 0 || pt.x >= img.cols || pt.y < 0 || pt.y >= img.rows)
	//		continue;

		vecPoints.push_back(pt);
	}

	return vecPoints;
}


double  Gracker::calcMatchingSimilarity(GraphModel& srcG, GraphModel& dstG, vector<MatchInfo>& matches, cv::Mat& x)
{
	double sim = 0.0;
	int count = 0;
	for (int i = 0; i < matches.size(); i++) {
		if (x.at<double>(i, 0) > 0.5) {
			int p_n1 = matches[i].idx_g1;
			int p_n2 = matches[i].idx_g2;
			const GraphNode& srcNode = srcG.getNode(p_n1);
			const GraphNode& dstNode = dstG.getNode(p_n2);
			sim += GraphModel::compareNode(srcNode, dstNode);
			count++;
		}
	}
	return sim / count;
}

int Gracker::genAffinityMatrix(GraphModel& srcG, GraphModel& dstG, Warpping& w,
	vector<MatchInfo>& matches, cv::Mat_<double>& group1, cv::Mat_<double>& group2, cv::Mat_<double>& affinity)
{
	clock_t t0 = clock();
	genCandidateSet(srcG, dstG, w, matches, group1, group2);
	clock_t t1 = clock() - t0;
	t0 = clock();

	if (matches.size() <= 0)
		return 0;

	vector<cv::Point2f> warpPts;
	for (int i = 0; i < matches.size(); i++) {
		int n1 = matches[i].idx_g1;
		cv::Point2f mp = w.doWarpping(srcG.getNode(n1).kp.pt);
		warpPts.push_back(mp);
	}

	affinity = cv::Mat_<double>::zeros((int)matches.size(), (int)matches.size());
	for (int p = 0; p < matches.size(); p++) {
		int p_n1 = matches[p].idx_g1;
		int p_n2 = matches[p].idx_g2;

		double sim_p = matches[p].score;

		double px = dstG.getNode(p_n2).kp.pt.x;
		double py = dstG.getNode(p_n2).kp.pt.y;
		double mpx = warpPts[p].x;
		double mpy = warpPts[p].y;


		for (int q = 0; q < matches.size(); q++) {
			int q_n1 = matches[q].idx_g1;
			int q_n2 = matches[q].idx_g2;

			double sim_q = matches[q].score;

			double qx = dstG.getNode(q_n2).kp.pt.x;
			double qy = dstG.getNode(q_n2).kp.pt.y;
			double mqx = warpPts[q].x;
			double mqy = warpPts[q].y;

			if (p == q) {		// node affinity
				affinity(p, q) = sim_p;
			}
			else if (p_n1 != q_n1 && p_n2 != q_n2) {	// edge affinity
				//	double dist1 = sqrt((mpx - px) * (mpx - px) + (mpy - py) * (mpy - py));
				//	double dist2 = sqrt((mqx - qx) * (mqx - qx) + (mqy - qy) * (mqy - qy));
				//	affinity(p, q) = m_dGeoWeight * exp(-(dist1 + dist2) / m_fPointGeoTol);
				//	affinity(p, q) = max(0.0, m_dGeoWeight * (m_nPointGeoTol - (dist1 + dist2)));

				double dist1 = sqrt((px - qx) * (px - qx) + (py - qy) * (py - qy));
				double dist2 = sqrt((mpx - mqx) * (mpx - mqx) + (mpy - mqy) * (mpy - mqy));
				affinity(p, q) = max(0.0, m_dGeoWeight * (m_nPointGeoTol - abs(dist1 - dist2)) / m_nPointGeoTol);
				affinity(p, q) = affinity(p, q) + (sim_p + sim_q) / 2;

				//	double dx = px - qx;
				//	double dy = py - qy;
				//	double dmx = mpx - mqx;
				//	double dmy = mpy - mqy;
				//	double dist = sqrt((dx - dmx) * (dx - dmx) + (dy - dmy) * (dy - dmy));
				//	//	affinity(p, q) = m_dGeoWeight * exp(-dist / m_fPointGeoTol);
				//	affinity(p, q) = max(0.0, m_dGeoWeight * (m_nPointGeoTol - dist) / m_nPointGeoTol);
				//	affinity(p, q) = affinity(p, q) + (sim_p + sim_q) / 2;
			}
			else {
				affinity(p, q) = 0.0;
			}

		}
	}

	clock_t t2 = clock() - t0;

	//	cout << "candidate set time : " << t1 << ",  affinity time : " << t2 << endl;

	return 1;

}

int Gracker::genCandidateSet(GraphModel& srcG, GraphModel& dstG, Warpping& w,
	vector<MatchInfo>& matches, cv::Mat_<double>& group1, cv::Mat_<double>& group2)
{
	matches.clear();

	for (int di = 0; di < dstG.getNodeNum(); di++) {
		const GraphNode& dstNode = dstG.getNode(di);
		cv::Point2f p1 = w.invWarpping(dstNode.kp.pt);
		double x1 = p1.x;
		double y1 = p1.y;

		// generate candidate match points
		vector<cv::KeyPoint> candidates;
		for (int si = 0; si < srcG.getNodeNum(); si++) {
			const GraphNode& srcNode = srcG.getNode(si);
			double x2 = srcNode.kp.pt.x;
			double y2 = srcNode.kp.pt.y;

			double dist = sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));

			// filter points under appearance and geometric constraints
			if ( dist < m_nPointGeoTol) {
				double sim = GraphModel::compareNode(srcNode, dstNode);
				if (sim >= m_dPointAppTol) {
					cv::KeyPoint kp = srcNode.kp;
					kp.class_id = si;
					kp.response = (float)sim;
					candidates.push_back(kp);
				}
			}
		}

		// select points with maximum similarity
		sort(candidates.begin(), candidates.end(), CompareKeypoints);
		int size = min((int)candidates.size(), m_nMaxCandidates);
		candidates.resize(size);
		for (int j = 0; j < candidates.size(); j++) {
			MatchInfo info;
			info.idx_g1 = candidates[j].class_id;
			info.idx_g2 = di;
			info.idx_match = (int)matches.size();
			info.score = candidates[j].response;
			matches.push_back(info);
		}
	}

	// generate groups information
	group1 = cv::Mat_<double>::zeros((int)matches.size(), srcG.getNodeNum());
	group2 = cv::Mat_<double>::zeros((int)matches.size(), dstG.getNodeNum());
	for (int i = 0; i < matches.size(); i++) {
		group1.at<double>(i, matches[i].idx_g1) = 1.0;
		group2.at<double>(i, matches[i].idx_g2) = 1.0;
	}

	return 1;
}

Warpping Gracker::computeWarpping(const GraphModel& srcG, const GraphModel& dstG, 
	vector<MatchInfo>& matches, cv::Mat_<double>& x, Warpping& predWarp)
{
	Warpping w(m_nWarppingMode);

	// compute mean error under given init affine
	vector<cv::Point2f> srcPts, dstPts;
	for (int i = 0; i < x.rows; i++) {
		if (x(i, 0) > 0.0) {
			int idx1 = matches[i].idx_g1;
			int idx2 = matches[i].idx_g2;
			srcPts.push_back(srcG.getNode(idx1).kp.pt);
			dstPts.push_back(dstG.getNode(idx2).kp.pt);
		}
	}  

	vector<cv::Point2f> predPts = predWarp.doWarpping(srcPts);

	int num = (int)srcPts.size() / 2;
	for (int iter = 0; iter < num; iter++) {
		w.computeWarpping(srcPts, dstPts, predPts, m_dPredWeight);

		vector<cv::Point2f> warpPts = w.doWarpping(srcPts);

		// repeatedly remove the match with maximum error
		vector<double> errs(srcPts.size(), 0.0);
		double max_err = 0.0;
		int max_idx = 0;
		for (int i = 0; i < srcPts.size(); i++) {
			double dx = dstPts[i].x - warpPts[i].x;
			double dy = dstPts[i].y - warpPts[i].y;
			errs[i] = sqrt(dx * dx + dy * dy);

			if (errs[i] > max_err) {
				max_err = errs[i];
				max_idx = i;
			}
		}

		srcPts.erase(srcPts.begin() + max_idx);
		dstPts.erase(dstPts.begin() + max_idx);
		predPts.erase(predPts.begin() + max_idx);
	}


	return w;
}

/*
Warpping Gracker::computeWarpping(vector<cv::KeyPoint>& dst, vector<MatchInfo>& matches, cv::Mat_<double> &x)
{
	Warpping w(m_nWarppingMode);

	vector<cv::Point2f> vecSrcPts, vecDstPts;
	for (int i = 0; i < x.rows; i++) {
		if (x(i, 0) > 0.0) {
			int idx1 = matches[i].idx_g1;
			int row = idx1 / m_model.getWidth();
			int col = idx1 % m_model.getWidth();
			vecSrcPts.push_back(m_model.getKeyPoint(col, row).pt);

			int idx2 = matches[i].idx_g2;
			vecDstPts.push_back(dst[idx2].pt);
		}
	}


	int num = (int)vecSrcPts.size();
	for (int iter = 0; iter < num / 2; iter++) {
		w.computeWarpping(vecSrcPts, vecDstPts);

		vector<cv::Point2f> warpPts = w.doWarpping(vecSrcPts);

		// repeatedly remove the match with maximum error
		vector<double> errs(vecSrcPts.size(), 0.0);
		double max_err = 0.0;
		int max_idx = 0;
		for (int i = 0; i < vecSrcPts.size(); i++) {
			double dx = vecDstPts[i].x - warpPts[i].x;
			double dy = vecDstPts[i].y - warpPts[i].y;
			errs[i] = sqrt(dx * dx + dy * dy);

			if (errs[i] > max_err) {
				max_err = errs[i];
				max_idx = i;
			}
		}

		vecSrcPts.erase(vecSrcPts.begin() + max_idx);
		vecDstPts.erase(vecDstPts.begin() + max_idx);
	}

//	w.computeWarpping(vecSrcPts, vecDstPts);

	return w;
}
*/
/*
int Gracker::computeWarpping2(vector<cv::KeyPoint>& dst, vector<MatchInfo>& matches, cv::Mat_<double>& x, Warpping& w)
{
	vector<cv::Point2f> vecSrcPts, vecDstPts;
	for (int i = 0; i < x.rows; i++) {
		if (x(i, 0) > 0.0) {
			int idx1 = matches[i].idx_g1;
			int row = idx1 / m_model.getWidth();
			int col = idx1 % m_model.getWidth();
			vecSrcPts.push_back(m_model.getKeyPoint(col, row).pt);

			int idx2 = matches[i].idx_g2;
			vecDstPts.push_back(dst[idx2].pt);
		}
	}

	GMC gmc;
	vector<int> flag = gmc.getConsensus(m_frmRect, vecSrcPts, vecDstPts);

	vector<cv::Point2f> obj, scene;
	for (int i = 0; i < flag.size(); i++) {
		if (flag[i]) {
			obj.push_back(vecSrcPts[i]);
			scene.push_back(vecDstPts[i]);
		}
	}

	if (obj.size() < 10)
		return 0;
	else {
		w.computeWarpping(obj, scene);
		return 1;
	}
}
*/

/*
cv::Mat_<double> Gracker::optimizeAffine(vector<cv::KeyPoint>& src, vector<cv::KeyPoint>& dst, vector<MatchInfo>& matches, cv::Mat_<double>& x)
{
	cv::Mat_<double> x_ = x.clone();
	cv::Mat_<double> T;

	for (int iter = 0; iter < 2; iter++) {

		// minimize the joint pairwise error 
		int count = 0;
		for (int i = 0; i < x_.rows; i++) {
			if (x_(i, 0) > 0.0)
				count++;
		}

		int num = count * (count - 1) / 2;

		cv::Mat_<double> srcGraph = cv::Mat_<double>::zeros(3, num);
		cv::Mat_<double> dstGraph = cv::Mat_<double>::zeros(2, num);
		int index = 0;
		for (int p = 0; p < matches.size(); p++) {
			if (x_(p, 0) <= 0.0)
				continue;

			int p_idx1 = matches[p].idx_g1;
			double p_x1 = src[p_idx1].pt.x;
			double p_y1 = src[p_idx1].pt.y;

			int p_idx2 = matches[p].idx_g2;
			double p_x2 = dst[p_idx2].pt.x;
			double p_y2 = dst[p_idx2].pt.y;

			double pro_p = x_(p, 0);


			for (int q = p + 1; q < matches.size(); q++) {
				if (x_(q, 0) <= 0.0)
					continue;

				int q_idx1 = matches[q].idx_g1;
				double q_x1 = src[q_idx1].pt.x;
				double q_y1 = src[q_idx1].pt.y;

				int q_idx2 = matches[q].idx_g2;
				double q_x2 = dst[q_idx2].pt.x;
				double q_y2 = dst[q_idx2].pt.y;

				double pro_q = x_(q, 0);
				double pro = pro_p * pro_q;

				srcGraph(0, index) = pro * (p_x1 - q_x1);
				srcGraph(1, index) = pro * (p_y1 - q_y1);
				srcGraph(2, index) = pro * 1;

				dstGraph(0, index) = pro * (p_x2 - q_x2);
				dstGraph(1, index) = pro * (p_y2 - q_y2);

				index++;
			}
		}

		cv::Mat_<double> Y = dstGraph;
		cv::Mat_<double> X = srcGraph;
		T = (Y * X.t()) * (X * X.t()).inv();

		// remove outliers, and recomputate in the next iteration
		double avg_err = 0.0;
		double * errs = new double[matches.size()];
		for (int i = 0; i < matches.size(); i++) {
			if (x_(i, 0) <= 0.0)
				continue;

			cv::Point2f srcPt = src[matches[i].idx_g1].pt;
			cv::Point2f dstPt = dst[matches[i].idx_g2].pt;
			cv::Point2f affPt = CalcAffinePoint(srcPt, T);
			errs[i] = sqrt((dstPt.x - affPt.x) * (dstPt.x - affPt.x) + (dstPt.y - affPt.y) * (dstPt.y - affPt.y));
			avg_err += errs[i];
		}
		avg_err /= count;

		for (int i = 0; i < matches.size(); i++) {
			if (errs[i] > avg_err)
				x_(i, 0) = 0.0;
		}
		delete[]errs;
	}
	return T;
}
*/

void Gracker::recordWarpping(vector<Warpping>& warpHist, Warpping& w)
{
	if (warpHist.size() >= m_nMaxWarpHist) {
		warpHist.erase(warpHist.begin());
	}
	warpHist.push_back(w);
}


Warpping Gracker::predictWarpping(vector<Warpping>& warpHist)
{
	Warpping w(m_nWarppingMode);
	w.zero();

	if (warpHist.size() <= 0) {
		return w;
	}

	vector<Warpping> vecWarpGrad, vecWarpQuard;
	Warpping grad_sum(m_nWarppingMode), quard_sum(m_nWarppingMode);
	Warpping grad(m_nWarppingMode);

	switch (m_predictor) {
	case PREDICTOR_DIRECT:
		w = warpHist[warpHist.size() - 1];
		break;

	case PREDICTOR_LINEAR:
		if (warpHist.size() < 2)
			return warpHist[warpHist.size() - 1];

		grad_sum.zero();
		for (int i = 0; i < warpHist.size() - 1; i++) {
			Warpping grad = warpHist[i + 1] - warpHist[i];
			vecWarpGrad.push_back(grad);
			grad_sum = grad_sum + grad;
		}
		grad_sum = grad_sum * (1.0 / (double)vecWarpGrad.size());
	
		w = warpHist[warpHist.size() - 1] + grad_sum;
		break;

	case PREDICTOR_QUARD:
		if (warpHist.size() < 3)
			return warpHist[warpHist.size() - 1];

		for (int i = 0; i < warpHist.size() - 1; i++) {
			Warpping grad = warpHist[i + 1] - warpHist[i];
			vecWarpGrad.push_back(grad);
		}

		quard_sum.zero();
		for (int i = 0; i < vecWarpGrad.size() - 1; i++) {
			Warpping quard = vecWarpGrad[i + 1] - vecWarpGrad[i];
			vecWarpQuard.push_back(quard);
			quard_sum = quard_sum + quard;
		}
		quard_sum = quard_sum * (1.0 / (double)vecWarpGrad.size());

		grad = vecWarpGrad[vecWarpGrad.size() - 1] + quard_sum;
		w = warpHist[warpHist.size() - 1] + grad;
		break;
	}

	return w;
}


bool Gracker::isWarppingValid(Warpping& w)
{
//	if (!w.isValid())
//		return false;

	planimetry::Polygon initRect(m_initPolygon);

	vector<cv::Point2f> dstPoints;
	for (int i = 0; i < m_initPolygon.size(); i++) {
		dstPoints.push_back(w.doWarpping(m_initPolygon[i]));
	}

	// check area
	planimetry::Polygon dstRect(dstPoints);
	if (dstRect.getArea() * 16 < initRect.getArea())
		return false;
	if (dstRect.getArea() > 16 * initRect.getArea())
		return false;

	// check edge length
	int n = m_initPolygon.size();
	for (int i = 0; i < n; i++) {
		double dist1 = planimetry::getDistance(m_initPolygon[i], m_initPolygon[(i + 1) % n]);
		double dist2 = planimetry::getDistance(dstPoints[i], dstPoints[(i + 1) % n]);
		if (dist1 * 8.0 < dist2 || dist1 > 8.0 * dist2)
			return false;
	}

	// check suddenly deformation to the previous frame
//	cv::Mat H = w.getH();
//	cv::Mat predH = predW.getH();
//	cv::Mat dH = H - predH;
//	dH.at<double>(0, 2) = 0;
//	dH.at<double>(1, 2) = 0;
//	if (cv::norm(dH) >= 5)
//		return false;

	return true;
}

