
#include "GraphModel.h"
#include <iostream>
#include <time.h>


int GraphModel::m_nDescType = TYPE_SIFT;

GraphModel::GraphModel()
{
}


GraphModel::~GraphModel()
{
}

GraphModel::GraphModel(const GraphModel& G)
{
	m_vecNodes.assign(G.m_vecNodes.begin(), G.m_vecNodes.end());
	m_vecEdges.assign(G.m_vecEdges.begin(), G.m_vecEdges.end());
	m_matAdj = G.m_matAdj.clone();
	m_imgTemplate = G.m_imgTemplate.clone();
}


double GraphModel::compareNode(const GraphNode& node1, const GraphNode& node2)
{
	double sim = 0.0;

	if (m_nDescType == TYPE_BRISK) {
		const BRISK_D& fea1 = node1.brisk_desc;
		const BRISK_D& fea2 = node2.brisk_desc;
		// Hamming distance
		unsigned int dist = fea1.HammingDist(fea2);
		sim = 1.0 - (double)dist / (double)(sizeof(fea1.desc) * 8);
	}
	else if (m_nDescType == TYPE_SIFT) {
		const SIFT_D& fea1 = node1.sift_desc;
		const SIFT_D& fea2 = node2.sift_desc;

		// L2 norm
		int sum12 = 0, sum11 = 0, sum22 = 0;
		for (int i = 0; i < 128; i++){
			sum12 += fea1.desc[i] * fea2.desc[i];
			sum11 += fea1.desc[i] * fea1.desc[i];
			sum22 += fea2.desc[i] * fea2.desc[i];
		}
		sim = (double)sum12 / sqrt((double)sum11 * (double)sum22);
	}

	return sim;
}

int GraphModel::construct(cv::Mat& img, cv::Rect rect)
{
	m_vecNodes.clear();
	m_vecEdges.clear();

	m_imgTemplate.release();
	img(rect).copyTo(m_imgTemplate);

	// extract keypoints and descriptors
	vector<cv::KeyPoint> keypoints;
	vector<SIFT_D> sift_descriptors;
	vector<BRISK_D> brisk_descriptors;

	if (m_nDescType == TYPE_BRISK) {
		m_brisker.CalcMaximumBrisk(img, rect, m_nModelGrids, keypoints, brisk_descriptors);
	}
	else if (m_nDescType == TYPE_SIFT) {
		m_siftor.CalcMaximumSift(img, rect, m_nModelGrids, keypoints, sift_descriptors);
	}

	// construct graph nodes
	for (int i = 0; i < keypoints.size(); i++) {
		GraphNode node;
		node.index = i;
		node.kp = keypoints[i];
		if (m_nDescType == TYPE_BRISK) {
			node.brisk_desc = brisk_descriptors[i];
		}
		else if (m_nDescType == TYPE_SIFT) {
			node.sift_desc = sift_descriptors[i];
		}
		m_vecNodes.push_back(node);
	}

	// construct graph edges using Delaunay trianglation
	cv::Subdiv2D div(rect);
	vector<cv::Point2f> pts;
	for (int i = 0; i < keypoints.size(); i++) {
		div.insert(keypoints[i].pt);
		pts.push_back(keypoints[i].pt);
	}

	vector<Vec4f> edges;
	div.getEdgeList(edges);

	// retrieve edges
	m_matAdj = cv::Mat_<int>::zeros(keypoints.size(), keypoints.size());
	for (int i = 0; i < edges.size(); i++){
		Vec4f t = edges[i];
		int ind1 = getPointIndex(rect, cv::Point2f(t[0], t[1]), pts);
		int ind2 = getPointIndex(rect, cv::Point2f(t[2], t[3]), pts);

		if (ind1 >= 0 && ind2 >= 0) {
			m_matAdj(ind1, ind2) = 1;
			m_matAdj(ind2, ind1) = 1;
		}
	}

	for (int row = 0; row < keypoints.size(); row++) {
		for (int col = 0; col < keypoints.size(); col++) {
			if (m_matAdj(row, col)) {
				GraphEdge edge;
				edge.tail = row;
				edge.head = col;
				m_vecEdges.push_back(edge);
			}
		}
	}

	return keypoints.size();
}

/*
void GraphModel::updateDesc(cv::Mat& img, cv::Rect& rect)
{
	vector<cv::Point2i> vecPoints;
	for (int i = 0; i < m_vecNodes.size(); i++) {
		cv::Point2i pt;
		pt.x = rect.x + m_vecNodes[i].kp.pt.x;
		pt.y = rect.y + m_vecNodes[i].kp.pt.y;
		vecPoints.push_back(pt);
	}

	vector<cv::KeyPoint> keypoints;
	vector<DescType> descriptors;
	m_siftor.CalcPointSift(img, vecPoints, keypoints, descriptors);

	for (int i = 0; i < m_vecNodes.size(); i++) {
		m_vecNodes[i].desc = descriptors[i];
	}
}

void GraphModel::resetDesc()
{
for (int i = 0; i < m_vecNodes.size(); i++) {
m_vecNodes[i].desc = m_vecNodes[i].init_desc;
}
}
*/

void GraphModel::transferNodeLoc(Warpping& w)
{
	for (int i = 0; i < m_vecNodes.size(); i++) {
		m_vecNodes[i].kp.pt = w.doWarpping(m_vecNodes[i].kp.pt);
	}
}



int GraphModel::getPointIndex(cv::Rect r, ::Point2f pt, vector<cv::Point2f>& pts)
{
	int index = -1;
	if (pt.x < r.x || pt.x > r.x + r.width || pt.y < r.y || pt.y > r.y + r.height)
		return index;

	double min_dist = 1.0e6;
	for (int i = 0; i < pts.size(); i++) {
		double dist = sqrt((pts[i].x - pt.x) * (pts[i].x - pt.x) + (pts[i].y - pt.y) * (pts[i].y - pt.y));
		if (dist < min_dist) {
			min_dist = dist;
			index = i;
		}
	}
	return index;
}