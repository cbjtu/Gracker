#pragma once

#include "common.h"
#include "Siftor.h"
#include "Brisker.h"

/*
#if GRACK_DESC_TYPE == TYPE_BRISK
#include "Brisker.h"
typedef BRISK_D DescType;
#elif GRACK_DESC_TYPE == TYPE_SIFT
#include "Siftor.h"
typedef SIFT_D DescType;
#endif
*/

struct GraphNode{
	int index;
	cv::KeyPoint kp;
	SIFT_D	sift_desc;
	BRISK_D	brisk_desc;
//	DescType desc;			// descriptor of keypoint, maybe update per frame
//	DescType init_desc;
};

struct GraphEdge{
	int tail;
	int head;
};

#include "Warpping.h"

class GraphModel
{
public:
	GraphModel();
	~GraphModel();
	GraphModel(const GraphModel& G);

	int construct(cv::Mat& img, cv::Rect rect);
	void transferNodeLoc(Warpping& w);
//	void updateDesc(cv::Mat& img, cv::Rect& rect);
//	void resetDesc();

	static double compareNode(const GraphNode& node1, const GraphNode& node2);
	static void setDescType(int type)	{ m_nDescType = type; }

	void setGrids(int nGrid) { m_nModelGrids = nGrid; }

	const cv::Mat& getImage()	{ return m_imgTemplate; }
	const GraphNode& getNode(int index) const { return m_vecNodes[index]; }

	int getNodeNum() const { return m_vecNodes.size(); }
	int getWidth() const	{ return m_imgTemplate.cols; }
	int getHeight() const { return m_imgTemplate.rows; }

protected:
	int getPointIndex(cv::Rect r, ::Point2f pt, vector<cv::Point2f>& pts);


protected:

	Brisker				m_brisker;
	Siftor				m_siftor;

	vector<GraphNode> m_vecNodes;		// node list
	vector<GraphEdge> m_vecEdges;		// edge list
	cv::Mat_<int>	  m_matAdj;			// adjacency matrix
	
	cv::Mat			  m_imgTemplate;	// save image template

	int	 m_nModelGrids = 30;		// the maximum number of keypoints in the model
	
	static int  m_nDescType;
};

