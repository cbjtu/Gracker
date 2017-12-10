#pragma once

#include <time.h>
#include "common.h"
#include "GMA.h"
#include "Warpping.h"
#include "Polygon.h"
#include "ObjDetector.h"
#include "GraphModel.h"




#if GRACK_DESC_TYPE == TYPE_BRISK
	#include "Brisker.h"
#elif GRACK_DESC_TYPE == TYPE_SIFT
	#include "Siftor.h"
#endif

struct GraConfig{
	int		nWarppingMode = WARPPINGMODE_PERSPECTIVE;
	int     nDescType = TYPE_SIFT;
	int     nModelGrids = 10;
	int		nMaxIter = 1;
	bool	bPredictShape = false;
//	bool    bUpdateDesc = false;
	double  dPointAppTol = 0.6;      // the appearance tolerance of two points of a candidate match
	double  dAppTolUpdate = 0.7;
	double  dDetectAppTol = 0.6;
	double  dPredWeight = 0.5;
};

ostream& operator << (ostream& out, GraConfig& grac);


class Gracker
{
public:
	Gracker(const GraConfig& conf);
	~Gracker();

	int initModel(cv::Mat& frame, vector<cv::Point2f>& initPoints);

	Warpping processFrame(cv::Mat& frame);

	void setQuietMode(int mode = 1) { m_nQuietMode = mode; }

	int getModelWidth()	{ return m_initGraph.getWidth(); }
	int getModelHeight() { return m_initGraph.getHeight(); }
	Warpping getCurrentWarp()	{ return m_curWarpping; }
	cv::Mat getDebugImage()			{ return m_imgDebug; }
	vector<cv::Point2f> getInitPolygon()	{ return m_initPolygon; }

public:
	clock_t m_lWarpTime;
	clock_t m_lConstructTime;
	clock_t m_lMatrixTime;
	clock_t m_lMatchingTime;
	clock_t m_lAffineTime;
	clock_t m_lTotalTime;

protected:
	vector<cv::Point2f> calcModelLoc(const cv::Mat& img, const vector<cv::Point2i> & loc, Warpping& w);

	Warpping computeWarpping(const GraphModel& srcG, const GraphModel& dstG, vector<MatchInfo>& matches, cv::Mat_<double>& x, Warpping& predWarp);
//	Warpping computeWarpping(vector<cv::KeyPoint>& dst, vector<MatchInfo>& matches, cv::Mat_<double>& x);
//	Warpping computeWarpping(vector<cv::KeyPoint>& dst, vector<MatchInfo>& matches, cv::Mat_<double>& x, Warpping& predWarp);
	Warpping predictWarpping(vector<Warpping>& warpHist);
	void recordWarpping(vector<Warpping>& warpHist, Warpping& w);


	int genAffinityMatrix(GraphModel& srcG, GraphModel& dstG, Warpping& w,
		vector<MatchInfo>& matches, cv::Mat_<double>& group1, cv::Mat_<double>& group2, cv::Mat_<double>& affinity);
	int genCandidateSet(GraphModel& srcG, GraphModel& dstG, Warpping& w,
		vector<MatchInfo>& matches, cv::Mat_<double>& group1, cv::Mat_<double>& group2);

	double calcMatchingSimilarity(GraphModel& srcG, GraphModel& dstG, vector<MatchInfo>& matches, cv::Mat& x);
	bool   isWarppingValid(Warpping& w);

	void drawDebugImg(cv::Mat& frame, GraphModel& srcG, vector<MatchInfo>& matches, cv::Mat_<double>& x, Warpping& curWarp);
	void drawWarppingRect(cv::Mat& img, vector<cv::Point2f>& points, Warpping& w, cv::Scalar color);


protected:

	GraphModel			m_initGraph;
	GraphModel			m_dstG;
	ObjDetector			m_detector;

	Warpping			 m_curWarpping;
	vector<Warpping>	 m_vecWarpHist;

	cv::Mat				 m_imgDebug;
	cv::Mat				 m_imgInit;
	cv::Rect			 m_rctInit;

	vector<cv::Point2f>  m_initPolygon;				// points of the initial polygon
	cv::Point2f			 m_initCenter;
	cv::Rect			 m_frmRect;


	int					 m_nQuietMode;

	enum WARPPING_PREDICTOR{
		PREDICTOR_DIRECT = 0,
		PREDICTOR_LINEAR,
		PREDICTOR_QUARD
	};

	WARPPING_PREDICTOR	m_predictor = PREDICTOR_LINEAR ;		// quadratic predictor of affine changes
	
	int	  m_nWarppingMode = WARPPINGMODE_PERSPECTIVE;
	int   m_nDescType = TYPE_SIFT;
	bool  m_bPredictShape = false;
//	bool  m_bUpdateDesc = false;
//	bool  m_bDescUpdated = false;

	int  m_nMaxIter = 5;				// the maximum number of iteration for warpping optimization
	int  m_nMaxWarpHist = 2;			// the maximum number of recorded affine transformation history
	int  m_nMaxCandidates = 5;			// the maximum number of candidate corresponding points of each model point
	int	 m_nPointGeoTol = 20;			// the geometirc tolerance of two points of a candidate match
	
	int	 m_nModelGrids = 10;		// the maximum number of keypoints in the model


	double  m_dPointAppTol = 0.6;      // the appearance tolerance of two points of a candidate match
	double  m_dGeoWeight = 1.0;         // the weight of geometric consistency in computing matches affinity
	double  m_dPredWeight = 0.5;

	// for adaptive point appearance tolarance
	double  m_dInitPointAppTol = 0.6;
	double  m_dAppTolUpdate = 0.7;
};