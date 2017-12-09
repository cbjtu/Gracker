#pragma once
#include "common.h"
#include "GMA.h"
#include "Warpping.h"
#include "Siftor.h"

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/gpu/gpu.hpp>

class ObjDetector
{
public:
	ObjDetector();
	~ObjDetector();

	bool setModel(const cv::Mat& model);
//	void setModel(ObjModel* pModel)	{ m_pModel = pModel; }

	bool detect(const cv::Mat& frame, Warpping& w);


	void setAppTol(double tol) { m_dAppTol = tol; }

//	bool detect2(const cv::Mat& frame, AffineT& aff);

protected:
	int genCandidateSet(const vector<cv::KeyPoint>& keypoints, const cv::Mat& descriptors,
		vector<MatchInfo>& matches, cv::Mat_<double>& group1, cv::Mat_<double>& group2);
	int genAffinityMatrix(const vector<cv::KeyPoint>& keypoints, const cv::Mat& descriptors,
		vector<MatchInfo>& matches, cv::Mat_<double>& group1, cv::Mat_<double>& group2, cv::Mat_<double>& affinity);

	bool templateMatch(const cv::Mat& frame, Warpping& w);


//	int genCandidateSet2(const vector<cv::KeyPoint>& keypoints, const vector<DescType>& descriptors,
//		vector<MatchInfo>& matches, cv::Mat_<double>& group1, cv::Mat_<double>& group2);
//	int genAffinityMatrix2(const vector<cv::KeyPoint>& keypoints, const vector<DescType>& descriptors,
//		vector<MatchInfo>& matches, cv::Mat_<double>& group1, cv::Mat_<double>& group2, cv::Mat_<double>& affinity);



//	AffineT computeAffine(vector<cv::Point2f>& src, vector<cv::Point2f>& dst);

	cv::Rect boxing(const cv::Mat& frame);
	void calcColorHist(cv::Mat& img, vector<int>& hist);

//	void drawDebugImage(const cv::Mat& frame, AffineT& aff, vector<Point2f>& obj, vector<Point2f>& scene, vector<Point2f>& inliersObj, vector<Point2f>& inliersScene);

protected:
	cv::FeatureDetector* m_pFeatureDetector;
	cv::DescriptorExtractor* m_pDescriptorExtractor;

	cv::HOGDescriptor*	m_pHogExtractor;
	cv::Size			m_hogWindow;
	cv::Rect			m_objRect;

	vector<cv::KeyPoint>	m_vecModelPoint;
	cv::Mat					m_matModelDesc;
	cv::Mat					m_imgModel;


#if GRACK_DESC_TYPE == TYPE_BRISK
	Brisker				m_brisker;
#elif GRACK_DESC_TYPE == TYPE_SIFT
	Siftor				 m_siftor;
#endif

//	ObjModel*	m_pModel;


	double  m_dGeoWeight = 1.0;         // the weight of geometric consistency in computing matches affinity
	double m_dAppTol = 0.70;			 // the appearance tolerance of two points of a candidate match
	int m_nMaxCandidates = 5;			// the maximum number of candidate corresponding points of each model point
	int m_nMinKeypointDist = 5;		// the minimum location distance between two detected keypoints 
	int m_nMaxModelPoints = 100;
};

