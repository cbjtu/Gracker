#pragma once
#include "common.h"

struct MatchInfo{
	int idx_match;
	int idx_g1;
	int idx_g2;
	double score;
};

class GMA
{
public:
	GMA();
	~GMA();

	cv::Mat SM(cv::Mat& affinity);

	cv::Mat IPFP(cv::Mat& affinity, vector<MatchInfo>& matches, cv::Mat& group1, cv::Mat& group2);
	cv::Mat IPFP(cv::Mat& affinity, vector<MatchInfo>& matches, cv::Mat& group1, cv::Mat& group2, cv::Mat& x0);

	void bistocNorm(double* pX, int N,
		int* pIdx1, int* pID1, int* pIdx2, int* pID2,
		double* pTol, double* pDumDim, double* pDumVal, int* pMaxIter,
		double* pY);
	void make_groups_slack(const vector<MatchInfo>& matches, const cv::Mat& group1, const cv::Mat& group2,
		double& dumDim, double& dumVal, int& dumSize, int* idx1, int* ID1, int* idx2, int* ID2);
	cv::Mat RRWM(cv::Mat& affinity, vector<MatchInfo>& matches, cv::Mat& group1, cv::Mat& group2);

	cv::Mat PSM(cv::Mat& affinity, vector<MatchInfo>& matches, cv::Mat& group1, cv::Mat& group2);

	cv::Mat GNCCP_APE(cv::Mat& affinity, vector<MatchInfo>& matches, cv::Mat& group1, cv::Mat& group2);



	cv::Mat normalize(cv::Mat& x, cv::Mat& group);
	cv::Mat biNormalize(cv::Mat& x, vector<MatchInfo>& matches, int n1, int n2, int nIter);
	cv::Mat greedyMapping(cv::Mat& x, vector<MatchInfo>& matches);

protected:
	int GMA::AdaptiveStep(cv::Mat& x, cv::Mat& x_est, int sLen);
	double Obj_GNCCP(cv::Mat& x, cv::Mat& K, double gamma);
	cv::Mat FW_GNCCP(cv::Mat& x, cv::Mat& K, vector<MatchInfo>& matches, cv::Mat& group1, cv::Mat& group2, double gamma);
	cv::Mat GMA::lineSearch(cv::Mat& y, cv::Mat& x, cv::Mat& K, double gamma, double eta);
	cv::Mat GMA::EstimateByTangent(vector<cv::Mat>& x_store, vector<int>& sLen_store, bool bRefine);

protected:
	const double DBL_MINVAL = 1.0e-15;
};

