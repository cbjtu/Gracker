#pragma once
#include "common.h"

#ifdef _MSC_VER
// Todo: find the equivalent to may_alias
#define UCHAR_ALIAS unsigned char //__declspec(noalias)
#define UINT32_ALIAS unsigned int //__declspec(noalias)
#define __inline__ __forceinline
#endif


struct BRISK_D {
	unsigned char desc[64];
public:
	BRISK_D(){ memset(desc, 0, sizeof(desc)); }

	inline size_t bit_cnt(unsigned int value) const
	{
		value = (value & 0x55555555) + ((value >> 1) & 0x55555555);
		value = (value & 0x33333333) + ((value >> 2) & 0x33333333);
		value = (value & 0x0f0f0f0f) + ((value >> 4) & 0x0f0f0f0f);
		value = (value & 0x00ff00ff) + ((value >> 8) & 0x00ff00ff);
		value = (value & 0x0000ffff) + ((value >> 16) & 0x0000ffff);

		return (size_t)value;
	}

	size_t HammingDist(const BRISK_D& B) const{
		size_t dist = 0;

		unsigned int* p1 = (unsigned int*)desc;
		unsigned int* p2 = (unsigned int*)B.desc;
		for (int i = 0; i < 64 / sizeof(unsigned int); i++) {
			unsigned int val = (*p1) ^ (*p2);
			dist += bit_cnt(val);

			p1++;
			p2++;
		}

		return dist;
	}
};


class Brisker
{
public:
	// some helper structures for the Brisk pattern representation
	struct BriskPatternPoint{
		float x;         // x coordinate relative to center
		float y;         // x coordinate relative to center
		float sigma;     // Gaussian smoothing sigma
	};
	struct BriskShortPair{
		unsigned int i;  // index of the first pattern point
		unsigned int j;  // index of other pattern point
	};
	struct BriskLongPair{
		unsigned int i;  // index of the first pattern point
		unsigned int j;  // index of other pattern point
		int weighted_dx; // 1024.0/dx
		int weighted_dy; // 1024.0/dy
	};


public:
	Brisker(bool rotationInvariant = true, bool scaleInvariant = false, float patternScale = 1.0f);
	~Brisker();

	int calcPointBrisk(const cv::Mat& image, const vector<cv::Point2i>& vecPoints, vector<cv::KeyPoint>& keypoints,
		vector<BRISK_D>& descriptors);

	int calcPointBrisk(const cv::Mat& image, const cv::Rect rect, vector<vector<cv::KeyPoint>>& keypoints, vector<vector<BRISK_D>>& descriptors);
	int CalcMaximumBrisk(const cv::Mat& img, cv::Rect rect, int nGrid, vector<cv::KeyPoint>& keypoints, vector<BRISK_D>& descriptors);


protected:

	void countBlockMag(double* mag, int* ori, int len, int div, double* magcounts);

	void calcPointDesc(const Mat& image, const Mat& _integral, cv::KeyPoint& kp, uchar desc[]);

	void calcPointMag(const cv::Mat& img, cv::Point2i pt, double & mag, int & ori);

	// call this to generate the kernel:
	// circle of radius r (pixels), with n points;
	// short pairings with dMax, long pairings with dMin
	void generateKernel(std::vector<float> &radiusList,
		std::vector<int> &numberList, float dMax = 5.85f, float dMin = 8.2f,
		std::vector<int> indexChange = std::vector<int>());

	bool RoiPredicate(const float minX, const float minY,
		const float maxX, const float maxY, const KeyPoint& keyPt);

	inline int smoothedIntensity(const cv::Mat& image,
		const cv::Mat& integral, const float key_x,
		const float key_y, const unsigned int scale,
		const unsigned int rot, const unsigned int point) const;


protected:
	bool rotationInvariance;
	bool scaleInvariance;

	// pattern properties
	BriskPatternPoint* patternPoints_; 	//[i][rotation][scale]
	unsigned int points_; 				// total number of collocation points
	float* scaleList_; 					// lists the scaling per scale index [scale]
	unsigned int* sizeList_; 			// lists the total pattern size per scale index [scale]
	static const unsigned int scales_;	// scales discretization
	static const float scalerange_; 	// span of sizes 40->4 Octaves - else, this needs to be adjusted...
	static const unsigned int n_rot_;	// discretization of the rotation look-up


	// pairs
	int strings_;						// number of uchars the descriptor consists of
	float dMax_; 						// short pair maximum distance
	float dMin_; 						// long pair maximum distance
	BriskShortPair* shortPairs_; 		// d<_dMax
	BriskLongPair* longPairs_; 			// d>_dMin
	unsigned int noShortPairs_; 		// number of shortParis
	unsigned int noLongPairs_; 			// number of longParis

	// general
	static const float basicSize_;

};

