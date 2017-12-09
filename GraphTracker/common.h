#ifndef COMMON_H

#define COMMON_H

#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#ifdef _DEBUG
#pragma comment(lib, "opencv_core249d.lib")
#pragma comment(lib, "opencv_highgui249d.lib")
#pragma comment(lib, "opencv_imgproc249d.lib")
#pragma comment(lib, "opencv_contrib249d.lib")
#pragma comment(lib, "opencv_features2d249d.lib")
#pragma comment(lib, "opencv_calib3d249d.lib")  
#pragma comment(lib, "opencv_ml249d.lib")  
#pragma comment(lib, "opencv_video249d.lib")  
#pragma comment(lib, "opencv_nonfree249d.lib")  
#pragma comment(lib, "opencv_flann249d.lib")  
#pragma comment(lib, "opencv_gpu249d.lib")  
#pragma comment(lib, "opencv_objdetect249d.lib")  
#else
#pragma comment(lib, "opencv_core249.lib")
#pragma comment(lib, "opencv_highgui249.lib")
#pragma comment(lib, "opencv_imgproc249.lib")
#pragma comment(lib, "opencv_contrib249.lib")
#pragma comment(lib, "opencv_features2d249.lib")
#pragma comment(lib, "opencv_calib3d249.lib")  
#pragma comment(lib, "opencv_ml249.lib")  
#pragma comment(lib, "opencv_video249.lib")  
#pragma comment(lib, "opencv_nonfree249.lib")  
#pragma comment(lib, "opencv_flann249.lib")  
#pragma comment(lib, "opencv_gpu249.lib")  
#pragma comment(lib, "opencv_objdetect249.lib")  
#endif

using namespace std;
using namespace cv;



#define TYPE_SIFT (0)
#define TYPE_BRISK (1)

#define GRACK_DESC_TYPE		TYPE_SIFT

/*
using cv::Mat;
using cv::Point2f;
using cv::Rect;
using cv::Size2f;
using std::string;
using std::vector;
*/


#endif /* end of include guard: COMMON_H */
