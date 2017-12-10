// GraphTracker.cpp : 定义控制台应用程序的入口点。
//

#include "common.h"

#include <iostream>
#include <fstream>
#include <thread>
#include <direct.h>

#include "Video.h"
#include "Gracker.h"

struct TResult
{
	double rate;		// average accuracy
	double variantion;	// standard variantion
	int frames;
	clock_t clocks;
};

static const int kPatchWidth = 150;
static const int kPatchHeight = 150;
static const char* kWindowName = "Gracker";

static bool bEvaluation = false;
static bool bSaveResult = false;
static bool bQuietMode = false;
static int  nLineWidth = 3;

double calcOverlappingRate(int imgW, int imgH, Warpping& w, vector<cv::Point2f>& initPoints, vector<cv::Point2f>& gtPoints);
double calcOverlappingRate(int imgW, int imgH, cv::Mat& H, vector<cv::Point2f>& initPoints, vector<cv::Point2f>& gtPoints);
TResult processVideo(GraConfig& grac);


void reviseImgFileName(const char* filename)
{
	char outname[256];
	sprintf(outname, "%s.txt", filename);
	ifstream fin(filename);
	ofstream fout(outname);
	if (!fin || !fout)
		return;

	char buf[256];
	int lines = 0;
	while (!fin.eof()) {
		memset(buf, 0, sizeof(buf));
		fin.getline(buf, 255);
		char *p = strstr(buf, "frame");
		if (!p) {
			fout << buf << endl;
		}
		else {
			char* p2 = strstr(p, ".jpg");
			if (!p2) {
				fout << buf << endl;
			}
			else {
				int id = atoi(p + 5);
				char imgname[32];
				sprintf(imgname, "frame%05d.jpg", id);

				p2 += 4;
				fout << imgname << p2 << endl;
			}
		}
	}

	fin.close(); 
	fout.close();

}

int main(int argc, char* argv[]){
	clock_t ts = clock();
	bQuietMode = true;
	namedWindow(kWindowName);
	cvMoveWindow(kWindowName, 640, 20);
	bEvaluation = false;
	GraConfig grac;
	processVideo(grac);
	cout << "---";
	clock_t tf = clock();
	cout << endl << (tf - ts) / CLOCKS_PER_SEC << endl;
	getchar();
	return 0;
}

/*int main(int argc, char* argv[])
{
	clock_t ts = clock();
	if (argc < 3) {
		cout << "Command Format: GraphTracker [VideoName] [ShowMode]" << endl;
		cout << "Sample: GraphTracker nl_newspaper 1" << endl;
		getchar();
		return EXIT_SUCCESS;
	}

	char* video = argv[1];//video name
	bQuietMode = atoi(argv[2]);
	//bQuietMode = false;
	if (bQuietMode)//real-time image
	{
		namedWindow(kWindowName);
		cvMoveWindow(kWindowName, 640, 20);
	}

	bEvaluation = true;//Accuracy at real-time

	GraConfig grac;
	processVideo(string(video), grac);

	clock_t tf = clock();
	cout << endl << (tf - ts) / CLOCKS_PER_SEC << endl;
	getchar();
	return EXIT_SUCCESS;
}*/



double calcOverlappingRate(int imgW, int imgH, vector<cv::Point2f>& warpedPoints, vector<cv::Point2f>& gtPoints)
{
	for (int i = 0; i < warpedPoints.size(); i++) {
		// refine the border to reduce the computation time of overlapping area
		if (warpedPoints[i].x < -0.5 * imgW || warpedPoints[i].x > 1.5 * imgW)
			return 0.0;
		if (warpedPoints[i].y < -0.5 * imgH || warpedPoints[i].y > 1.5 * imgH)
			return 0.0;
	}
	
	planimetry::Polygon R1(warpedPoints), R2(gtPoints);


	if (R1.getArea() <= 1e-3 || R2.getArea() <= 1e-3)
		return 0;

	return R1.calcOverlapRate(R2);
}

double calcOverlappingRate(int imgW, int imgH, Warpping& w, vector<cv::Point2f>& initPoints, vector<cv::Point2f>& gtPoints)
{
	if (initPoints.size() < 4 || gtPoints.size() < 4 || initPoints.size() != gtPoints.size())
		return 0.0;

	planimetry::Vector2d corInit[4], corGt[4];
	vector<cv::Point2f> dstPoints;
	for (int i = 0; i < initPoints.size(); i++) {
		dstPoints.push_back(w.doWarpping(initPoints[i]));
	}

	return calcOverlappingRate(imgW, imgH, dstPoints, gtPoints);
}


double calcCornerDistance(vector<cv::Point2f>& warpedPoints, vector<cv::Point2f>& gtPoints)
{
	double dist = 0.0;
	for (int i = 0; i < warpedPoints.size(); i++) {
		double x1 = warpedPoints[i].x;
		double y1 = warpedPoints[i].y;
		double x2 = gtPoints[i].x;
		double y2 = gtPoints[i].y;

		dist += sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
	}
	return dist / warpedPoints.size();
}

vector<float> parseDigit(const char* buf)
{
	const char* p = buf;

	// move to first digit
	while (*p && !isdigit(*p))
		p++;

	vector<float> vals;
	while (*p) {
		float v = atof(p);
		vals.push_back(v);

		// skip digits for v;
		while (isdigit(*p) || *p == '.')
			p++;

		// move to next digit
		while (*p && !isdigit(*p))
			p++;
	}

	return vals;
}

IplImage *src = 0;
vector<Point2f> initPointProducer;
Video* pVideoInput = 0;
int endflag = 0;
void on_mouse(int event, int x, int y, int flags, void *ustc){
	CvFont font;
	cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 0.5, 0.5, 0, 1, CV_AA);
	if (event == CV_EVENT_LBUTTONDOWN){
		CvPoint pt = cvPoint(x, y);
		//cout << x << ' ' << y << endl;
		initPointProducer.push_back(Point2f((float)x, (float)y));
		char temp[16];
		sprintf_s(temp, "(%d,%d)", pt.x, pt.y);
		cvPutText(src,temp, pt, &font, cvScalar(255, 255, 255, 0));
		cvCircle(src, pt, 2, cvScalar(0, 0, 255, 0), CV_FILLED, CV_AA, 0);
		cvShowImage("src", src);
		endflag++;
		/*if (endflag == 4){
			cvDestroyWindow("src");
			cvReleaseImage(&src);
		}*/
	}
}

vector<Point2f> getPointsMouse(){
	initPointProducer.clear();
	endflag = 0;
	Mat InitMat;
	pVideoInput->ReadFrame(InitMat);
	src = new IplImage(InitMat);
	cvNamedWindow("src", 1);
	cvShowImage("src", src);
	cvSetMouseCallback("src", on_mouse, 0);
	cvWaitKey(0);
	cvDestroyWindow("src");
	for (int i = 0; i < initPointProducer.size(); i++) cout << initPointProducer[i] << endl;
	return initPointProducer;
}

TResult processVideo(GraConfig& grac) 
{
	VideoWriter writer;
	vector<Point2f> initPoints, warpSrc;
	pVideoInput = new Video(0, 0, 1);

	/*Mat show;
	while (pVideoInput->isVideoCapOpen()){
		pVideoInput->ReadFrame(show);
		imshow("test", show);
		if (waitKey(20)) break;
	}
	cout << "exit" << endl;*/

	Gracker GM(grac);
	GM.setQuietMode(bQuietMode);
	Mat frame, frameGrey;

	int frameIdx = 0;
	for (frameIdx = 0;; ++frameIdx){
		bool ok = false;
		ok = pVideoInput->ReadFrame(frame);
		if (!ok) break;
		if (frame.channels() > 1) cv::cvtColor(frame, frameGrey, CV_RGB2GRAY);
		else frame.copyTo(frameGrey);

		int imgW = frameGrey.cols;
		int imgH = frameGrey.rows;

		if (frameIdx == 0){
			vector<Point2f> gtPoints = getPointsMouse();
			initPoints.clear();
			initPoints.assign(gtPoints.begin(), gtPoints.end());

			int left = min(initPoints[0].x, min(initPoints[1].x, min(initPoints[2].x, initPoints[3].x)));
			int right = max(initPoints[0].x, max(initPoints[1].x, max(initPoints[2].x, initPoints[3].x)));
			int top = min(initPoints[0].y, min(initPoints[1].y, min(initPoints[2].y, initPoints[3].y)));
			int bottom = max(initPoints[0].y, max(initPoints[1].y, max(initPoints[2].y, initPoints[3].y)));

			warpSrc.assign(initPoints.begin(), initPoints.end());
			for (int k = 0; k < warpSrc.size(); k++){
				warpSrc[k].x -= left;
				warpSrc[k].y -= top;
			}

			GM.initModel(frameGrey, initPoints);
		}
		else{
			vector<Point2f> warpedPoints;
			Warpping w = GM.processFrame(frameGrey);
			warpedPoints = w.doWarpping(GM.getInitPolygon());

			if (bQuietMode){
				Mat show;
				show = GM.getDebugImage().clone();
				imshow(kWindowName, show);
				int key = waitKey(1);
			}
		}
	}

	if (writer.isOpened()) writer.release();

	if (pVideoInput) delete pVideoInput;

	return TResult();
}

/*
void testUCSB(const char* algo, Config& conf, bool bMultithread)
{
	char* result_path = "./save/UCSB/";
	char* database = "UCSB";

	std::string path = "d:/video/UCSB/";
	std::string images[6] = { "bricks", "building", "mission", "paris", "sunset", "wood" };
	std::string motions[16] = { "dynamic_lighting", "static_lighting", "motion1", "motion2", "motion3", "motion4", "motion5",
		"motion6", "motion7", "motion8", "motion9", "panning", "perspective", "rotation", "zoom", "unconstrained" };

	std::string classes[8] = { "lighting", "motion", "panning", "perspective", "rotation", "zoom", "unconstrained", "Total" };

		int counts[8];
		double rates[8];
		memset(counts, 0, sizeof(counts));
		memset(rates, 0, sizeof(rates));

		int nTotalFrame = 0;
		clock_t nTotalClock = 0;

		char file[256];
		sprintf(file, "%s_UCSB.txt", algo);
		ofstream fout(file);
		if (!fout) {
			cout << "cannot open file to write!" << endl;
			return;
		}

		GraConfig grac;
	//	grac.bPredictShape = false;
	//	grac.bUpdateDesc = false;
	//	grac.nWarppingMode = WARPPINGMODE_PERSPECTIVE;
	//	grac.dPredWeight = 0.1;
	//	grac.dPointAppTol = 0.7;
	//	grac.dDetectAppTol = 0.7;
		fout << grac;

		for (int i = 0; i < 6; i++) {

			vector<Config> vecConf;
			vecConf.assign(15, conf);

			std::vector<std::thread> threads;

			for (int k = 0; k < 15; k++) {
				char video_path[256], video[256];
				sprintf(video, "%s_%s", images[i].c_str(), motions[k].c_str());
				sprintf(video_path, "%s%s_%s", path.c_str(), images[i].c_str(), motions[k].c_str());
				vecConf[k].videoPath = video_path;
				vecConf[k].videoName = video;
				vecConf[k].useVideo = 0;

				if (bMultithread) {
					vecConf[k].quietMode = 1;
					threads.push_back(std::thread(processVideo, database, algo, &vecConf[k], grac));
				}
				else {
				//	vecConf[k].quietMode = 1;
					TResult ret = processVideo(database, algo, &vecConf[k], grac);
					nTotalFrame += ret.frames;
					nTotalClock += ret.clocks;
				}
			}

			if (bMultithread) {
				std::for_each(threads.begin(), threads.end(),
					std::mem_fn(&std::thread::join)); //调用它们的join方法
			}

			for (int k = 0; k < 15; k++) {
				double rate = vecConf[k].rate;
				fout << vecConf[k].videoPath.c_str() << "\t" << rate << endl;

				if (k < 2) {	// lighting
					rates[0] += rate;
					counts[0] ++;
				}
				else if (k < 11) {  // motion with different blur level
					rates[1] += rate;
					counts[1] ++;
				}
				else if (k == 11) {  // panning
					rates[2] += rate;
					counts[2] ++;
				}
				else if (k == 12) { // perspective
					rates[3] += rate;
					counts[3] ++;
				}
				else if (k == 13) { // rotation
					rates[4] += rate;
					counts[4]++;
				}
				else if (k == 14) { // zoom
					rates[5] += rate;
					counts[5] ++;
				}

				rates[7] += rate;
				counts[7] ++;

			}
		}

		// process unconstrained
		vector<Config> vecConf;
		vecConf.assign(6, conf);
		std::vector<std::thread> threads;
		for (int i = 0; i < 6; i++) {
			char video_path[256], video[256];
			sprintf(video, "%s_%s", images[i].c_str(), motions[15].c_str());
			sprintf(video_path, "%s%s_%s", path.c_str(), images[i].c_str(), motions[15].c_str());
			vecConf[i].videoPath = video_path;
			vecConf[i].videoName = video;
			vecConf[i].useVideo = 0;
			if (bMultithread) {
				vecConf[i].quietMode = 1;
				threads.push_back(std::thread(processVideo, database, algo, &vecConf[i], grac));
			}
			else {
			//	vecConf[i].quietMode = 1;
				TResult ret = processVideo(database, algo, &vecConf[i], grac);
				nTotalFrame += ret.frames;
				nTotalClock += ret.clocks;
			}
		}
		if (bMultithread) {
			std::for_each(threads.begin(), threads.end(),
				std::mem_fn(&std::thread::join)); //调用它们的join方法
		}
		for (int i = 0; i < 6; i++) {
			double rate = vecConf[i].rate;
			fout << vecConf[i].videoPath.c_str() << "\t" << rate << endl;
			// unconstrained
			rates[6] += rate;
			counts[6]++;

			rates[7] += rate;
			counts[7] ++;
		}



		fout << endl << endl;
		for (int i = 0; i < 8; i++) {
			rates[i] = rates[i] / (double)counts[i];
			char buf[128];
			sprintf(buf, "%s: \t%.4f", classes[i].c_str(), rates[i]);
			fout << buf << endl;
		}

		double tm = (double)nTotalClock / (double)nTotalFrame;
		tm = tm * 1.0 / CLOCKS_PER_SEC;

		fout << "Total Frame: " << nTotalFrame << ", Total Clock: " << nTotalClock << ", avg_time: " << tm << endl;

		fout.close();
}
*/

