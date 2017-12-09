#include "Polygon.h"

using namespace planimetry;

cv::Point2f planimetry::rotatePoint(cv::Point2f pt, cv::Point2f center, double angle)
{
	double ang = cv::fastAtan2(pt.y - center.y, pt.x - center.x);
	double len = sqrt((pt.y - center.y) * (pt.y - center.y) + (pt.x - center.x) * (pt.x - center.x));
	ang += angle;
	
	double radian = ang * CV_PI / 180.0;
	double cosv = cos(radian);
	double sinv = sin(radian);


	double x = len * cos(radian) + center.x;
	double y = len * sin(radian) + center.y;
	
	return cv::Point2f((float)x, (float)y);
}

Polygon::Polygon(vector<cv::Point2f>& corner)
{
	corners.clear();
	for (int i = 0; i < corner.size(); i++) {
		corners.push_back(Vector2d(corner[i].x, corner[i].y));
	}
}

Polygon::Polygon(vector<cv::Point2i>& corner)
{
	corners.clear();
	for (int i = 0; i < corner.size(); i++) {
		corners.push_back(Vector2d(corner[i].x, corner[i].y));
	}
}

Polygon::~Polygon()
{
}

bool Polygon::isPointInPolygon(const Vector2d pointP)
{
	bool bIn = false;
	for (int i = 0; i < corners.size() - 2; i++) {
		Vector2d a = corners[0];
		Vector2d b = corners[i + 1];
		Vector2d c = corners[i + 2];
		Triangle tri(a, b, c);
		if (tri.isPointInTriangle2(pointP)) {
			bIn = true;
			break;
		}
	}
	return bIn;
}


double Polygon::calcOverlapRate(Polygon& R)
{
	double left1 = 1e6, top1 = 1e6;
	double right1 = -1e6, bottom1 = -1e6;
	for (int i = 0; i < corners.size(); i++) {
		left1 = min(left1, corners[i].x_);
		right1 = max(right1, corners[i].x_);
		top1 = min(top1, corners[i].y_);
		bottom1 = max(bottom1, corners[i].y_);
	}

	double left2 = 1e6, top2 = 1e6;
	double right2 = -1e6, bottom2 = -1e6;
	for (int i = 0; i < R.corners.size(); i++) {
		left2 = min(left2, R.corners[i].x_);
		right2 = max(right2, R.corners[i].x_);
		top2 = min(top2, R.corners[i].y_);
		bottom2 = max(bottom2, R.corners[i].y_);
	}


	if (left1 > right2 || right1 < left2)
		return 0.0;
	if (top1 > bottom2 || bottom1 < top2)
		return 0.0;

	int left = min(left1, left2);
	int right = max(right1, right2);
	int top = min(top1, top2);
	int bottom = max(bottom1, bottom2);

	int andNum = 0, orNum = 0;
	for (int x = left; x <= right; x++) {
		for (int y = top; y <= bottom; y++) {
			bool v1 = this->isPointInPolygon(Vector2d(x, y));
			bool v2 = R.isPointInPolygon(Vector2d(x, y));
			if (v1 && v2) {
				andNum++;
			}
			if (v1 || v2) {
				orNum++;
			}
		}
	}

	return (double)andNum / (double)orNum;
}

double Polygon::getArea()
{
	double area = 0.0;
	for (int i = 0; i < corners.size() - 2; i++) {
		Vector2d a = corners[0];
		Vector2d b = corners[i + 1];
		Vector2d c = corners[i + 2];
		Triangle tri(a, b, c);
		area += tri.computeTriangleArea();
	}

	return area;
}
