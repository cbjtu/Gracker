#pragma once
#include "common.h"
#include <math.h>

namespace planimetry {

	cv::Point2f rotatePoint(cv::Point2f pt, cv::Point2f center, double angle);

	inline double getDistance(cv::Point2f pt1, cv::Point2f pt2) {
		return sqrt((pt1.x - pt2.x) * (pt1.x - pt2.x) + (pt1.y - pt2.y) * (pt1.y - pt2.y));
	}

	//类定义：二维向量
	class Vector2d
	{
	public:
		double x_;
		double y_;

	public:
		Vector2d(double x, double y) :x_(x), y_(y){}
		Vector2d() :x_(0), y_(0){}

		//二维向量叉乘, 叉乘的结果其实是向量，方向垂直于两个向量组成的平面，这里我们只需要其大小和方向
		double crossProduct(const Vector2d vec)
		{
			return x_*vec.y_ - y_*vec.x_;
		}

		double dotProduct(const Vector2d vec)
		{
			return x_ * vec.x_ + y_ * vec.y_;
		}

		//二维向量减法
		Vector2d minus(const Vector2d vec) const
		{
			return Vector2d(x_ - vec.x_, y_ - vec.y_);
		}

		//判断点M,N是否在直线AB的同一侧
		static bool isPointAtSameSideOfLine(const Vector2d &pointM, const Vector2d &pointN,
			const Vector2d &pointA, const Vector2d &pointB)
		{
			Vector2d AB = pointB.minus(pointA);
			Vector2d AM = pointM.minus(pointA);
			Vector2d AN = pointN.minus(pointA);

			//等于0时表示某个点在直线上
			return AB.crossProduct(AM) * AB.crossProduct(AN) >= 0;
		}
	};

	//三角形类
	class Triangle
	{
	private:
		Vector2d pointA_, pointB_, pointC_;

	public:
		Triangle(Vector2d point1, Vector2d point2, Vector2d point3)
			:pointA_(point1), pointB_(point2), pointC_(point3)
		{
			//todo 判断三点是否共线
		}

		//计算三角形面积
		double computeTriangleArea()
		{
			//依据两个向量的叉乘来计算，可参考http://blog.csdn.net/zxj1988/article/details/6260576
			Vector2d AB = pointB_.minus(pointA_);
			Vector2d BC = pointC_.minus(pointB_);
			return fabs(AB.crossProduct(BC) / 2.0);
		}

		bool isPointInTriangle1(const Vector2d pointP)
		{
			double area_ABC = computeTriangleArea();
			double area_PAB = Triangle(pointP, pointA_, pointB_).computeTriangleArea();
			double area_PAC = Triangle(pointP, pointA_, pointC_).computeTriangleArea();
			double area_PBC = Triangle(pointP, pointB_, pointC_).computeTriangleArea();

			if (fabs(area_PAB + area_PBC + area_PAC - area_ABC) < 0.000001)
				return true;
			else return false;
		}

		bool isPointInTriangle2(const Vector2d pointP)
		{
			return Vector2d::isPointAtSameSideOfLine(pointP, pointA_, pointB_, pointC_) &&
				Vector2d::isPointAtSameSideOfLine(pointP, pointB_, pointA_, pointC_) &&
				Vector2d::isPointAtSameSideOfLine(pointP, pointC_, pointA_, pointB_);
		}

		bool isPointInTriangle3(const Vector2d pointP)
		{
			Vector2d AB = pointB_.minus(pointA_);
			Vector2d AC = pointC_.minus(pointA_);
			Vector2d AP = pointP.minus(pointA_);
			double dot_ac_ac = AC.dotProduct(AC);
			double dot_ac_ab = AC.dotProduct(AB);
			double dot_ac_ap = AC.dotProduct(AP);
			double dot_ab_ab = AB.dotProduct(AB);
			double dot_ab_ap = AB.dotProduct(AP);

			double tmp = 1.0 / (dot_ac_ac * dot_ab_ab - dot_ac_ab * dot_ac_ab);

			double u = (dot_ab_ab * dot_ac_ap - dot_ac_ab * dot_ab_ap) * tmp;
			if (u < 0 || u > 1)
				return false;
			double v = (dot_ac_ac * dot_ab_ap - dot_ac_ab * dot_ac_ap) * tmp;
			if (v < 0 || v > 1)
				return false;

			return u + v <= 1;
		}
	};



	class Polygon
	{
	public:
		Polygon(vector<cv::Point2f>& corner);
		Polygon(vector<cv::Point2i>& corner);
		~Polygon();

		bool isPointInPolygon(const Vector2d pointP);
		double calcOverlapRate(Polygon& R);
		double getArea();

	protected:
		vector<Vector2d>  corners;
	};

};