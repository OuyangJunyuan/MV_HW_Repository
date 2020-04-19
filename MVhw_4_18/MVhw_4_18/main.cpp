/*********************
 *@author:欧阳俊源 
 *@date:2020/0419
 *机器视觉图像特征提取与计算作业
 ********************/
#include <opencv2/opencv.hpp>
#include <iostream>
#include "RegionFeatures.h"
using namespace cv;
using namespace std;


void app(const Mat& src, vector<vector<int>>& _circle)
{
	double begintime = getTickCount();
	vector<vector<Point3i>> comp;
	vector<Point> comp_center;
	vector<int> comp_area;
	vector<vector<int>> comp_dmax;
	vector<float> comp_cicurlarity;
	Mat copy, gray, binary;

	_circle.clear();

	
	cvtColor(src, gray, COLOR_RGB2GRAY);
	mythreshold(gray, binary, 120, THRESHOLD_TYPE::THRESHOLD_BINARY);
	get_component(binary, comp, 123); /* 得到连通域，以行程为单元存放在一个集合中 */

	/* 对连通区域进行特征提取 */
	for (int i = 0; i < comp.size(); i++)
	{
		/* 计算连通域重心与面积 */
		comp_area.push_back(0);
		comp_center.push_back(component_center(comp[i], &comp_area[i]));


		/* 计算连通域最大直径 */
		comp_dmax.push_back(component_dmax(comp[i]));

		/* 计算连通域圆性 */
		comp_cicurlarity.push_back(component_circularity(comp[i], comp_dmax[i][4], comp_area[i]));
		if (comp_cicurlarity[i] > 0.8)
		{
			int r = sqrtf(comp_cicurlarity[i] * comp_dmax[i][4] * comp_dmax[i][4] / 4);
			_circle.push_back(vector<int>() = { comp_center[i].x,comp_center[i].y,comp_area[i],r });
		}
	}
	cout << "cost:"<<((getTickCount() - begintime) / getTickFrequency())*1000<<"ms" << endl;

	display_component("component_segmentation", gray, copy, comp);
	cvtColor(copy, copy, COLOR_GRAY2BGR);
	for (int i = 0; i < comp_center.size(); i++)
		circle(copy, comp_center[i], 3, Scalar(0, 0, 255), 3);
	for (int i = 0; i < _circle.size(); i++)
		circle(copy, Point(_circle[i][0], _circle[i][1]),_circle[i][3], Scalar(0, 255, 0), 2);

	imshow("comp", copy);
}

int main(void)
{
	while (1)
	{
		Mat srcImage = imread("exp.bmp");
		vector<vector<int>> circles;

		cout << "-----------------------" << endl;
		cout << "timestamp:" << getTickCount << endl;

		app(srcImage,circles);
		for (int i = 0; i < circles.size(); i++)
		{
			cout << "circle" << i << ": pos(" << circles[i][0] << "," << circles[i][1] << ") area("
				<< circles[i][2] <<") radius(" << circles[i][3] <<")"<< endl;
		}
		

		waitKey(1);
	}
	return 0;
}

