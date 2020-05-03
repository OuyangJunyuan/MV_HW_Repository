/*********************
 *@author:Å·Ñô¿¡Ô´
 *@date:
 *
 ********************/
#include <opencv2/opencv.hpp>
#include <iostream>
#include "template_matching.h"
using namespace cv;
using namespace std;


int main(void)
{
	Mat pattern = imread("./images/pattern.bmp"); // imread("./face.png");
	int64 t0, t1;
	cvtColor(pattern, pattern, COLOR_RGB2GRAY);

	class pattern ptrn(pattern);
	ptrn.ncc_pattern_training(5,-30,60);
	float meancost = 0;
	for (int i = 15; i <= 34; i++)
	{
		char imname[50];
		sprintf_s(imname, 50, "./images/IMAGEB%d.bmp", i);
		Mat srcImage = imread(imname),tptrn;
		cvtColor(srcImage, srcImage, COLOR_RGB2GRAY);

		vector<vector<float>> result;
		/* matching */
		t0 = getTickCount();
		ncc_pattern_matching(srcImage, result, ptrn,false);
		t1 = getTickCount();

	/* display results */
		rotate_img(pattern, tptrn, result[0][2]);
	
		float radian = result[0][2] / 180 * CV_PI;
		Point p1 = Point(pattern.cols * cosf(radian) - pattern.rows * sinf(radian), pattern.cols * sinf(radian) + pattern.rows * cosf(radian))/2;
		Point p2 = Point(pattern.cols * cosf(radian) + pattern.rows * sinf(radian), pattern.cols * sinf(radian) - pattern.rows * cosf(radian))/2;
		Point c = Point(result[0][0] + tptrn.cols / 2, result[0][1] + tptrn.rows / 2);
		

		circle(srcImage,c, 8, Scalar(255, 255, 255));
		Point pa = Point(c.x + p1.x, c.y - p1.y), pb = Point(c.x + p2.x, c.y - p2.y),
			pc = Point(c.x - p1.x, c.y + p1.y), pd = Point(c.x - p2.x, c.y + p2.y);
		line(srcImage, pa, pb, Scalar(255, 255, 255));
		line(srcImage, pb, pc, Scalar(255, 255, 255));
		line(srcImage, pc, pd, Scalar(255, 255, 255));
		line(srcImage, pd, pa, Scalar(255, 255, 255));
		line(srcImage, (pa + pd) / 2, (pc + pb) / 2, Scalar(255, 255, 255));
		line(srcImage, (pa + pb) / 2, (pc + pd) / 2, Scalar(255, 255, 255));
		imshow("matching_result", srcImage);

		float cost = (t1 - t0) / getTickFrequency() * 1000.0;
		meancost += cost;
		cout <<"-----------------------------"
			 <<"\nTimeStamp : "<<getTickCount
			 <<"\npicture : "<<imname
			 <<"\nX : " << result[0][0] 
			 <<"\nY : " << result[0][1] 
			 <<"\nTheta : " << result[0][2] 
			 <<"\nScore : " << result[0][3] 
			 <<"\nTimeCost:" << cost <<" ms" << endl;
		
		//while (waitKey() != 'q');
		waitKey(1);
	}
	cout << "average_cost" << meancost /(34-15)<<"ms" << endl;

	return 0;
}

