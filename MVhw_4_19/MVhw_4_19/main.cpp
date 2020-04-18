#include <opencv2/opencv.hpp>
#include <iostream>
#include "myfilter.h"
using namespace cv;
using namespace std;

/*********************
 *@author:欧阳俊源 
 *@date:2020/04/18
 *机器视觉编程作业
 ********************/

int main(void)
{
	while (1)
	{
		Mat srcImage=imread("lena.jpg"),grayImage,meanImage;
		resize(srcImage, srcImage, Size(480,680));
		cvtColor(srcImage, grayImage,COLOR_RGB2GRAY);
		int size = 21;
		
		/*   Implementation algorithm    */
		cout << "-------------------------------" << endl;
		cout << "timestamp:" << getTickCount << endl;
		cout << "kernelsize:" << size << endl;

		

		mean_filter(grayImage, meanImage, Size(size, size), MEAD_FILTER_METHOD::MY_FILTER_METHOD_NORMAL);
		mean_filter(grayImage, meanImage, Size(size, size), MEAD_FILTER_METHOD::MY_FILTER_METHOD_SEPARATION);
		mean_filter(grayImage, meanImage, Size(size, size), MEAD_FILTER_METHOD::MY_FILTER_METHOD_RECURSION);
		mean_filter(grayImage, meanImage, Size(size, size), MEAD_FILTER_METHOD::MY_FILTER_METHOD_OPENCV);

		/*   Implementation algorithm    */
		
		imshow("image", meanImage);
		waitKey(1);
	}
	return 0;
}

