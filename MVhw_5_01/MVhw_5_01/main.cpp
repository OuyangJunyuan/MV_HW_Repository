#include <opencv2/opencv.hpp>
#include <iostream>
#include "measure.h"
using namespace cv;
using namespace std;

/*********************
 *@author:Å·Ñô¿¡Ô´ 
 *@date:
 *
 ********************/

int main(void)
{
	Mat srcImage = imread("brake_disk_part_06.png"), grayImage;
	cvtColor(srcImage, grayImage, COLOR_RGB2GRAY);
	resize(grayImage,grayImage ,Size(640, 512));
	namedWindow("pic");
	windows_mouse_combind("pic");

	vector<Point3i> circleset;
	int rt = 10, at = 8, tt = 10, threshold = 210;

	creat_setting_bar(&rt, &tt, &at, &threshold);
	while (1)
	{	
		
		
		long t0 = getTickCount();
		/*   Implementation algorithm    */
		CircleLocating(grayImage, circleset, threshold, 500, 0.8);
		CircleMeasuring(grayImage, circleset, rt, at, tt, true);
		/*   Implementation algorithm    */

		//cout << (getTickCount() - t0) / getTickFrequency()/1000<<" ms" << endl;
		waitKey(1);
	}
	return 0;
}

