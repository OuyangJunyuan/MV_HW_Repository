#ifndef MYFILTER_H
#include <opencv2/opencv.hpp>
/*********************
 *@author:欧阳俊源
 *@date:2020/04/18
 *机器视觉编程作业
 ********************/
using namespace std;
using namespace cv;


 enum class PADDING_METHOD{
	MY_PADDING_COPY = 0,
	MY_PADDING_ZEROS,
};
 enum class MEAD_FILTER_METHOD {
     MY_FILTER_METHOD_OPENCV=0,
     MY_FILTER_METHOD_NORMAL,
     MY_FILTER_METHOD_SEPARATION,
     MY_FILTER_METHOD_RECURSION,
 };

 void padding(const Mat &src, Mat &dst, PADDING_METHOD method,Size boundsize);

 void mean_filter(const Mat& src, Mat& dst, Size kernel, MEAD_FILTER_METHOD method);
 void mean_filter_normal(const Mat& src, Mat& dst, Size kernel);
 void mean_filter_separation(const Mat& src, Mat& dst, Size kernel);
 void mean_filter_recuration(const Mat& src, Mat& dst, Size kernel);




#endif // !MYFILTER_H
