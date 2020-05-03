#ifndef TEMPLATE_MATCHING_H
#include <opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

class pattern {
private:
	Mat ptrn;
	void _ncc_pattern_training(const Mat& pattern, int factor[], Size orgsize);
public:
	int level;
	float angle_start, angle_range;
	bool subpixel;
	vector<vector<int>> anglediv;
	vector<Mat> pyr_ptrn;
	struct 
	{
		vector<vector<int*>> factor;
		vector<vector<Size>> size;
	} ncc_factor;
	

	void ncc_pattern_training(int level = 1, float angle_start = 0, float angle_range = 0);
	pattern(const Mat src);
	~pattern();
};
void rotate_img(const Mat& src, Mat& dst, float angle_d);
void ncc_pattern_matching(const Mat& src, vector<vector<float>>& result,const pattern& ptrn,bool subpixel);

#endif // !TEMPLATE_MATCHING_H

