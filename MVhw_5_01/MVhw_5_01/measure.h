#ifndef CIRCLE_FIND_H
#define CIRCLE_FIND_H
#include <opencv.hpp>
#include <iostream>
#include "RegionFeatures.h"
using namespace cv;
using namespace std;


extern int mouse_x, mouse_y;

void creat_setting_bar(int* rt, int* tt, int* at, int* threshold);
vector<Point3f> CircleMeasuring(const Mat& src, vector<Point3i> circleset, int rrate, int arate, int trrate, bool display);
void CircleLocating(const Mat& src, vector<Point3i>& _circle, int bt, int at, int ct);
unsigned char BilinearInterpolation(const Mat& src, float x, float y);
void windows_mouse_combind(string winname);

#endif
/*********************
 *@author:Å·Ñô¿¡Ô´
 *@date:
 *
 ********************/