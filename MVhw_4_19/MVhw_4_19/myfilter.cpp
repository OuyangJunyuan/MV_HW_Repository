#include "myfilter.h"
#include <stddef.h>
/*********************
*@author:欧阳俊源
*@date:2020/04/18
*机器视觉编程作业
********************/


void padding(const Mat& src, Mat& dst, PADDING_METHOD method, Size boundsize)
{
	boundsize.width = (boundsize.width - 1) / 2;
	boundsize.height = (boundsize.height - 1) / 2;
	int srows = src.rows, scols = src.cols;
	int drows = srows + boundsize.height * 2, dcols = scols + boundsize.width*2;

	Mat temp_dst = Mat(Size(dcols,drows), CV_8UC1);
	
	const uchar* sptr;
	uchar* dptr;
	int grayvalue;
	switch (method)
	{
	case PADDING_METHOD::MY_PADDING_COPY:
		for (int i = 0; i < drows; i++)
		{
			dptr = temp_dst.ptr<uchar>(i);
			if (i<boundsize.height) 
				sptr = src.ptr<uchar>(0);
			else if(i >= srows + boundsize.height)
				sptr = src.ptr<uchar>(srows-1);
			else
				sptr = src.ptr<uchar>(i- boundsize.height);
			
			for (int j = 0; j < dcols; j++)
			{
				if (j < boundsize.width)
					grayvalue = sptr[0];
				else if (j >= scols + boundsize.width)
					grayvalue = sptr[scols - 1];
				else
					grayvalue = sptr[j - boundsize.width];
				dptr[j] = grayvalue;
			}

		}

		break;
	case PADDING_METHOD::MY_PADDING_ZEROS:
		for (int i = boundsize.height; i < srows + boundsize.height; i++)
		{
			sptr = src.ptr<uchar>(i - boundsize.height);
			dptr = temp_dst.ptr<uchar>(i);
			for (int j = boundsize.width; j < scols + boundsize.width; j++)
			{
				dptr[j] = sptr[j - boundsize.width];
			}
		}
		break;
	default:
		break;

	}

	dst = temp_dst;
}


void mean_filter(const Mat& src, Mat& dst, Size kernel,MEAD_FILTER_METHOD method)
{
	
	double t1 = getTickCount(), cost;

	padding(src, dst, PADDING_METHOD::MY_PADDING_COPY, kernel);
	switch (method)
	{
	case MEAD_FILTER_METHOD::MY_FILTER_METHOD_OPENCV:
		blur(src, dst,kernel);
		cost = (getTickCount() - t1) / getTickFrequency();
		cout << "OPENCV" << ": " << cost * 1000 << "ms" << endl;
		break;
	case MEAD_FILTER_METHOD::MY_FILTER_METHOD_NORMAL:
		mean_filter_normal(src, dst, kernel);
		cost = (getTickCount() - t1) / getTickFrequency();
		cout << "NORMAL" << ": " <<cost * 1000 << "ms" << endl;
		break;

	case MEAD_FILTER_METHOD::MY_FILTER_METHOD_SEPARATION:
		mean_filter_separation(src, dst, kernel);
		cost = (getTickCount() - t1) / getTickFrequency();
		cout << "SEPARATION" << ": " << cost * 1000 << "ms" << endl;
		break;

	case MEAD_FILTER_METHOD::MY_FILTER_METHOD_RECURSION:
		mean_filter_recuration(src, dst, kernel);
		cost = (getTickCount() - t1) / getTickFrequency();
		cout << "RECURSION" << ": " << cost * 1000 << "ms" << endl;
		break;


	default:
		break;
	}
	
}
void mean_filter_normal(const Mat& src, Mat& dst, Size kernel)
{
	int area = kernel.area();
	kernel.width = (kernel.width - 1) / 2;
	kernel.height = (kernel.height - 1) / 2;
	int rows_begin = kernel.height , rows_end = src.rows - kernel.height;
	int cols_begin = kernel.width , cols_end = src.cols - kernel.width;
	Mat temp_dst = Mat(Size(src.cols - 2 * kernel.width, src.rows-2*kernel.height), CV_8UC1);


	uchar* dptr;
	int sum;

	/*  o(ijxy)  */
	for (int i = rows_begin; i < rows_end; i++)
	{
		dptr = temp_dst.ptr<uchar>(i-rows_begin);
		for (int j = cols_begin; j < cols_end; j++)
		{
			sum = 0;
			for (int x = -kernel.height; x <= kernel.height; x++)
			{
				for (int y = -kernel.width; y <= kernel.width; y++)
				{
					sum += src.ptr<uchar>(i + x)[j+y];
				}
			}
			dptr[j - cols_begin] = sum/ area;
		}
	}
	dst = temp_dst;
}
void mean_filter_separation(const Mat& src, Mat& dst, Size kernel)
{
	int area = kernel.area();
	kernel.width = (kernel.width - 1) / 2;
	kernel.height = (kernel.height - 1) / 2;

	int srows = src.rows, scols = src.cols, step = scols;
	int rows_begin = kernel.height, rows_end = srows - kernel.height,cols_begin = kernel.width, cols_end = scols - kernel.width;

	Mat temp_dst = Mat(Size(src.cols - 2 * kernel.width, src.rows - 2 * kernel.height), CV_8UC1);
	uint16_t* sumcols_dst = new uint16_t[srows * scols];
	uint16_t* sumrows_dst = new uint16_t[srows * scols];

	/*  o(ij(x+y))  */

	/*	 sum of cols	 */
	for (int i = rows_begin; i < rows_end; i++)
	{
		for (int j = 0; j < scols; j++)
		{
			sumcols_dst[i * step + j] = 0;
			sumrows_dst[i * step + j] = 0;
			for (int x = -kernel.height; x <= kernel.height; x++)
			{
				
				sumcols_dst[i * step + j] += src.ptr<uchar>(i + x)[j];
			}
		}
	}
	/*	 sum of rows	 */
	for (int i = rows_begin; i < rows_end; i++)
	{
		for (int j = cols_begin; j < cols_end; j++)
		{
			for (int y = -kernel.width; y <= kernel.width; y++)
			{
				sumrows_dst[i * step + j] += sumcols_dst[i * step + j + y];
			}
			temp_dst.ptr<uchar>(i-rows_begin)[j-cols_begin] = sumrows_dst[i * step + j]/ area;
		}
	}
	dst = temp_dst;
	delete[] sumrows_dst,sumcols_dst;
}

void mean_filter_recuration(const Mat& src, Mat& dst, Size kernel) 
{
	int area = kernel.area();
	kernel.width = (kernel.width - 1) / 2;
	kernel.height = (kernel.height - 1) / 2;

	int srows = src.rows, scols = src.cols, step = scols;
	int rows_begin = kernel.height, rows_end = srows - kernel.height, cols_begin = kernel.width, cols_end = scols - kernel.width;

	Mat temp_dst = Mat(Size(src.cols - 2 * kernel.width, src.rows - 2 * kernel.height), CV_8UC1);
	uint16_t* sumcols_dst = new uint16_t[srows * scols];
	uint16_t* sumrows_dst = new uint16_t[srows * scols];

	for (uint16_t i = 0; i < srows; i++)
		for (uint j = 0; j < scols; j++)
			sumcols_dst[i * step + j] = sumrows_dst[i * step + j] = 0;


	/*	 sum of cols	 */
	/*  init cols recuration base  */
	for (int j = 0; j < scols; j++)
		for (int x = -kernel.height; x <= kernel.height; x++)
			sumcols_dst[rows_begin * step + j] += src.ptr<uchar>(rows_begin + x)[j];


	for (int i = rows_begin + 1; i < rows_end; i++)
		for (int j = 0; j < scols; j++)
			sumcols_dst[i * step + j] = sumcols_dst[(i - 1) * step + j] + src.ptr<uchar>(i + kernel.height)[j] - src.ptr<uchar>(i - kernel.height - 1)[j];
	

	/*	 sum of rows	 */
	/*  init rows recuration base  */

	for (int i = rows_begin; i < rows_end; i++)
	{
		for (int y = -kernel.width; y <= kernel.width; y++)
		{
			sumrows_dst[i * step + cols_begin] += sumcols_dst[i * step + cols_begin + y];
		}
		temp_dst.ptr<uchar>(i - rows_begin)[0] = sumrows_dst[i * step + cols_begin] / area;
	}

	for (int i = rows_begin; i < rows_end; i++)
	{
		for (int j = cols_begin+1; j < cols_end; j++)
		{
			sumrows_dst[i * step + j] = sumrows_dst[i * step + j - 1] + sumcols_dst[i * step + j + kernel.width] - sumcols_dst[i * step + j - kernel.width - 1];
			temp_dst.ptr<uchar>(i - rows_begin)[j - cols_begin] = sumrows_dst[i * step + j] / area;
		}
	}
	//imshow("temp", temp_dst);
	dst = temp_dst;
	delete[] sumrows_dst, sumcols_dst;
}
