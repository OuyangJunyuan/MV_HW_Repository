#include "template_matching.h"



pattern::pattern(const Mat src):subpixel(false),level(1),angle_start(0),angle_range(0)
{
	this->ptrn = src.clone();
	this->pyr_ptrn.push_back(this->ptrn);
}

pattern::~pattern()
{
	for (int i = 0; i < level; i++)
		for (int j = 0; j < anglediv[i].size(); ++j)
			delete[] this->ncc_factor.factor[i][j];
}

void pattern::_ncc_pattern_training(const Mat& pattern, int factor[], Size orgsize)
{
	/* 当前是使用0来表示非目标区域，之后可以考虑以下用通道2做标记 */
	int prows = pattern.rows, pcols = pattern.cols, pstep = pattern.step;
	int tmean = 0;
	float ts = 0; 

	for (int i = 0; i < prows; i++)
		for (int j = 0; j < pcols; j++)
			tmean += pattern.data[i * pstep + j];//非目标区域位0，加上不变。
	tmean /= orgsize.area();//由于旋转后图像面积变了，用原始图像作为元素个数。

	for (int i = 0; i < prows; i++){
		for (int j = 0; j < pcols; j++){
			if (pattern.data[i * pstep + j] == 0)
				continue;//非目标区域不做计算
			ts += (pattern.data[i * pstep + j] - tmean) * (pattern.data[i * pstep + j] - tmean);
		}
	}
	ts /= orgsize.area();//由于旋转后图像面积变了，用原始图像作为元素个数。
	ts = sqrtf(ts);

	for (int i = 0; i < prows; i++) {
		for (int j = 0; j < pcols; j++) {
			if (pattern.data[i * pstep + j] == 0)
				factor[i * pstep + j] = 0;//非目标区域因子为0，这也求CNN后不变。
			else 
				factor[i * pstep + j] = ((pattern.data[i * pstep + j] - tmean) << 16) / ts; //避免浮点运算 
		}
	}
}

void pattern::ncc_pattern_training(int level, float angle_start, float angle_range)
{
	this->level = level;
	this->angle_start = angle_start;
	this->angle_range = angle_range;

	this->anglediv = vector<vector<int>>(level);
	this->ncc_factor.factor = vector<vector<int*>>(level);
	this->ncc_factor.size = vector <vector<Size>>(level);

	for (int i = 0; i < level; i++)
	{
		int divstep = pow(2, i);
		for (int j = 0; j <= angle_range / divstep; j++)
		{
			anglediv[i].push_back(angle_start + j * divstep);
		}
	}
	
	Mat rotimg;
	for (int i = 0; i < level; i++)
	{
		if (i < level - 1)//倒数第第一层之前都要计算下采样。
		{
			(this->pyr_ptrn).push_back(Mat());
			pyrDown(this->pyr_ptrn[i], this->pyr_ptrn[i + 1], this->pyr_ptrn[i].size() / 2);
		}

		for (int j = 0; j < anglediv[i].size(); j++)//对第i层的所有角度进行离线训练
		{

			//每i层都进行多角度模板训练,模板金字塔只保留0角图像。
			rotate_img(this->pyr_ptrn[i], rotimg, anglediv[i][j]);
			//第i层第j个角度值的模板训练
			this->ncc_factor.factor[i].push_back(new int[rotimg.size().area()]);
			this->ncc_factor.size[i].push_back(rotimg.size());
			_ncc_pattern_training(rotimg, this->ncc_factor.factor[i][j], this->pyr_ptrn[i].size());
		}
	}
}

void rotate_img(const Mat& src, Mat& dst, float angle)
{
	float radian = angle / 180 * CV_PI;
	Point p1 = Point(src.cols * cosf(radian) - src.rows * sinf(radian), src.cols * sinf(radian) + src.rows * cosf(radian));
	Point p2 = Point(src.cols * cosf(radian) + src.rows * sinf(radian), src.cols * sinf(radian) - src.rows * cosf(radian));

	int mc = max(abs(p1.x) + 1, abs(p2.x) + 1), mr = max(abs(p1.y) + 1, abs(p2.y) + 1);
	int dx = round((mc - src.cols) / 2), dy = (mr - src.rows) / 2;
	
	copyMakeBorder(src, dst, dy, dy, dx, dx, BORDER_CONSTANT);
	Mat rot_mat = getRotationMatrix2D(Point(dst.cols / 2, dst.rows / 2), angle, 1.0);
	warpAffine(dst, dst, rot_mat, dst.size());
}

float scoring(const Mat& src,const int factor[],int orgsize,int pr,int pc,int r,int c)
{
	if ((r + pr > src.rows) || (c + pc > src.cols))
		return 0;

	int pstep = pc, sstep = src.step, row_end = r + pr, col_end = c + pc;

	int fmean = 0, fs2 = 0, score = 0;
	float fs = 0;
	for (int i = r; i < row_end; i++)
		for (int j = c; j < col_end; j++)
			fmean += src.data[i * sstep + j];
	fmean /= pr * pc;

	for (int i = r; i < row_end; i++)
		for (int j = c; j < col_end; j++)
			fs2 += (src.data[i * sstep + j] - fmean) * (src.data[i * sstep + j] - fmean);
	fs = sqrtf(((float)(fs2)) / (pr * pc));

	//计算分数
	for (int i = 0, k = r; i < pr; k++, i++)
		for (int j = 0, p = c; j < pc; p++, j++)
			score += ((factor[i * pstep + j]) * (src.data[k * sstep + p] - fmean)) >> 16;/* 避免浮点运算提高运算速度 */

	return score / (orgsize * fs);
}

void scoring_ROI(const Mat& src, Rect& ROI, const int factor[], int orgsize,Size psize,float* max, int* x, int* y)
{
	int rstart = ROI.tl().y, cstart = ROI.tl().x, step = src.step;
	int rend = rstart + ROI.height, cend = cstart + ROI.width;

	float temp=0;
	int pr = psize.height, pc = psize.width;

	int tx = 0, ty = 0;
	float tmax = 0;

	for (int i = rstart; i < rend; i++)
	{
		for (int j = cstart; j < cend; j++)
		{
			temp = scoring(src, factor,orgsize, pr, pc, i, j);
			if (temp > tmax)
			{
				tmax = temp;
				tx = j;
				ty = i;
			}
		}
	}
	*x = tx;*y = ty;*max = tmax;
}

Point2f subpixel_2d(const Point& center, double F9[])
{
	int cx = center.x, cy = center.y;
	int x[] = {0, cx - 1,cx,cx + 1 }, y[] = {0, cy - 1,cy,cy + 1 };
	double dA[]={x[1] * x[1],y[1] * y[1],x[1] * y[1],x[1],y[1],1,
				x[2] * x[2],y[1] * y[1],x[2] * y[1],x[2],y[1],1,
				x[3] * x[3],y[1] * y[1],x[3] * y[1],x[3],y[1],1,

				x[1] * x[1],y[2] * y[2],x[1] * y[2],x[1],y[2],1,
				x[2] * x[2],y[2] * y[2],x[2] * y[2],x[2],y[2],1,
				x[3] * x[3],y[2] * y[2],x[3] * y[2],x[3],y[2],1,

				x[1] * x[1],y[3] * y[3],x[1] * y[3],x[1],y[3],1,
				x[2] * x[2],y[3] * y[3],x[2] * y[3],x[2],y[3],1,
				x[3] * x[3],y[3] * y[3],x[3] * y[3],x[3],y[3],1 };

	Mat p, temp, A = Mat(9, 6, CV_64F, dA), F = Mat(9, 1, CV_64F, F9);

	invert(A.t()* A, temp);
	p = temp * A.t() * F;
	double * pd = (double *)(p.data);
	float den = pd[2] * pd[2] - 4 * pd[0] * pd[1];
	return Point2f((2 * pd[1] * pd[3] - pd[4] * pd[2]) / den, (-pd[2] * pd[3] + 2 * pd[0] * pd[4]) / den);
}
float subpixel_1d(int ca, double F3[])
{
	int x[] = { ca - 1,ca,ca + 1 };
	double dA[] = { x[0] * x[0],x[0],1,
					x[1] * x[1],x[1],1,
					x[2] * x[2],x[2],1 };

	Mat p, temp, A = Mat(3, 3, CV_64F, dA), F = Mat(3, 1, CV_64F, F3);
	invert(A, temp);
	p = temp * F;
	double* pd = (double*)(p.data);
	 
	return -pd[1]/(2.0*pd[0]) ;
}

void ncc_pattern_matching(const Mat& src, vector<vector<float>>& result,const pattern &ptrn,bool subpixel)
{
	int level = ptrn.level;
	vector<Mat> src_pyr(level, Mat());//待匹配图金字塔
	vector<Rect> src_pyr_ROI(level, Rect());//匹配过程设置的ROI，用来减小搜索范围
	vector<Point> angle_ROI(level, Point());//匹配过程设置的角度ROI，用来减小搜索范围

	src_pyr[0] = src;
	for (int j = 0; j < level - 1; j++)
		pyrDown(src_pyr[j], src_pyr[j + 1], src_pyr[j].size()/2);//计算金字塔

	int cx = -1, cy = -1,tcx=-1,tcy=-1,ca=-1,a1=0,a2=0;
	float cmax = -1,tmax=-1;

	src_pyr_ROI[level - 1] = Rect(Point(0,0),src_pyr[level-1].size());//从金字塔顶层开始匹配，且顶层是ROI为全图。
	angle_ROI[level - 1] = Point(0,ptrn.anglediv[level - 1].size() - 1);
	
	for (int i = level - 1; i >= 0; i--)//从顶层开始向下
	{
		cmax = -1;
		for (int j = angle_ROI[i].x; j <= angle_ROI[i].y; j++)
		{
			scoring_ROI(src_pyr[i], src_pyr_ROI[i], ptrn.ncc_factor.factor[i][j],ptrn.pyr_ptrn[i].size().area()
				,ptrn.ncc_factor.size[i][j], &tmax, &tcx, &tcy);
			if (tmax > cmax)
			{
				cmax = tmax;
				cx = tcx;
				cy = tcy;
				ca = j;
			}

		}
		if (i > 0)
		{
			src_pyr_ROI[i - 1] = (Rect(cx * 2 - 2, cy * 2 - 2, 5, 5));
			a1 = ca * 2 + 2 > (ptrn.anglediv[i - 1].size() - 1) ? ptrn.anglediv[i - 1].size() - 1 : ca * 2 + 2;
			a2 = (a1 - 5) < 0 ? 0 : a1 - 5;
			angle_ROI[i - 1] = Point(a2, a1);
		}
	}
	Point2f subpixel_p;
	float subangle=-1;
	if (subpixel) 
	{
		double score3x3[9];
		for (int i = -1; i <= 1; i++)
		{
			for (int j = -1; j <= 1; j++)
			{
				score3x3[(i+1)*3+j+1]=scoring(src_pyr[0], ptrn.ncc_factor.factor[0][ca], ptrn.pyr_ptrn[0].size().area(),
					ptrn.ncc_factor.size[0][ca].height, ptrn.ncc_factor.size[0][ca].width,cy+i, cx + j);
			}
		}
		subpixel_p=subpixel_2d(Point(cx, cy),score3x3);
		
		
		if (ca != 0 && ca != (ptrn.anglediv[0].size() - 1)) {
			double score3x1[3];
			for (int i = -1; i <= 1; i++)
			{
				score3x1[i + 1] = scoring(src_pyr[0], ptrn.ncc_factor.factor[0][ca + i], ptrn.pyr_ptrn[0].size().area(),
					ptrn.ncc_factor.size[0][ca+i].height, ptrn.ncc_factor.size[0][ca].width, cy, cx);
			}
			subangle=subpixel_1d(ptrn.anglediv[0][ca], score3x1);
		}
		result = vector<vector<float>>(1, vector<float>{subpixel_p.x, subpixel_p.y, subangle==-1? float(ptrn.anglediv[0][ca]):subangle, cmax});
		
	}else
		result=vector<vector<float>>(1, vector<float>{float(cx), float(cy), float(ptrn.anglediv[0][ca]),cmax});
}

