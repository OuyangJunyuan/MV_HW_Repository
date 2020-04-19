#include "RegionFeatures.h"
#include <algorithm>

void mythreshold(const Mat& src, Mat& dst, uchar threshold, THRESHOLD_TYPE type)
{
	int srows = src.rows, scols = src.cols;
	Mat temp_dst=Mat(Size(scols,srows),CV_8UC1);
	const uchar* sptr;
	uchar* dptr;
	switch (type)
	{
	case THRESHOLD_TYPE::THRESHOLD_BINARY:
		for (int i = 0; i < srows; i++)
		{
			sptr = src.ptr<uchar>(i);
			dptr = temp_dst.ptr<uchar>(i);

			for (int j = 0; j < scols; j++)
				dptr[j] = sptr[j] > threshold ? UCHAR_MAX : 0;
		}
		dst = temp_dst;
		break;
	case THRESHOLD_TYPE::THRESHOLD_INVERSE:
		;
		break;
	default:
		break;
	}
}

void rungenerator(const Mat& src, vector<Point3i>& runset)
{
	int rows = src.rows, cols = src.cols;
	const uchar* ptr;
	bool flag=false;
	int start = -1;
	for (int i = 0; i < rows; i++)
	{
		ptr = src.ptr<uchar>(i);
		for (int j = 0; j < cols-1; j++)
		{
			if (ptr[j] ==255 && flag == false)
			{
				start = j;
				flag = true;
			}
			else if (ptr[j] == 0 && flag == true)
			{
				runset.push_back(Point3i(i, start, j - 1));
				flag = false;
			}
		}
		if (flag == true)/* 忽略最后一列的独立点 */
		{
			runset.push_back(Point3i(i, start, cols - 1));
			flag = false;
		}
	}
}

void labelrun(vector<Point3i>& runset,vector<int>& labelset,vector<Point2i>& connectship)
{
	labelset.assign(runset.size(), -1);
	int current_row = 0, current_row_first_run=0,last_row_first_run_idx = 0, last_row_last_run_idx = -1;
	int label_last_id =1;
	for (int i = 0; i < runset.size(); i++)
	{ 
		if (runset[i].x != current_row)
		{
			last_row_first_run_idx = current_row_first_run;
			last_row_last_run_idx = i - 1;
			current_row_first_run = i;
			current_row = runset[i].x;
		}
		for (int j = last_row_first_run_idx; j <= last_row_last_run_idx; j++)
		{
			if (runset[i].RUN_HEAD <= runset[j].RUN_END && runset[j].RUN_HEAD <= runset[i].RUN_END 
				&& runset[i].x == runset[j].x + 1)/* 有可能中间隔了一行没行程的行*/
			{
				if (labelset[i] == -1) /* 当本行和上行只有一个行程交接，则本行程肯定没被标记过，赋予其和上行交接行程同标记 */
					labelset[i] = labelset[j];
				else if(labelset[i] != labelset[j])/* 当本行和上行有2个行程相交接，由于上行两行程可能有同一个label(他们和他们的上行同一个行程交接时) 所以需要判断是否相等而不是判断有没有标记 */
					connectship.push_back(Point2i(labelset[i], labelset[j]));/* 记录两个标签的具有连接关系 */
			}
		}
		if (labelset[i] == -1)
			labelset[i] = label_last_id++;
	}
}

void relabel(vector<int>& labelset, vector<Point2i>& connectship)
{
	int maxLabel = *max_element(labelset.begin(), labelset.end());
	vector<vector<bool>> eqTab(maxLabel, vector<bool>(maxLabel, false)); /* 初始化生成无向等价图 */

	for (int i = 0; i < connectship.size(); i++)/* 根据配对表设置无向图中的联通情况 */
	{
		eqTab[connectship[i].x-1][connectship[i].y-1] = true;
		eqTab[connectship[i].y-1][connectship[i].x-1] = true;
	}

	vector<int> labelFlag(maxLabel, 0);
	vector<vector<int>> equaList;
	vector<int> tempList;

	for (int i = 1; i <= maxLabel; i++)
	{
		if (labelFlag[i - 1])
		{
			continue;
		}
		labelFlag[i - 1] = equaList.size() + 1;
		tempList.push_back(i);
		for (vector<int>::size_type j = 0; j < tempList.size(); j++)
		{
			for (vector<bool>::size_type k = 0; k != eqTab[tempList[j] - 1].size(); k++)
			{
				if (eqTab[tempList[j] - 1][k] && !labelFlag[k])
				{
					tempList.push_back(k + 1);
					labelFlag[k] = equaList.size() + 1;
				}
			}
		}
		equaList.push_back(tempList);
		tempList.clear();
	}
	for (vector<int>::size_type i = 0; i != labelset.size(); i++)
	{
		labelset[i] = labelFlag[labelset[i] - 1];
	}
}

void component_labeling(const Mat &src, vector<Point3i>& runset, vector<int>& labelset)
{
	vector<Point2i> connectship;
	rungenerator(src, runset);

	if (!runset.empty())
		labelrun(runset, labelset, connectship);
	if(!connectship.empty())
		relabel(labelset, connectship);
}

void component_classification(vector<Point3i>& runset, vector<int>& labelset, vector<vector<Point3i>>& componentset,int area_threshold)
{
	vector<int> area_set;/* label值对应的面积 */
	area_set.assign(*max_element(labelset.begin(),labelset.end()), 0);
	for(int i = 0; i < runset.size(); i++)
		area_set[labelset[i]-1] += runset[i].z - runset[i].y + 1;
	int num = 0;

	for (int i = 0; i < area_set.size(); i++)
	{
		if (area_threshold==-1 ||area_set[i] > area_threshold )
		{
			
			componentset.push_back(vector<Point3i>());
			for (int j = 0; j < labelset.size(); j++)
			{
				/* 第j个run 的label的值为 此处面积大于阈值的label */
				if (labelset[j] == i+1)
				{
					componentset[num].push_back(runset[j]);
				}
			}
			num++;
		}
	}
}

void get_component(const Mat& src, vector<vector<Point3i>>& component, int area_threshold)
{
	vector<Point3i> runset;
	vector<int> labelset;
	component_labeling(src, runset, labelset);
	component_classification(runset, labelset, component, area_threshold);
}

void display_component(string imname, const Mat& src, Mat& dst, vector<vector<Point3i>>& component)
{
	dst = Mat::zeros(Size(src.cols, src.rows), CV_8UC1);

	int step = int(255.0 / component.size());
	for (int i = 0; i < component.size(); ++i)
	{
		for (int j = 0; j < component[i].size(); j++)
		{
			for (int k = component[i][j].y; k <= component[i][j].z; k++)
			{
				dst.ptr<uchar>(component[i][j].x)[k] = (i + 1) * step;
			}
		}
	}
}

int component_area(vector<Point3i>& component)
{
	int area=0;
	for (int i = 0; i < component.size(); i++)
	{
		area += component[i].z - component[i].y + 1;
	}
	return area;
}

Point2i component_center(vector<Point3i>& component,int *_area)
{
	float hor = 0, ver = 0;
	for (int i = 0; i < component.size(); i++)
	{
		ver += component[i].x*(component[i].z- component[i].y+1);
		hor += (component[i].z* component[i].z - component[i].y* component[i].y + component[i].z + component[i].y) * 0.5;
	}
	int area = component_area(component);
	*_area = area;
	return Point2i(int(hor/area), int(ver/area));
}

vector<int> component_dmax(vector<Point3i>& component)
{
	int max = 0, temp = 0, dx1 = 0, dx2 = 0, dy = 0;
	vector<int> output = {-1,-1,-1,-1,-1};
	int flag = 0;
	for (int i = 0; i < component.size()-1; i++)
	{
		for (int j = i+1; j < component.size(); j++)
		{
			dx1 = component[j].z - component[i].y;
			dx2 = component[j].y - component[i].z;
			dy = component[i].x - component[j].x;
			dx1 *=dx1;
			dx2 *= dx2;
			dy *= dy;
			temp =  (flag=(dx1 > dx2)) ? dx1 + dy : dx2 + dy;
			if (temp > max)
			{
				output[0] = flag ? component[j].z: component[j].y;
				output[1] = component[j].x;
				output[2] = flag ? component[i].y : component[i].z;
				output[3] = component[i].x;
				output[4] = sqrtf(temp);
				max = temp;
			}
		}
	}
	return output;
}

float component_circularity(vector<Point3i>& component,int dmax,int area)
{
	return 4 * area / (CV_PI * dmax * dmax);
}