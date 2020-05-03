#include "measure.h"
#include <algorithm>
/*********************
 *@author:ŷ����Դ
 *@date:
 *
 ********************/
int mouse_x;
int mouse_y;
bool settingmode = false;
void onchange(int pos, void* userdata) { ; }
void creat_setting_bar(int *rt,int*tt,int *at,int * threshold)
{
    createTrackbar("����Χ", "pic", rt, 50, onchange);
    createTrackbar("����Χ", "pic", tt, 50, onchange);
    createTrackbar("�ֳɼ���", "pic", at, 50, onchange);
    createTrackbar("��ֵ", "pic", threshold, 255, onchange);
}
void mouse_callback(int event, int x, int y, int flags, void* param)
{
    mouse_x = x;
    mouse_y = y;
    /*enum MouseEventTypes {
        EVENT_MOUSEMOVE = 0, //!< indicates that the mouse pointer has moved over the window.
        EVENT_LBUTTONDOWN = 1, //!< indicates that the left mouse button is pressed.
        EVENT_RBUTTONDOWN = 2, //!< indicates that the right mouse button is pressed.
        EVENT_MBUTTONDOWN = 3, //!< indicates that the middle mouse button is pressed.
        EVENT_LBUTTONUP = 4, //!< indicates that left mouse button is released.
        EVENT_RBUTTONUP = 5, //!< indicates that right mouse button is released.
        EVENT_MBUTTONUP = 6, //!< indicates that middle mouse button is released.
        EVENT_LBUTTONDBLCLK = 7, //!< indicates that left mouse button is double clicked.
        EVENT_RBUTTONDBLCLK = 8, //!< indicates that right mouse button is double clicked.
        EVENT_MBUTTONDBLCLK = 9, //!< indicates that middle mouse button is double clicked.
        EVENT_MOUSEWHEEL = 10,//!< positive and negative values mean forward and backward scrolling, respectively.
        EVENT_MOUSEHWHEEL = 11 //!< positive and negative values mean right and left scrolling, respectively.
    };*/
    switch (event)
    {
    case EVENT_MOUSEMOVE:
    {   /* �϶�������ð뾶 */
        ;
    }
    case EVENT_LBUTTONDOWN:
    {   /* ���������ѡԲ�� */
        ;
    }
    case EVENT_LBUTTONUP:
    {   /* ȷ����ѡԲ���� */
        ;
    }
    case EVENT_LBUTTONDBLCLK:
    {
        /* ���˫����������ģʽ����ʼ��ѡԲ */
        settingmode = true;
        ;
    }
    case EVENT_RBUTTONDBLCLK:
    {
        /* �Ҽ�˫����ʼ���� */
        settingmode = false;
        ;
    }
    default:
        break;
    }
}

void windows_mouse_combind(string winname)
{
    setMouseCallback(winname, mouse_callback);
}

//˫���Բ�ֵ
inline unsigned char BilinearInterpolation(const Mat& src, float x, float y)
{
    /*
     *  x1 x x2
     *y1
     *y    p
     *y2
     */

    int x1 = (int)x, y1 = (int)y, step = src.step;

    float
        fR1 = (x1 + 1 - x) * src.data[y1 * step + x1] + (x - x1) * src.data[step * y1 + x1 + 1],
        fR2 = (x1 + 1 - x) * src.data[(y1 + 1) * step + x1] + (x - x1) * src.data[(y1 + 1) * step + x1 + 1];

    return (unsigned char)((y1 + 1 - y) * fR1 + (y - y1) * fR2);
}
//��������������ȡ��
float subpixel_1d(int ca, double F3[])
{
    int x[] = { ca - 1,ca,ca + 1 };
    if (abs((F3[0] + F3[2]) / 2 - F3[1]) < 1e-6)
        return ca;
    double dA[] = { x[0] * x[0],x[0],1,
                    x[1] * x[1],x[1],1,
                    x[2] * x[2],x[2],1 };

    Mat p, temp, A = Mat(3, 3, CV_64F, dA), F = Mat(3, 1, CV_64F, F3);
    invert(A, temp);
    p = temp * F;
    double* pd = (double*)(p.data);

    return -pd[1] / (2.0 * pd[0]);
}
//���¶���Բ���
Point3f LS_circle_fitting(vector<Point2f> pointset)
{
    /*          F  =  Px  
        -x^2 - y^2 = a*x + b*y + c 
     */
    float * _Y = new float[pointset.size()], * _P = new float[3 * pointset.size()],a,b,c;
    for (int i = 0; i < pointset.size(); i++)
    {
        _Y[i] = -pointset[i].x * pointset[i].x - pointset[i].y * pointset[i].y;
        _P[3 * i] = pointset[i].x;
        _P[3 * i + 1] = pointset[i].y;
        _P[3 * i + 2] = 1;
    }
    Mat Y(pointset.size(), 1, CV_32FC1, _Y), P(pointset.size(), 3, CV_32FC1, _P), X, temp;
    invert(P.t() * P, temp);
    X= temp * P.t() * Y;
    a = X.ptr<float>(0)[0];
    b = X.ptr<float>(1)[0];
    c = X.ptr<float>(2)[0];
    return Point3f(-a/2,-b/2,sqrtf(a*a+b*b-4*c)/2);

}
//�ԻҶ�ͼ��Բ���г�����λ bt:��ֵ����ֵ��at:�����ֵ��ct��Բ����ֵ,Point XYR
vector<vector<Point3i>> comp;
void CircleLocating(const Mat& src, vector<Point3i>& _circle, int bt, int at, int ct)
{
    
    vector<Point> comp_center;
    vector<int> comp_area;
    vector<vector<int>> comp_dmax;
    vector<float> comp_cicurlarity;
    Mat binary;

    _circle.clear();
    comp.clear();
    mythreshold(src, binary, bt, THRESHOLD_TYPE::THRESHOLD_BINARY);
    get_component(binary, comp, at); /* �õ���ͨ�����г�Ϊ��Ԫ�����һ�����򼯺��� */
    
    /* ����ͨ�������������ȡ */
    for (int i = 0; i < comp.size(); i++)
    {
        /* ������ͨ����������� */
        comp_area.push_back(0);
        comp_center.push_back(component_center(comp[i], &comp_area[i]));
        

        /* ������ͨ�����ֱ�� */
        comp_dmax.push_back(component_dmax(comp[i]));

        /* ������ͨ��Բ�� */
        comp_cicurlarity.push_back(component_circularity(comp[i], comp_dmax[i][4], comp_area[i]));
        if (comp_cicurlarity[i] > ct)
        {
            int r = sqrtf(comp_cicurlarity[i] * comp_dmax[i][4] * comp_dmax[i][4] / 4);
            _circle.push_back(Point3i( comp_center[i].x,comp_center[i].y,r));
        }
    }
}

//radiusrate:�뾶ɨ�����ҿ�ȵİٷֱȡ�anglerate:��ԲΪn�ݽ�.vw:��ֱɨ���߿��ռ�뾶�ٷֱ�*100

vector<Point3f> CircleMeasuring(const Mat& src, vector<Point3i> circleset, int rrate, int arate, int trrate, bool display)
{
    vector<Point3f> result;
    vector<int> radial_len, tangential_len;//��ֱ��1d�����ľ�ֵ�˲�������
    vector<vector<Point2f>> subpixel_edage(circleset.size(), vector<Point2f>());//�����ؾ��ȱ��ء�
    vector<vector<Point>> search_roi_display(circleset.size(), vector<Point>());

    float anglestep = CV_2PI / arate;//1d������Բ�ϵĸ�����
    float x0 = 0, y0 = 0, x1 = 0, y1 = 0; //�������������Լ������ꡣ
    /* ����ÿ��Բ */
    for (int cidx = 0; cidx < circleset.size(); ++cidx)
    {
        radial_len.push_back(circleset[cidx].z * rrate / 100.0);
        tangential_len.push_back(trrate * circleset[cidx].z / 100.0);
        /* ����Բ��ĳ��1D���� */
        for (int dir = 0; dir < arate; dir++)
        {
            int max_idx = -1, max_value = -1;
            vector<int> scanline_filted, diff_1(radial_len[cidx] * 2 + 1);
            /* ����1Dɨ����ÿ���� */
            for (int rlen = -radial_len[cidx]; rlen <= radial_len[cidx]; rlen++)
            {
                x0 = circleset[cidx].x + (circleset[cidx].z + rlen) * cos(dir * anglestep);//ĳ��Բĳ��������1d������ĳ���㡣
                y0 = circleset[cidx].y - (circleset[cidx].z + rlen) * sin(dir * anglestep);
                
                int sum = 0;
                /* ɨ����ĳ���㴹ֱ�����˲� */
                for (int tlen = -tangential_len[cidx]; tlen <= tangential_len[cidx]; tlen++)
                {
                    x1 = x0 + tlen * sin(dir * anglestep);
                    y1 = y0 + tlen * cos(dir * anglestep);
                    sum += BilinearInterpolation(src, x1, y1);
                    if (display)
                        if (radial_len[cidx] == abs(rlen) && tangential_len[cidx] == abs(tlen))
                            search_roi_display[cidx].push_back(Point(x1, y1));
                }
                scanline_filted.push_back(sum / (2 * tangential_len[cidx] + 1));
            }
            /* �ݶȼ��� */
            diff_1[0]=0;//�߽紦��    
            diff_1[diff_1.size()-1]=0;//�߽紦��
            for (int i = 1; i < scanline_filted.size() - 1; i++)
            {
                diff_1[i] = abs(scanline_filted[i + 1] - scanline_filted[i]) / 2;
                if (diff_1[i] >= max_value)
                {
                    max_value = diff_1[i];
                    max_idx = i;
                }
            }
            
            if (max_idx == -1)//���û�ҵ��߽����Ϊ�м��Ǳ߽�
                max_idx = scanline_filted.size() / 2;
            /* �����ؾ�����ȡ */
            if (1 <= max_idx && max_idx <= scanline_filted.size() - 2)//����������ֵ���Լ�����������ϡ�
            {
                double diff_data[3] = { diff_1[max_idx - 1],diff_1[max_idx],diff_1[max_idx + 1] };
                float sub_idx = subpixel_1d(max_idx, diff_data);
                subpixel_edage[cidx].push_back(
                    Point2f(
                        circleset[cidx].x + (circleset[cidx].z + sub_idx - radial_len[cidx]) * cos(dir * anglestep),
                        circleset[cidx].y - (circleset[cidx].z + sub_idx - radial_len[cidx]) * sin(dir * anglestep)));
            }
            else
                subpixel_edage[cidx].push_back(
                    Point2f(circleset[cidx].x + (circleset[cidx].z + max_idx - radial_len[cidx]) * cos(dir * anglestep),
                        circleset[cidx].y - (circleset[cidx].z + max_idx - radial_len[cidx]) * sin(dir * anglestep)));
        }
        /* ��С����Բ��� */
        result.push_back(LS_circle_fitting(subpixel_edage[cidx]));
    }
    if (display)
    {
        Mat pic = src.clone();
        //display_component("pic", src, pic, comp,true);
        //mythreshold(pic, pic, 210, THRESHOLD_TYPE::THRESHOLD_BINARY);
        cvtColor(pic, pic, COLOR_GRAY2RGB);
        for (int cidx = 0; cidx < result.size(); cidx++)
        {
            /* ���Ƽ��Բ�뾶�򡪡���ɫ */
            circle(pic, Point(circleset[cidx].x, circleset[cidx].y), circleset[cidx].z + radial_len[cidx], Scalar(0, 255, 0), 1, 16);
            circle(pic, Point(circleset[cidx].x, circleset[cidx].y), circleset[cidx].z - radial_len[cidx], Scalar(0, 255, 0), 1, 16);
            /* ���Ƽ����������ɫ */
            circle(pic, Point2d(cvRound(result[cidx].x), cvRound(result[cidx].y)), 1, Scalar(255, 0, 255), 1, 16);
            circle(pic, Point2d(cvRound(result[cidx].x), cvRound(result[cidx].y)), cvRound(result[cidx].z), Scalar(255, 0, 255), 1, 16);
            for (int dir = 0; dir < arate; dir++)
            {

                Point2f p1, p2, p3, p4;
                p1 = subpixel_edage[cidx][dir] + circleset[cidx].z / 30 * Point2f(1, 1);
                p2 = subpixel_edage[cidx][dir] - circleset[cidx].z / 30 * Point2f(1, 1);
                p3 = subpixel_edage[cidx][dir] + circleset[cidx].z / 30 * Point2f(1, -1);
                p4 = subpixel_edage[cidx][dir] - circleset[cidx].z / 30 * Point2f(1, -1);
                line(pic,
                    Point(cvRound(p1.x), cvRound(p1.y)),
                    Point(cvRound(p2.x), cvRound(p2.y)), Scalar(255,0 , 0), 1, 16);
                line(pic,
                    Point(cvRound(p3.x), cvRound(p3.y)),
                    Point(cvRound(p4.x), cvRound(p4.y)), Scalar(255,0 , 0), 1, 16);

                /* ���Ʊ��ص㡪����ɫ */
                circle(pic, subpixel_edage[cidx][dir], 1, Scalar(0, 0, 0), 1, 16);

                /* ����1Dɨ��� ������ɫ */
                line(pic, search_roi_display[cidx][dir * 4], search_roi_display[cidx][dir * 4 + 1], Scalar(0, 125, 255), 1, 16);
                line(pic, search_roi_display[cidx][dir * 4 + 1], search_roi_display[cidx][dir * 4 + 3], Scalar(0, 125, 255), 1, 16);
                line(pic, search_roi_display[cidx][dir * 4 + 3], search_roi_display[cidx][dir * 4 + 2], Scalar(0, 125, 255), 1, 16);
                line(pic, search_roi_display[cidx][dir * 4 + 2], search_roi_display[cidx][dir * 4], Scalar(0, 125, 255), 1, 16);

            }
            char textbuff[30];
            sprintf_s(textbuff, 30, "(%3.2f,%3.2f,%3.2f)", result[cidx].x, result[cidx].y, result[cidx].z);
            putText(pic, textbuff, Point(result[cidx].x, result[cidx].y) + Point(10, 0), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 50, 255), 1);
            
            /* ������ʾ�����½� */
            sprintf_s(textbuff, 30, "(%d,%d)", mouse_x, mouse_y);
            putText(pic, textbuff, Point(10, src.rows - 10), FONT_HERSHEY_TRIPLEX, 0.5, Scalar(255, 255, 255), 1);

        }
        imshow("pic", pic);
        cout << "___________________" << endl;
        cout << "���:\n" << endl;
        cout << result << endl;
        cout << "�ֲ�:\n" << endl;
        cout << circleset << endl;
    }
    
return result;
}