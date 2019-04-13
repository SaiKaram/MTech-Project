#include<iostream>
#include<string>

#include<opencv2/opencv.hpp>
#include<cassert>
#include<cmath>
#include<fstream>
#include <iomanip>
#include "opencv2/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/stitching.hpp"
#include <time.h>
#include <algorithm>
 

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;





#define STD 1.0
#define MEAN 0.0
#define K_SIZE 60


const int SMOOTHING_RADIUS = 60;//In frames. The larger the more stable the video
const int HORIZONTAL_BORDER_CROP = 30;//In pixels. Crops the border to reduce the black borders from stabilization being too noticable
struct TransformParam
{
	TransformParam(){}
	TransformParam(double _dx, double _dy, double _da)
	{
		dx = _dx;
		dy = _dy;
		da = _da;
		
	}
	double dx;
	double dy;
	double da;

};

struct Trajectory
{
	Trajectory(){}
	Trajectory(double _x, double _y, double _a)
	{
		x = _x;
		y = _y;
		a = _a;
		
	}
	double x;
	double y;
	double a;
	
};
void stabilization(string filename1,string filename2,vector<Trajectory> &trajectory1,vector<Trajectory> &trajectory2,vector<TransformParam> &prev_to_cur_transform1,vector<TransformParam> &prev_to_cur_transform2,vector<Mat> &C1,vector<Mat> &P1,vector<Mat> &C2,vector<Mat> &P2);

void new_stabilization(string filename1,string filename2,vector<Trajectory> &trajectory1,vector<Trajectory> &trajectory2,vector<TransformParam> &prev_to_cur_transform1,vector<TransformParam> &prev_to_cur_transform2, vector<Trajectory> &T12,vector<TransformParam> &new_transform,vector<Mat> &C1,vector<Mat> &P1,vector<Mat> &C2,vector<Mat> &P2,vector<Mat> &T_map);

void overlap(string filename1,string filename2,vector<Trajectory> &trajectory1,vector<TransformParam> &prev_to_cur_transform1,vector<Mat> &C1,vector<Mat> &P1);
void find_matches(int k, Mat &frame1,Mat &frame2,vector<Point2f> &prev_corner1,vector<Point2f> &prev_corner3, Mat &T0);

void video_stitching(string filename1,string filename2);
void Video_stabilization(string filename1);
Mat Stitching(Mat image1, Mat image2);
void FilterCreation1d(double *GKernel);