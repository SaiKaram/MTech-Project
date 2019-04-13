#include "header1.hpp"

#include<opencv2/opencv.hpp>
#include<iostream>
#include<cassert>
#include<cmath>
#include<fstream>
#include<string>

using namespace cv;
using namespace std;


int height;
int width;
double minval, maxval;
Point minloc, maxloc;
Rect roi;
Mat res;


Point find_overlap_start(Mat left_img, Mat right_img)
{
    // Check if the sizes of images are the same.
    if (left_img.size() == right_img.size())
    {
        // Get rows columns of left image.
        height = left_img.rows;
        width = left_img.cols;
        
 
        // Copy left image to new variable.
        Mat haystack = left_img;

        // Create matrix "neddle" to be used as our "template". 
//         roi.width = (left_img.size().width);
//         roi.height = left_img.size().height;
        
        Rect roi(0,0,width - (width / 2) ,height - (height / 2));
        Mat needle = right_img(roi);
        
//         imshow("image", needle);

        // Apply template matching and store result in res.
        matchTemplate(haystack, needle, res, TM_CCORR_NORMED);
        minMaxLoc(res, &minval, &maxval, &minloc, &maxloc);

        // Return top-left coordinate where the matching starts.
        return maxloc;
    }
}

// Create vector of points where images start to overlap.
vector<Point> find_overlaps(vector<Mat> images)
{
    // Create vector of overlaps coordinates.
    vector<Point> overlap_starts;

    for (int i = 0; i != images.size() - 1; i++) 
    {
        // Find overlap start between two images.
        Point overlap = find_overlap_start(images[i], images[i + 1]);

        // Add overlap point to vector of points.
        overlap_starts.push_back(overlap);
    }

    // Return vector of "overlaping points" (top-left).
    return overlap_starts;
}


void overlap(string filename1,string filename2,vector<Trajectory> &trajectory1,vector<TransformParam> &prev_to_cur_transform1,vector<Mat> &C1,vector<Mat> &P1)
{
    
    VideoCapture cap1(filename1);
    assert(cap1.isOpened());

    
    VideoCapture cap2(filename2);
    assert(cap2.isOpened());
    
     
    Mat prev1,prev_grey1, prev2, prev_grey2;

    cap1 >> prev1;
    cap2 >> prev2;
    
    cvtColor(prev1, prev_grey1, COLOR_BGR2GRAY);
    cvtColor(prev2, prev_grey2, COLOR_BGR2GRAY);
    
     
    vector<Mat> various_images;
    various_images.push_back(prev_grey1);
    various_images.push_back(prev_grey2);
  	
    vector<Point> overlap_starts = find_overlaps(various_images);
    
    cout << "SIZE : " <<  overlap_starts[0,0] << endl;
    
    
    
    Rect roi(overlap_starts[0,0].x,overlap_starts[0,0].y,prev_grey1.cols-overlap_starts[0.0].x,prev_grey1.rows-overlap_starts[0.0].y);
//         Rect roi(overlap_starts[0,0].x,overlap_starts[0,0].y,img1.cols-overlap_starts[0.0].x,img1.rows-overlap_starts[0.0].y);
    
   
   
   Mat frame5 = prev_grey1(roi).clone();
//    Mat frame6 = frame2(roi).clone();
   
   
    
    
    
    // Step 1 - Get previous to current frame transformation (dx, dy, da) for all frames
 

    int max_frames1 = cap1.get(CV_CAP_PROP_FRAME_COUNT);
    int max_frames2 = cap2.get(CV_CAP_PROP_FRAME_COUNT);
    
    int max_frames = (max_frames1 > max_frames2) ? max_frames2 : max_frames1;
    
    Mat last_T;
   
    
    int k = 0;
    
    while(k < max_frames - 1)
    {
        
        Mat cur1, cur_grey1, cur2, cur_grey2;
        
        cap1 >> cur1;
        cap2 >> cur2;

        if(cur1.data == NULL || cur2.data == NULL) {
            break;
        }

        cvtColor(cur1, cur_grey1, COLOR_BGR2GRAY);
        cvtColor(cur2, cur_grey2, COLOR_BGR2GRAY);
    
    
//         Rect roi1(overlap_starts[0,0].x,overlap_starts[0,0].y,cur_grey1.cols-overlap_starts[0.0].x,cur_grey1.rows-overlap_starts[0.0].y);
        
        
        Mat frame6 = cur_grey1(roi).clone();
        
        // vector from prev to cur
        vector<Point2f> prev_corner,cur_corner;
        
        vector<uchar> status;
        vector<float> err;
    
        
        int GRID_SIZE = 100;
        int height = cur1.rows;
        int width = cur1.cols;
        
        

        Mat T0;
       
       // Track the features

        if(prev_corner.size() < 800)
        {
//             for (int y = 0; y < height - GRID_SIZE; y = y + GRID_SIZE)
//             {
//                 for (int x = 0; x < width - GRID_SIZE; x = x + GRID_SIZE) 
//                 {
//                     vector <Point2f> prev_corner0;
//                     vector <Point2f> prev_corner;
// 		
//               
//                     Rect grid_rect(x, y, GRID_SIZE, GRID_SIZE);
//               
            

//                     goodFeaturesToTrack(frame5(grid_rect),prev_corner0,1000,0.01,10);
            
                    goodFeaturesToTrack(frame5,prev_corner,1000,0.01,10);
                /*
//                     Weed out bad matches
                    for(size_t i = 0; i < prev_corner0.size(); i++)
                    {
                            double x_temp1 = prev_corner0[i].x + x ;
                            double x_temp2 = prev_corner0[i].y + y;
                            prev_corner.push_back(Point2f(x_temp1,x_temp2));
                	
                
                    }
                
                }
            
            }*/
        }
        

        calcOpticalFlowPyrLK(frame5,frame6, prev_corner,cur_corner,status,err);
    
        vector<Point2f> prev_corner2,cur_corner2;
        
        // weed out bad matches
        for(size_t i=0; i < status.size(); i++) 
        {
            if(status[i]) 
            {
                
                prev_corner2.push_back(prev_corner[i]);
                cur_corner2.push_back(cur_corner[i]);
            }
        }

        prev_corner.clear();
        
        for(int i = 0;i < prev_corner2.size();i++)
        {
            prev_corner.push_back(prev_corner2[i]);
        }

        // translation + rotation only
        Mat T = estimateRigidTransform(prev_corner2, cur_corner2,false); // false = rigid transform, no scaling/shearing
        
        if(T.data == NULL)
        {
            last_T.copyTo(T);
        }
        
        Mat H1(Size(3,3),CV_64F,double(0));
        
        H1.at<double>(0,0) = T.at<double>(0,0);
        H1.at<double>(0,1) = T.at<double>(0,1);
        H1.at<double>(1,1) = T.at<double>(1,1);
        H1.at<double>(0,2) = T.at<double>(0,2);
        H1.at<double>(1,2) = T.at<double>(1,2);
        H1.at<double>(1,0) = T.at<double>(1,0);
        H1.at<double>(2,0) = 0;//T.at<double>(2,0);
        H1.at<double>(2,1) = 0;//T.at<double>(2,1);
        H1.at<double>(2,2) = 0;//T.at<double>(2,2);
    
        P1.push_back(H1.clone());
        C1.push_back(H1.clone());
        
        
      
        
       
        
        // decompose T
        double dx1 = T.at<double>(0,2);
        double dy1 = T.at<double>(1,2);
        double da1 = atan2(T.at<double>(1,0), T.at<double>(0,0));


        


        prev_to_cur_transform1.push_back(TransformParam(dx1, dy1, da1));

        cur1.copyTo(prev1);
        cur_grey1.copyTo(prev_grey1);

        cur2.copyTo(prev2);
        cur_grey2.copyTo(prev_grey2);
        
        frame6.copyTo(frame5);

        cout << "Frame: " << k << "/" << max_frames << " - good optical flow: " << prev_corner2.size() << endl;
        k++;
    }

    // Step 2 - Accumulate the transformations to get the image trajectory

    // Accumulated frame to frame transform
    double a = 0;
    double x = 0;
    double y = 0;

    for(size_t i=0; i < prev_to_cur_transform1.size(); i++) {
        x += prev_to_cur_transform1[i].dx;
        y += prev_to_cur_transform1[i].dy;
        a += prev_to_cur_transform1[i].da;

        trajectory1.push_back(Trajectory(x,y,a));


    }
    
   
     

  
  cap1.release();
  cap2.release();
}
