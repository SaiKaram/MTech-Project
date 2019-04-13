#include "header1.hpp"

#include<opencv2/opencv.hpp>
#include<iostream>
#include<cassert>
#include<cmath>
#include<fstream>
#include<string>

using namespace cv;
using namespace std;




void stabilization(string filename1,string filename2,vector<Trajectory> &trajectory1,vector<Trajectory> &trajectory2,vector<TransformParam> &prev_to_cur_transform1,vector<TransformParam> &prev_to_cur_transform2,vector<Mat> &C1,vector<Mat> &P1,vector<Mat> &C2,vector<Mat> &P2)
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
    
    
    // Step 1 - Get previous to current frame transformation (dx, dy, da) for all frames
 

    int max_frames1 = cap1.get(CV_CAP_PROP_FRAME_COUNT);
    int max_frames2 = cap2.get(CV_CAP_PROP_FRAME_COUNT);
    
    int max_frames = (max_frames1 > max_frames2) ? max_frames2 : max_frames1;
    
    Mat last_T;
    Mat last_T1;
    Mat last_T0;

    vector<Point2f> prev_corner3;
    vector<Point2f> prev_corner1;
   
    
    int k = 0;
    
    while(k < max_frames - 1) {
        
        Mat cur1, cur_grey1, cur2, cur_grey2;
        
        cap1 >> cur1;
        cap2 >> cur2;

        if(cur1.data == NULL || cur2.data == NULL) {
            break;
        }

        cvtColor(cur1, cur_grey1, COLOR_BGR2GRAY);
        cvtColor(cur2, cur_grey2, COLOR_BGR2GRAY);
        
        // vector from prev to cur
        vector<Point2f> prev_corner5;
        vector<Point2f> prev_corner4;vector<Point2f> cur_corner4;vector<Point2f> cur_corner5;
        vector<Point2f> prev_corner2;vector<Point2f> cur_corner2;
        vector<Point2f> cur_corner1;
          vector<Point2f> cur_corner3;
        
        vector<uchar> status1;
        vector<uchar> status2;
        vector<float> err1;
        vector<float> err2;
        
        int GRID_SIZE = 100;
        int height = cur1.rows;
        int width = cur1.cols;
        
        
        Mat prev_grey1_copy = prev_grey1.clone();
        Mat prev_grey2_copy = prev_grey2.clone();
        Mat prev_grey2_copy2 = prev_grey2.clone();
        Mat prev_grey1_copy2 = prev_grey1.clone();
     
        Mat T0;
       
       // Track the features

        if(prev_corner1.size() < 800 || prev_corner3.size() < 800)
        {
            for (int y = 0; y < height - GRID_SIZE; y = y + GRID_SIZE)
            {
                for (int x = 0; x < width - GRID_SIZE; x = x + GRID_SIZE) 
                {
                    vector <Point2f> prev_corner0;
                    vector <Point2f> prev_corner;
		
              
                    Rect grid_rect(x, y, GRID_SIZE, GRID_SIZE);
              
            

                    goodFeaturesToTrack(prev_grey1(grid_rect),prev_corner0,1000,0.01,10);
                
//                     Weed out bad matches
                    for(size_t i = 0; i < prev_corner0.size(); i++)
                    {
                            double x_temp1 = prev_corner0[i].x + x ;
                            double x_temp2 = prev_corner0[i].y + y;
                            prev_corner1.push_back(Point2f(x_temp1,x_temp2));
                	
                
                    }
                
                    goodFeaturesToTrack(prev_grey2(grid_rect),prev_corner,1000,0.01,10);
                
                
                
                    for(size_t i = 0; i < prev_corner.size(); i++)
                    {
                            double x_temp1 = prev_corner[i].x + x;
                            double x_temp2 = prev_corner[i].y + y;
                            prev_corner3.push_back(Point2f(x_temp1,x_temp2));
                	                
                    }
                

            
                
                    }
            
                }
        }
        

        calcOpticalFlowPyrLK(prev_grey1,cur_grey1, prev_corner1,cur_corner1,status1,err1);
    
        
        
        // weed out bad matches
        for(size_t i=0; i < status1.size(); i++) 
        {
            if(status1[i]) 
            {
                
                prev_corner2.push_back(prev_corner1[i]);
                cur_corner2.push_back(cur_corner1[i]);
            }
        }

        prev_corner1.clear();
        
        for(int i = 0;i < prev_corner2.size();i++)
        {
            prev_corner1.push_back(prev_corner2[i]);
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
        
        
      
        
        calcOpticalFlowPyrLK(prev_grey2,cur_grey2,prev_corner3,cur_corner3,status2,err2);
        // weed out bad matches
        for(size_t i=0; i < status2.size(); i++) {
            if(status2[i]) {
                prev_corner4.push_back(prev_corner3[i]);
                cur_corner4.push_back(cur_corner3[i]);
            }
            
        }
        
        
        prev_corner3.clear();
        
        for(int i = 0;i < prev_corner4.size();i++)
        {
            prev_corner3.push_back(prev_corner4[i]);
        }

        
        
        Mat T1 = estimateRigidTransform(prev_corner4, cur_corner4,false); // false = rigid transform, no scaling/shearing

       

        if(T1.data == NULL)
        {
            last_T1.copyTo(T1);
        }
        
        T1.copyTo(last_T1);
        

        
        Mat H2(Size(3,3),CV_64F,double(0));
        
        H2.at<double>(0,0) = T1.at<double>(0,0);
        H2.at<double>(0,1) = T1.at<double>(0,1);
        H2.at<double>(1,1) = T1.at<double>(1,1);
        H2.at<double>(0,2) = T1.at<double>(0,2);
        H2.at<double>(1,2) = T1.at<double>(1,2);
        H2.at<double>(1,0) = T1.at<double>(1,0);
        H2.at<double>(2,0) = 0;//T1.at<double>(2,0);
        H2.at<double>(2,1) = 0;//T1.at<double>(2,1);
        H2.at<double>(2,2) = 0;//T1.at<double>(2,2);
        
        P2.push_back(H2.clone());
        C2.push_back(H2.clone());
        
        
        // decompose T
        double dx1 = T.at<double>(0,2);
        double dy1 = T.at<double>(1,2);
        double da1 = atan2(T.at<double>(1,0), T.at<double>(0,0));

        double dx2 = T1.at<double>(0,2);
        double dy2 = T1.at<double>(1,2);
        double da2 = atan2(T1.at<double>(1,0), T1.at<double>(0,0));
        


        prev_to_cur_transform1.push_back(TransformParam(dx1, dy1, da1));
        prev_to_cur_transform2.push_back(TransformParam(dx2, dy2, da2));
   
        cur1.copyTo(prev1);
        cur_grey1.copyTo(prev_grey1);

        cur2.copyTo(prev2);
        cur_grey2.copyTo(prev_grey2);

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
    
    
     a = 0;
     x = 0;
     y = 0;

    vector<Trajectory> trajectory; // trajectory at all frames

    for(size_t i=0; i < prev_to_cur_transform2.size(); i++) {
        x += prev_to_cur_transform2[i].dx;
        y += prev_to_cur_transform2[i].dy;
        a += prev_to_cur_transform2[i].da;

        trajectory2.push_back(Trajectory(x,y,a));


    }

     

  
  cap1.release();
  cap2.release();
}
