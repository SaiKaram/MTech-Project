#include "header1.hpp"


double calculate_Gaussian(int t, int r, vector<TransformParam> &traject)
{
 
    double variance = SMOOTHING_RADIUS;
    double x = abs(traject[t].dx - traject[r].dx) + abs(traject[t].dy - traject[r].dy);
    double pdf_gaussian;
    
    pdf_gaussian = exp(-9 * pow(x,2) / (2 * pow(variance, 2 ) ));
    
    return pdf_gaussian;
}

double gauss(int t, int r, int sigma)
{
    double gaussian;
    
    gaussian = exp(- 9 * pow(abs(r - t),2) /  (2 * pow(sigma,2)));
    
    return gaussian;
}


Mat translateImg(Mat &img, int offsetx, int offsety)
{
	Mat trans_mat = (Mat_<double>(2, 3) << 1, 0, offsetx, 0, 1, offsety);

	warpAffine(img, img, trans_mat, Size(img.cols + 500, img.rows + 500) );//Size(3 * img.cols, 3 * img.rows)); // 3,4 is usual
	return trans_mat;
}

void warp_crops(Mat &im_1, Mat &im_2)//,Mat &H)//double x, double y ,double a)
{
        int minHessian = 400;
	cv::Ptr<Feature2D> f2d = xfeatures2d::SURF::create(minHessian);
        
//         Mat H;
        
        Mat img1 = im_1.clone();
        Mat img2 = im_2.clone();
        
        int height1 = im_1.rows / 2;
        int width1 = im_1.cols / 2;

        resize(img1,img1,Size(width1,height1));
        resize(img2,img2,Size(width1,height1));
        
        // 	Step 1: Detect the keypoints:
	std::vector<KeyPoint> keypoints_1, keypoints_2;
	f2d->detect( img1, keypoints_1 );
	f2d->detect( img2, keypoints_2 );

        
            for(int i = 0;i < keypoints_1.size();i++)
        {
            keypoints_1[i].pt.x = keypoints_1[i].pt.x * 2;
            keypoints_1[i].pt.y = keypoints_1[i].pt.y * 2;
        
            
        }
        
        for(int i = 0;i < keypoints_2.size();i++)
        {
            keypoints_2[i].pt.x = keypoints_2[i].pt.x * 2;
            keypoints_2[i].pt.y = keypoints_2[i].pt.y * 2;
        }
        
// 	Mat descriptors_1, descriptors_2;
// 	f2d->compute( im_1, keypoints_1, descriptors_1 );
// 	f2d->compute( im_2, keypoints_2, descriptors_2 );
//         
        
// 	Step 2: Calculate descriptors (feature vectors)
	Mat descriptors_1, descriptors_2;
	f2d->compute( im_1, keypoints_1, descriptors_1 );
	f2d->compute( im_2, keypoints_2, descriptors_2 );

// 	Step 3: Matching descriptor vectors using BFMatcher :
	BFMatcher matcher;
	std::vector< DMatch > matches;
	matcher.match( descriptors_1, descriptors_2, matches );

// 	Keep best matches only to have a nice drawing.
// 	We sort distance between descriptor matches
	Mat index;
	int nbMatch = int(matches.size());
	Mat tab(nbMatch, 1, CV_32F);
        double sum = 0;
	for (int i = 0; i < nbMatch; i++)
        {
		tab.at<float>(i, 0) = matches[i].distance;
     
                
        }
        
        
                
	sortIdx(tab, index, SORT_EVERY_COLUMN + SORT_ASCENDING);
	vector<DMatch> bestMatches;

	for (int i = 0; i < 100; i++)
        {
		bestMatches.push_back(matches[index.at < int > (i, 0)]);
                        
        }

       
        
        
// 	1st image is the destination image and the 2nd image is the src image
	std::vector<Point2f> dst_pts;                   //1st
	std::vector<Point2f> source_pts;                //2nd

	for (vector<DMatch>::iterator it = bestMatches.begin(); it != bestMatches.end(); ++it) {
// 		cout << it->queryIdx << "\t" <<  it->trainIdx << "\t"  <<  it->distance << "\n";
// 		-- Get the keypoints from the good matches
		dst_pts.push_back( keypoints_1[ it->queryIdx ].pt );
		source_pts.push_back( keypoints_2[ it->trainIdx ].pt );
	}

	Mat img_matches;
	drawMatches( im_1, keypoints_1, im_2, keypoints_2,
	          bestMatches, img_matches, Scalar::all(-1), Scalar::all(-1),
	          vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
	
// 	-- Show detected matches
	imwrite( "Good_Matches.jpg", img_matches );



	Mat H = findHomography(source_pts, dst_pts, CV_RANSAC );
// 	cout << H << endl;

        
//          Mat H(2,3,CV_64F);
//         
//         H.at<double>(0,0) = cos(a);
//         H.at<double>(0,1) = -sin(a);
//         H.at<double>(1,0) = sin(a);
//         H.at<double>(1,1) = cos(a);
//         H.at<double>(0,2) = x;
//         H.at<double>(1,2) = y;
    
        
        resize(im_1, im_1, Size(im_1.cols, im_1.rows));
        translateImg(im_1, 10, 15); // 2000 is usual 
        translateImg(im_2,10,15);
        resize(im_2, im_2, Size(im_1.cols, im_1.rows));
        
	Mat wim_2;

        warpPerspective(im_2,wim_2, H, im_1.size(), INTER_LINEAR, BORDER_CONSTANT,(0));
    

     // 	Step 1: Detect the keypoints:
	std::vector<KeyPoint> keypoints_11, keypoints_12;
	f2d->detect( im_1, keypoints_11 );
	f2d->detect( im_2, keypoints_12 );


      
	for (int i = 0; i < im_1.cols; i++)
		for (int j = 0; j < im_1.rows; j++) {
			Vec3b color_im1 = im_1.at<Vec3b>(Point(i, j));
			Vec3b color_im2 = wim_2.at<Vec3b>(Point(i, j));
			if (norm(color_im1) == 0)
				im_1.at<Vec3b>(Point(i, j)) = color_im2;

		}
		
		
	/*	
		
		// 	Step 2: Calculate descriptors (feature vectors)
	Mat descriptors_11, descriptors_12;
	f2d->compute(im_1, keypoints_11, descriptors_11 );
	f2d->compute(im_1, keypoints_12, descriptors_12 );

// 	Step 3: Matching descriptor vectors using BFMatcher :
// 	BFMatcher matcher;
	std::vector< DMatch > matches1;
	matcher.match( descriptors_11, descriptors_12, matches1 );

// 	Keep best matches only to have a nice drawing.
// 	We sort distance between descriptor matches
	Mat index1;
	int nbMatch1 = int(matches1.size());
	Mat tab1(nbMatch1, 1, CV_32F);
        double sum1 = 0;
	for (int i = 0; i < nbMatch1; i++)
        {
		tab1.at<float>(i, 0) = matches1[i].distance;
     
                
        }
        
        
                
	sortIdx(tab1, index1, SORT_EVERY_COLUMN + SORT_ASCENDING);
	vector<DMatch> bestMatches1;

	for (int i = 0; i < 100; i++)
        {
		bestMatches1.push_back(matches1[index.at < int > (i, 0)]);
                sum += matches1[i].distance;
        }
        
        for (vector<DMatch>::iterator it1 = bestMatches1.begin(); it1 != bestMatches1.end(); ++it1) {
// 		cout << it1->queryIdx << "\t" <<  it1->trainIdx << "\t"  <<  it1->distance << "\n";
// 		-- Get the keypoints from the good matches
// 		dst_pts.push_back( keypoints_1[ it->queryIdx ].pt );
// 		source_pts.push_back( keypoints_2[ it->trainIdx ].pt );
	}
  cout << "Distance :" << (sum / 100) << endl;*/


}






int video(int argc, char* argv[])
{
    
    vector<Trajectory> trajectory1;
    vector<Trajectory> trajectory2;
    int frame_count1, frame_count2;
    vector<TransformParam> prev_to_cur_transform1;
    vector<TransformParam> prev_to_cur_transform2;
    vector<Mat> C1,P1;
    vector<Mat> C2,P2;
    vector<Mat> P;
    
    
    clock_t begin = clock();
  
    //open the video file for reading
    if(argc < 2)
    {
        cout << "Insufficient Number of Argurments" << endl;
            exit(EXIT_FAILURE);	
    } 

    stabilization(argv[1],argv[2],trajectory1,trajectory2, prev_to_cur_transform1,prev_to_cur_transform2,C1,P1,C2,P2);
    
    const char* Video1 = "Video1";
    const char* Video2 = "Video2";
 
    //namedWindow(Video1,CV_WINDOW_AUTOSIZE);
    
    


   Mat W(trajectory1.size(),trajectory1.size(),CV_64F,double(0));
    
    
   for(int t = 0;t < W.cols;t++)
   {
       for(int r = -SMOOTHING_RADIUS;r<= SMOOTHING_RADIUS;r++)
       {
                    
           if((t+r) >= 0  && (t + r) < W.cols && (t != r))
           {
                    double gaussian0 = gauss(t,t+r,10);
                    double gaussian1 = calculate_Gaussian(t,t+r,prev_to_cur_transform1);
                    double gaussian2 = calculate_Gaussian(t,t+r,prev_to_cur_transform2);
                    double gaussian = (gaussian0 * gaussian1 * gaussian2) ;
                
                    W.at<double>(t,t+r) = gaussian;
                    
                
            }
       }
   }
   
    //cout << "W:" << W << endl;
   
      
   float lambda1 = 5;
   float lambda2 = 5;
   
   double *gamma = new double[trajectory1.size()];
   
   for(int t = 0;t < trajectory1.size(); t++)
   {   
	double sum_g = 0;
	
	for(int r = -SMOOTHING_RADIUS;r <= SMOOTHING_RADIUS;r++)
       {
           if((t+r) >= 0 && (t + r ) < trajectory1.size() && (t != r))
           {
                                 	
 		    sum_g += W.at<double>(t,t+r);
			
	          		    
			
           }
     }
		
   
        //cout << "sum :" << sum_g << endl;

  	gamma[t] = lambda1 + lambda2 +  2 * (sum_g) ;
        
   }

   
   
   int iterations = 20;
   
   vector<Mat> tempo;
   for(int i = 0;i < C1.size();i++)
   {
       tempo.push_back(C1[i].clone());
       P.push_back(C1[i].clone());
   }
   
   
 
for(int iter = 0; iter < iterations;iter++)
 {
    for(int t = 0; t < C1.size();t++)
    {
         Mat sum(Size(3, 3), CV_64F,double(0)); 
         for(int r = -SMOOTHING_RADIUS; r<= SMOOTHING_RADIUS;r++)
         {
             if((t+ r) >= 0 && (t + r) < C1.size()&& (t != r))
             {
		 Mat temp(Size(3,3), CV_64F, double(0));
                 temp = P[t+r].clone();
           	    
                 Mat temp1(Size(3,3), CV_64F, double(0));
                  
                 for(int i = 0;i < temp.rows;i++)
                 {
			for(int j = 0;j < temp.cols;j++)
			{
			   temp1.at<double>(i,j) = temp.at<double>(i,j) * W.at<double>(t,t+r);
                         }
                   }
                
            
		   
		   for(int i = 0;i < sum.rows;i++)
                   {
			for(int j = 0;j < sum.cols;j++)
			{
			   sum.at<double>(i,j) = sum.at<double>(i,j) + temp1.at<double>(i,j);
                         }
                   }
                   
                   temp1.release();
		   temp.release();
				
                }
            }
        
    
		   
          Mat add(Size(3,3), CV_64F, double(0)); 
  
          Mat add1(Size(3,3), CV_64F, double(0)); 
          for(int i = 0;i < add.rows;i++)
          {
             for(int j = 0;j < add.cols;j++)
             {
                add.at<double>(i,j) = sum.at<double>(i,j) * 2 ;
              }
            }
                
 	   
 	   Mat C1_temp = C1[t].clone();
           Mat C2_temp = C2[t].clone();
           
 	   for(int i = 0;i < add.rows;i++)
           {
             for(int j = 0;j < add.cols;j++)
             {
                    add1.at<double>(i,j) = add.at<double>(i,j) + lambda1 * C1_temp.at<double>(i,j) + lambda2 * C2_temp.at<double>(i,j);                       
                            
                }
            }           
 
        
 	  Mat new_temp(Size(3,3), CV_64F, double(0)); 
  		  for(int i = 0;i < add1.rows;i++)
                {
                    for(int j = 0;j < add1.cols;j++)
                    {
                       new_temp.at<double>(i,j) = add1.at<double>(i,j) / gamma[t];                       
                             
                    }
                }           
  
            
             
            tempo[t] = new_temp.clone();
     
        sum.release();
        add1.release();
        C1_temp.release();
        C2_temp.release();
        new_temp.release();

      add.release();
    } 
       
    for(int k = 0;k < tempo.size();k++)
    {
        P[k] = tempo[k].clone();
    }
    
    
}

W.release();


vector<TransformParam> prev_to_cur_transform1_temp;

for(int i = 0;i < P.size();i++)
{
    double dx1 = P[i].at<double>(0,2);
    double dy1 = P[i].at<double>(1,2);
    double da1 = atan2(P[i].at<double>(1,0), P[i].at<double>(0,0));
    
    prev_to_cur_transform1_temp.push_back(TransformParam(dx1, dy1, da1));
}


vector<Mat> newInverse;

for(int i = 0;i < C1.size();i++)
{
   Mat inverse(Size(3,3),CV_64F,double(0));
   inverse = C1[i].clone();
    newInverse.push_back(inverse);

    
    inverse.release();
    
    
}




vector<Mat> newInverse1;

for(int i = 0;i < P.size();i++)
{
   Mat inverse(Size(3,3),CV_64F,double(0));

   inverse = P[i].clone();
   
    newInverse1.push_back(inverse);
    
    inverse.release();
    
    
}

vector<Mat> newInverse2;

for(int i = 0;i < P.size();i++)
{
   Mat inverse(Size(3,3),CV_64F,double(0));

   inverse = C2[i].clone();
   
    newInverse2.push_back(inverse);
    
    inverse.release();
    
    
}

vector<Mat> product1;
vector<Mat> product2;
vector<Mat> U1;
vector<Mat> U2;


for(int i = 0; i < C1.size();i++)
{
    product1.push_back((newInverse[i].inv()) * ((newInverse1[i])));
    cout << "product1:" << product1[i] << endl;
}



for(int i = 0; i < C2.size();i++)
{
    product2.push_back((newInverse2[i].inv()) * ((newInverse1[i])));
    cout << "product2:" << product2[i] << endl;
    
}


vector<TransformParam> new_prev_to_cur_transform1,new_prev_to_cur_transform2;




for(int i = 0;i < C1.size();i++)
{
    double dx = product1[i].at<double>(0,2);
    double dy = product1[i].at<double>(1,2);
    double da = atan2(product1[i].at<double>(1,0), product1[i].at<double>(0,0));

    new_prev_to_cur_transform1.push_back(TransformParam(dx, dy, da));
}






for(int i = 0;i < C2.size();i++)
{
    double dx = product2[i].at<double>(0,2);
    double dy = product2[i].at<double>(1,2);
    double da = atan2(product2[i].at<double>(1,0), product2[i].at<double>(0,0));

    new_prev_to_cur_transform2.push_back(TransformParam(dx, dy, da));
}


double x = 0;
double y = 0;
double a = 0;

vector<Trajectory> optimal1,optimal2;

 for(int i = 0;i < new_prev_to_cur_transform1.size();i++)
    {
        
       x += prev_to_cur_transform1_temp[i].dx;
       y += prev_to_cur_transform1_temp[i].dy;
       a += prev_to_cur_transform1_temp[i].da;
       
       optimal1.push_back(Trajectory(x,y,a));
    }

    
x = 0;
y = 0;
a = 0;


vector<TransformParam> new_prev_to_cur_transform3,new_prev_to_cur_transform4;

   for(size_t i=0; i < prev_to_cur_transform1.size(); i++)
    {
        x += prev_to_cur_transform1[i].dx;
        y += prev_to_cur_transform1[i].dy;
        a += prev_to_cur_transform1[i].da;

        // target - current
        double diff_x = optimal1[i].x - x;
        double diff_y = optimal1[i].y - y;
        double diff_a = optimal1[i].a - a;

        double dx = prev_to_cur_transform1[i].dx + diff_x;
        double dy = prev_to_cur_transform1[i].dy + diff_y;
        double da = prev_to_cur_transform1[i].da + diff_a;


        new_prev_to_cur_transform3.push_back(TransformParam(dx, dy, da));

		
    }
    
    


x = 0;
y = 0;
a = 0;
    
for(size_t i=0; i < new_prev_to_cur_transform2.size(); i++)
    {
        x += prev_to_cur_transform2[i].dx;
        y += prev_to_cur_transform2[i].dy;
        a += prev_to_cur_transform2[i].da;

        // target - current
        double diff_x = optimal1[i].x - x;
        double diff_y = optimal1[i].y - y;
        double diff_a = optimal1[i].a - a;

        double dx = prev_to_cur_transform2[i].dx + diff_x;
        double dy = prev_to_cur_transform2[i].dy + diff_y;
        double da = prev_to_cur_transform2[i].da + diff_a;


        new_prev_to_cur_transform4.push_back(TransformParam(dx, dy, da));

		
    }
    
    VideoCapture cap(argv[1]);
    VideoCapture cap1(argv[2]);

  
   
    
   int max_frames0 = cap.get(CV_CAP_PROP_FRAME_COUNT);
   int max_frames1 = cap1.get(CV_CAP_PROP_FRAME_COUNT);
   
   int max_frames = (max_frames0 > max_frames1)? max_frames0: max_frames1;




   cap.set(CV_CAP_PROP_POS_FRAMES, 0);
   cap1.set(CV_CAP_PROP_POS_FRAMES, 0);
    
   Mat T(2,3,CV_64F);

   Mat T1(2,3,CV_64F),cur,cur1; 
  
   

    int k=0;
    
    int frame_width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH)); //get the width of frames of the video
    int frame_height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT)); //get the height of frames of the video
    
    Size frame_size(frame_width , frame_height);
    int frames_per_second = 30;

    string filename1 = "./MyVideo_out1.avi";
    string filename2 = "./MyVideo_out2.avi";    
    //Create and initialize the VideoWriter object 
    VideoWriter oVideoWriter(filename1, VideoWriter::fourcc('M', 'J', 'P', 'G'), 
                                                               frames_per_second, frame_size, true); 
    
    VideoWriter oVideoWriter1(filename2, VideoWriter::fourcc('M', 'J', 'P', 'G'), 
                                                               frames_per_second, frame_size, true); 
    int vert_border = HORIZONTAL_BORDER_CROP * 240 / 260; // get the aspect ratio correct
    
    //If the VideoWriter object is not initialized successfully, exit the program
    if (oVideoWriter.isOpened() == false || oVideoWriter1.isOpened() == false) 
    {
        cout << "Cannot save the video to a file" << endl;
        cin.get(); //wait for any key press
        return -1;
    }


    
    while(k < max_frames-2) { // don't process the very last frame, no valid transform
        cap >> cur;
        cap1 >> cur1;

        if(cur.data == NULL) {
            break;
        }
    

       

    T.at<double>(0,0) = cos(new_prev_to_cur_transform3[k].da);
    T.at<double>(0,1) = -sin(new_prev_to_cur_transform3[k].da);
    T.at<double>(1,0) = sin(new_prev_to_cur_transform3[k].da);
    T.at<double>(1,1) = cos(new_prev_to_cur_transform3[k].da);
    T.at<double>(0,2) = new_prev_to_cur_transform3[k].dx;
    T.at<double>(1,2) = new_prev_to_cur_transform3[k].dy;

    T1.at<double>(0,0) = cos(new_prev_to_cur_transform4[k].da);
    T1.at<double>(0,1) = -sin(new_prev_to_cur_transform4[k].da);
    T1.at<double>(1,0) = sin(new_prev_to_cur_transform4[k].da);
    T1.at<double>(1,1) = cos(new_prev_to_cur_transform4[k].da);
    T1.at<double>(0,2) = new_prev_to_cur_transform4[k].dx;
    T1.at<double>(1,2) = new_prev_to_cur_transform4[k].dy;
    
    Mat cur2,cur3;
	

    

    warpAffine(cur, cur2,T, frame_size);
    warpAffine(cur1,cur3,T1,frame_size);

    
    cur2 = cur2(Range(vert_border, cur2.rows-vert_border), Range(HORIZONTAL_BORDER_CROP, cur2.cols-HORIZONTAL_BORDER_CROP));

    //Resize cur2 back to cur size, for better side by side comparison
    resize(cur2, cur2, cur.size());

    cur3 = cur3(Range(vert_border, cur3.rows-vert_border), Range(HORIZONTAL_BORDER_CROP, cur3.cols-HORIZONTAL_BORDER_CROP));

    //Resize cur2 back to cur size, for better side by side comparison
    resize(cur3, cur3, cur.size());

    
    oVideoWriter.write(cur2);
    oVideoWriter1.write(cur3);
    
    cur2 = cur2(Range(vert_border, cur2.rows-vert_border), Range(HORIZONTAL_BORDER_CROP, cur2.cols-HORIZONTAL_BORDER_CROP));

        // Resize cur2 back to cur size, for better side by side comparison
    resize(cur2, cur2, cur.size());

        // Now draw the original and stablised side by side for coolness
    Mat canvas = Mat::zeros(cur1.rows, cur1.cols*2+10, cur.type());

    cur.copyTo(canvas(Range::all(), Range(0, cur2.cols)));
    cur2.copyTo(canvas(Range::all(), Range(cur2.cols+10, cur2.cols*2+10)));

 
//     // If too big to fit on the screen, then scale it down by 2, hopefully it'll fit :)
//     if(canvas.cols > 1920)
//     {
//         resize(canvas, canvas, Size(canvas.cols/2, canvas.rows/2));
//     }
// 
//     imshow("before and after", canvas);

//        // imshow("before and after", total);
//         
//         if (waitKey(10) == 27)
//         {
//             cout << "Esc key is pressed by the user. Stopping the video" << endl;
//             exit(-1);
//         }
// 
//       

        waitKey(10);

        k++;
    }

    oVideoWriter.release();
    oVideoWriter1.release();




    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    
    cout << "Total Time taken :" << time_spent << endl;
    


    
vector<Trajectory> trajectory3,trajectory4,new_T12;
vector<TransformParam> prev_to_cur_transform3,prev_to_cur_transform4,new_transform_1;
vector<Mat> C3,P3,C4,P4,new_T_map;
       
// new_stabilization(filename1,filename2,trajectory3,trajectory4, prev_to_cur_transform3,prev_to_cur_transform4,new_T12,new_transform_1,C3,P3,C4,P4,new_T_map);
    
// stabilization(filename1,filename2,trajectory3,trajectory4, prev_to_cur_transform3,prev_to_cur_transform4,C3,P3,C4,P4);

overlap(filename1,filename2,trajectory3, prev_to_cur_transform3,C3,P3);

a = 0;
 x = 0;
 y = 0;


    // Step 3 - Smooth out the trajectory using an averaging window
    vector <Trajectory> smoothed_trajectory3; // trajectory at all frames

    for(size_t i=0; i < trajectory3.size(); i++) {
        double sum_x = 0;
        double sum_y = 0;
        double sum_a = 0;
        int count = 0;

        for(int j=-SMOOTHING_RADIUS; j <= SMOOTHING_RADIUS; j++) {
            if(i+j >= 0 && i+j < trajectory3.size()) {
                sum_x += trajectory3[i+j].x;
                sum_y += trajectory3[i+j].y;
                sum_a += trajectory3[i+j].a;

                count++;
            }
        }

        double avg_a = sum_a / count;
        double avg_x = sum_x / count;
        double avg_y = sum_y / count;

        smoothed_trajectory3.push_back(Trajectory(avg_x, avg_y, avg_a));

//         out_smoothed_trajectory << (i+1) << " " << avg_x << " " << avg_y << " " << avg_a << endl;
    }

    // Step 4 - Generate new set of previous to current transform, such that the trajectory ends up being the same as the smoothed trajectory
    vector <TransformParam> new_prev_to_cur_transform7;

    // Accumulated frame to frame transform
    a = 0;
    x = 0;
    y = 0;
    
    
     VideoCapture cap3(filename1);
   VideoCapture cap4(filename2);
   
   char res[] = "New_output/", respath[70];

   int max_frames3 = cap3.get(CV_CAP_PROP_FRAME_COUNT);
   int max_frames4 = cap1.get(CV_CAP_PROP_FRAME_COUNT);

   int max_frames_new = (max_frames3 > max_frames4)? max_frames4: max_frames3;

 
   cap3.set(CV_CAP_PROP_POS_FRAMES, 0);
   cap4.set(CV_CAP_PROP_POS_FRAMES, 0);
    
   Mat T2(2,3,CV_64F),cur2;
   Mat T3(2,3,CV_64F),cur4;; 

   k=0;
    
    int frame_width1 = static_cast<int>(cap3.get(CAP_PROP_FRAME_WIDTH)); //get the width of frames of the video
    int frame_height1 = static_cast<int>(cap3.get(CAP_PROP_FRAME_HEIGHT)); //get the height of frames of the video
    
    Size frame_size1(frame_width1 , frame_height1);
    int frames_per_second1 = 30;

    string filename3 = "./out1.avi";
    string filename4 = "./out2.avi";
   
//     //Create and initialize the VideoWriter object 
    VideoWriter oVideoWriter3(filename3, VideoWriter::fourcc('M', 'J', 'P', 'G'), 
                                                               frames_per_second1, frame_size1, true); 

    VideoWriter oVideoWriter4(filename4, VideoWriter::fourcc('M', 'J', 'P', 'G'), 
                                                               frames_per_second1, frame_size1, true); 

    if (oVideoWriter3.isOpened() == false || oVideoWriter4.isOpened() == false)
    {
        cout << "Cannot save the video to a file" << endl;
        cin.get(); //wait for any key press
        return -1;
    }
     

    for(size_t i=0; i < prev_to_cur_transform3.size(); i++) {
        x += prev_to_cur_transform3[i].dx;
        y += prev_to_cur_transform3[i].dy;
        a += prev_to_cur_transform3[i].da;

        // target - current
        double diff_x = smoothed_trajectory3[i].x - x;
        double diff_y = smoothed_trajectory3[i].y - y;
        double diff_a = smoothed_trajectory3[i].a - a;

        double dx = prev_to_cur_transform3[i].dx + diff_x;
        double dy = prev_to_cur_transform3[i].dy + diff_y;
        double da = prev_to_cur_transform3[i].da + diff_a;

        new_prev_to_cur_transform7.push_back(TransformParam(dx, dy, da));

//         out_new_transform << (i+1) << " " << dx << " " << dy << " " << da << endl;
    

    
      // Step 3 - Smooth out the trajectory using an averaging window
//     vector <Trajectory> smoothed_trajectory4; // trajectory at all frames
// 
//     for(size_t i=0; i < trajectory4.size(); i++) {
//         double sum_x = 0;
//         double sum_y = 0;
//         double sum_a = 0;
//         int count = 0;
// 
//         for(int j=-SMOOTHING_RADIUS; j <= SMOOTHING_RADIUS; j++) {
//             if(i+j >= 0 && i+j < trajectory4.size()) {
//                 sum_x += trajectory4[i+j].x;
//                 sum_y += trajectory4[i+j].y;
//                 sum_a += trajectory4[i+j].a;
// 
//                 count++;
//             }
//         }
// 
//         double avg_a = sum_a / count;
//         double avg_x = sum_x / count;
//         double avg_y = sum_y / count;
// 
//         smoothed_trajectory4.push_back(Trajectory(avg_x, avg_y, avg_a));
// 
// //         out_smoothed_trajectory << (i+1) << " " << avg_x << " " << avg_y << " " << avg_a << endl;
//     }
// 
//     // Step 4 - Generate new set of previous to current transform, such that the trajectory ends up being the same as the smoothed trajectory
//     vector <TransformParam> new_prev_to_cur_transform8;
// 
//     // Accumulated frame to frame transform
//     a = 0;
//     x = 0;
//     y = 0;
// 
//     for(size_t i=0; i < prev_to_cur_transform4.size(); i++) {
//         x += prev_to_cur_transform4[i].dx;
//         y += prev_to_cur_transform4[i].dy;
//         a += prev_to_cur_transform4[i].da;
// 
//         // target - current
//         double diff_x = smoothed_trajectory4[i].x - x;
//         double diff_y = smoothed_trajectory4[i].y - y;
//         double diff_a = smoothed_trajectory4[i].a - a;
// 
//         double dx = prev_to_cur_transform4[i].dx + diff_x;
//         double dy = prev_to_cur_transform4[i].dy + diff_y;
//         double da = prev_to_cur_transform4[i].da + diff_a;
// 
//         new_prev_to_cur_transform8.push_back(TransformParam(dx, dy, da));
// 
// //         out_new_transform << (i+1) << " " << dx << " " << dy << " " << da << endl;
//     }

    
    
  
    if(k < max_frames_new-1) { // don't process the very last frame, no valid transform
        cap3 >> cur2;
        cap4 >> cur4;

        int vert_border = HORIZONTAL_BORDER_CROP * cur.rows / cur.cols; // get the aspect ratio correct
        if(cur2.data == NULL) {
            break;
        }
      
        T2.at<double>(0,0) = cos(new_prev_to_cur_transform7[i].da);
        T2.at<double>(0,1) = -sin(new_prev_to_cur_transform7[i].da);
        T2.at<double>(1,0) = sin(new_prev_to_cur_transform7[i].da);
        T2.at<double>(1,1) = cos(new_prev_to_cur_transform7[i].da);
        T2.at<double>(0,2) = new_prev_to_cur_transform7[i].dx;
        T2.at<double>(1,2) = new_prev_to_cur_transform7[i].dy;

        
//         T3.at<double>(0,0) = cos(new_prev_to_cur_transform8[k].da);
//         T3.at<double>(0,1) = -sin(new_prev_to_cur_transform8[k].da);
//         T3.at<double>(1,0) = sin(new_prev_to_cur_transform8[k].da);
//         T3.at<double>(1,1) = cos(new_prev_to_cur_transform8[k].da);
//         T3.at<double>(0,2) = new_prev_to_cur_transform8[k].dx;
//         T3.at<double>(1,2) = new_prev_to_cur_transform8[k].dy;





    Mat cur6,cur5;
    warpAffine(cur2,cur5,T2, frame_size1);
// //     warpAffine(cur4,cur6,T3,frame_size1);
    warpAffine(cur4,cur6,T2,frame_size1);
    //translateImg(cur5, 800, 1000); // 2000 is usual
    
   
            
    
//     sprintf(respath,"%s%d.jpg",res,k);
    
//     stitchImages(cur5,cur6,respath,new_T_map[k]);// new_transform_1[k].dx,new_transform_1[k].dy,new_transform_1[k].da);
    warp_crops(cur5,cur6);
    
   
//      if(k > 1)
//     {
//         Mat g1, g2;
//         cvtColor(cur5, g1, COLOR_BGR2GRAY);
//         cvtColor(prev, g2, COLOR_BGR2GRAY);
//         float error= mse(g2, g1);
//         out_stab_mse << " " << error << endl;
//     }
//     
//     Mat prev = cur2.clone();
    resize(cur5,cur5,frame_size1);
    
    oVideoWriter3.write(cur5);
//     oVideoWriter4.write(cur6);

        waitKey(10);

        k++;
    }
    }
    oVideoWriter3.release();
    oVideoWriter4.release();
    
    return max_frames;
 
}

int main(int argc, char **argv)
{

	ofstream out_fps("fps.txt");
	double t = (double) cv::getTickCount();
	int fc;
	fc= video(argc, argv);
	double afps = ((double) cv::getTickCount() - t) / cv::getTickFrequency();
	cout << "time= " << afps/(double)fc<<endl;
	double fps= 1.0/(afps/(double)fc);
	cout << "fps= " << 1.0/(afps/(double)fc)<<endl;
	cout << "f count= " << fc<<endl;
	out_fps << fps << endl;
}