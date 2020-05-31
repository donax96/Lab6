#include "../include/trackingImages.h"
#include<iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/video/tracking.hpp>

using namespace cv;
using namespace std;

TrackingImages::TrackingImages()
{
}

void TrackingImages::load_video(cv::String path)
{
	cout << endl << "Loading video..." << endl << endl;

	VideoCapture cap(path);

	if (cap.isOpened()) // check if we succeeded
	{

		fourcc = static_cast<int>(cap.get(CAP_PROP_FOURCC));

		for (;;)
		{
			Mat frame;
			cap >> frame;

			if (frame.empty())
				break;

			frames.push_back(frame);
		}
	}
	else
		cout << endl << "Error in loading video" << endl << endl;

}
	

void TrackingImages::locate_objects(String object_path)
{
	cout << endl << "Loadind images objects..." << endl << endl;

	String pattern = "*.png";
	vector<String> names_objects;
	utils::fs::glob(object_path, pattern, names_objects);

	int i;
	for (i = 0; i < names_objects.size(); i++)
		objects.push_back(imread(names_objects[i]));

	cout << endl << "Extracting features..." << endl << endl;

	Ptr<ORB> orb = ORB::create();
	orb->setMaxFeatures(5000);

	//extract orb feature of the objects
	for (i = 0; i < objects.size(); i++)
	{
		Mat dsp;
		vector<KeyPoint> key;

		orb->detectAndCompute(objects[i], Mat(), key, dsp);

		obj_descriptors.push_back(dsp);
		obj_keypoints.push_back(key);
	}

	//extract orb feature of the frames in the video

	Mat dsp;
	vector<KeyPoint> key;
	orb->setMaxFeatures(20000);

	orb->detectAndCompute(frames[0], Mat(), key, dsp);

	frames_descriptors.push_back(dsp);
	frames_keypoints.push_back(key);

	cout << endl << "Computing matches..." << endl << endl;
	
	vector<vector<DMatch>> matches;
	Ptr<BFMatcher> bf = BFMatcher::create(NORM_HAMMING, true);

	vector<float> min_distances;

	//Computing the matches for each objects
	for (i = 0; i < objects.size(); i++)
	{
		vector<DMatch> match;
		bf->match(obj_descriptors[i], frames_descriptors[0], match, Mat());
		matches.push_back(match);


		int j;
		min_distances.push_back(std::numeric_limits<float>::max());

		for (j = 0; j < match.size(); j++)
		{
			if (match[j].distance < min_distances[i])
				min_distances[i] = match[j].distance;
		}

	}

	//Selecting the matches

	cout << endl << "Selecting matches..." << endl << endl;

	//devo selezionare le features per ogni oggetto 

	Mat features_image = frames[0].clone();

	colors.push_back(Scalar(0, 0, 255));//red
	colors.push_back(Scalar(0, 255, 0));//green
	colors.push_back(Scalar(255, 0, 0));//blu
	colors.push_back(Scalar(168, 50, 129));//violet
	
	for (i = 0; i < matches.size(); i++)
	{
		vector<Point2f> obj;
		vector<Point2f> frame;
		Mat mask;

		int j;
		for (j = 0; j < matches[i].size(); j++)
		{
			if (matches[i][j].distance <= (2 * min_distances[i]))
			{
				obj.push_back(obj_keypoints[i][matches[i][j].queryIdx].pt);
				frame.push_back(frames_keypoints[0][matches[i][j].trainIdx].pt);
			}
		}

		Mat H = findHomography(obj, frame, mask, RANSAC, 1*min_distances[i]);

		vector<Point2f> corners;
		corners.push_back(Point2f(0, 0));
		corners.push_back(Point2f( objects[i].cols - 1, 0));
		corners.push_back(Point2f(0, objects[i].rows - 1));
		corners.push_back(Point2f(objects[i].cols - 1, objects[i].rows - 1));

		vector<Point2f> video_corners;
		perspectiveTransform(corners, video_corners, H);
		obj_corners.push_back(video_corners);

		line(features_image, video_corners[0], video_corners[1], colors[i]);
		line(features_image, video_corners[1], video_corners[3], colors[i]);
		line(features_image, video_corners[2], video_corners[3], colors[i]);
		line(features_image, video_corners[2], video_corners[0], colors[i]);

		vector<Point2f> matches;

		int k;
		int count = 0;
		for (k = 0; k < mask.rows; k++)
		{
			if ((unsigned int)mask.at<uchar>(k))
			{
				matches.push_back(frame[k]);
			}
		}

		best_matches.push_back(matches);
	}

	//draw the feature
	for (i = 0; i < best_matches.size(); i++)
	{
		int j;
		for (j = 0; j < best_matches[i].size(); j++)
		{
			circle(features_image, best_matches[i][j], 2, colors[i], 2);
		}
	}

	//resize(features_image, features_image, Size(features_image.cols / 2, features_image.rows / 2));
	//namedWindow("Features");
	//imshow("Features", features_image);
	//waitKey(0);
	//destroyAllWindows();

}

void TrackingImages::track_motion()
{
	vector<vector<vector<Point2f>>> moving_points(best_matches.size());
	// Devo distinguere il movimento dei singoli libri o faccio tutto insieme?
	for (int i = 0; i < best_matches.size(); i++)
	{
		vector<Point2f> temp;
		for (int j = 0; j < best_matches[i].size(); j++)
		{
			temp.push_back(best_matches[i][j]);
		}

		/* //Actually if I artficially add the detected corners of the object the result is even worse
		//add detected and matched corners as features
		for (int j = 0; j < 4; j++)
		{
			Point2f corner = obj_corners[i][j];
			if (corner.x > frames[0].cols)
				corner.x = frames[0].cols;
			if (corner.y > frames[0].rows)
				corner.y = frames[0].rows;
			temp.push_back(corner);
		}
		*/
		moving_points[i].push_back(temp);
	}

	for (int i = 1; i < frames.size(); i++)
	{
		for (int book = 0; book < best_matches.size(); book++)
		{
			Mat prevGray, gray;
			cvtColor(frames[i - 1], prevGray, COLOR_BGR2GRAY);
			cvtColor(frames[i], gray, COLOR_BGR2GRAY);

			vector<uchar> status;
			vector<float> err;

			vector<Point2f> calculated_points;
			Size winSize = Size(18, 11);
			calcOpticalFlowPyrLK(prevGray, gray, moving_points[book][i - 1], calculated_points, status, err, winSize, 3);
			moving_points[book].push_back(calculated_points);

			Mat H = findHomography(moving_points[book][i - 1], calculated_points, noArray(), RANSAC, 1.4);

			vector<Point2f> video_corners = obj_corners[book], video_corners_new;
			perspectiveTransform(video_corners, video_corners_new, H);

			//update corners, otherwise the reprojection would be incorrect
			obj_corners[book] = video_corners_new;

			//draw feature points
			for (int j = 0; j < calculated_points.size(); j++)
			{
				Point p;
				p.x = (int)calculated_points[j].x;
				p.y = (int)calculated_points[j].y;
				circle(frames[i], p, 3, colors[book], -1);
			}

			//draw the rectangle
			line(frames[i], video_corners_new[0], video_corners_new[1], colors[book]);
			line(frames[i], video_corners_new[1], video_corners_new[3], colors[book]);
			line(frames[i], video_corners_new[2], video_corners_new[3], colors[book]);
			line(frames[i], video_corners_new[2], video_corners_new[0], colors[book]);

			
		}
		if (i > 100 && i % 40 == 0)
		{
			//cout << "Homography matrix:" << endl << H << endl;

			imshow("Prova", frames[i]);
			waitKey(0);
		}
	}
}

void TrackingImages::save_video(cv::String path)
{
	VideoWriter vw;
	double fps = 30.;

	vw.open(path, fourcc, fps, frames[0].size(), true);

	if (!vw.isOpened()) {
		cout << "Could not open the output video for write: " << endl;
		exit(1);
	}


	for (int i = 0; i < frames.size(); i++)
	{
		vw << frames[i];
	}
}
