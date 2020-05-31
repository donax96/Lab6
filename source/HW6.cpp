#include<opencv2/opencv.hpp>
#include<iostream>
#include "../include/trackingImages.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	TrackingImages video;
	video.load_video(argv[1]);
	video.locate_objects(argv[2]);
	
	video.track_motion();

	//video.save_video("Output.mov");

	cout << "Everything went fine" << endl;

	return 0;
}