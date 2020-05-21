#include <opencv2/core/cvstd.hpp>
#include <opencv2/videoio.hpp>

class TrackingImages
{
public:

	TrackingImages();

	void load_video(cv::String path);

	void locate_objects(cv::String object_paths);

protected:

	std::vector<cv::Mat> frames, objects, equalized_objects;
	std::vector<cv::Mat> obj_descriptors, frames_descriptors;
	std::vector<std::vector<cv::KeyPoint>> obj_keypoints, frames_keypoints;

};
