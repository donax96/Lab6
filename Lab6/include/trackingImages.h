#include <opencv2/core/cvstd.hpp>
#include <opencv2/videoio.hpp>

class TrackingImages
{
public:

	TrackingImages();

	void load_video(cv::String path);

	void locate_objects(cv::String object_paths);

	void track_motion();

protected:

	std::vector<cv::Mat> frames, objects, equalized_objects;
	std::vector<cv::Mat> obj_descriptors;
	cv::Mat frames_descriptors, copy_frame;
	std::vector<cv::KeyPoint> frames_keypoints;
	std::vector<std::vector<cv::KeyPoint>> obj_keypoints;
	std::vector<std::vector<cv::Point2f>> best_matches, obj_corners;
	std::vector<cv::Scalar> colors;

};
