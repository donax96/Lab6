#include <opencv2/core/cvstd.hpp>
#include <opencv2/videoio.hpp>

class TrackingImages
{
public:

	TrackingImages();

	void load_video(cv::String path);

	void locate_objects(cv::String object_paths);

	void track_motion();

	void save_video(cv::String path);

protected:

	std::vector<cv::Mat> frames, objects, equalized_objects;
	std::vector<cv::Mat> obj_descriptors, frames_descriptors;
	std::vector<std::vector<cv::KeyPoint>> obj_keypoints, frames_keypoints;
	std::vector<std::vector<cv::Point2f>> best_matches, obj_corners;
	std::vector<cv::Scalar> colors;
	
private:
	int fourcc;

};
