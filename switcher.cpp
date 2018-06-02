
#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV

#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using ::boost::filesystem::path;

using namespace caffe;  // NOLINT(build/namespaces)

void DataAugmentation(cv::Mat img, string img_file);


#ifdef USE_OPENCV
int main(int argc, char** argv) {
	::google::InitGoogleLogging(argv[0]);

#ifndef GFLAGS_GFLAGS_H_
	namespace gflags = google;
#endif

	gflags::SetUsageMessage("Do augmentation using SSD mode.\n"
		"Usage:\n"
		"    ssd_augmentation [FLAGS] list_file\n");
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	if (argc < 2) {
		gflags::ShowUsageWithFlagsRestrict(argv[0], "examples/ssd/ssd_augmentation");
		return 1;
	}

	// https://www.buildinsider.net/small/opencv/005
	std::cout << cv::getBuildInformation() << std::endl;

	// Process image one by one.
	string filename = argv[3];
	std::ifstream infile(filename);

	// http://hwada.hatenablog.com/entry/20110611/1307781684
	path fullpath(filename);
	path rootdir = fullpath.root_directory();
	const string indir = rootdir.string();

	std::string img_file;
	string line;
	vector<std::string> fn;
	while (std::getline(infile, line)) {
		fn.push_back(line);
	}
	for (size_t k = 0; k < fn.size(); ++k)
	{
		img_file = fn[k];
		cv::Mat img = cv::imread(img_file);
		if (img.empty()) {
			std::cerr << "Failed to open image file." << std::endl;
			continue; //only proceed if sucsessful
		}

		DataAugmentation(img, img_file);
	}
	return 0;
}
#else
int main(int argc, char** argv) {
	LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
