
#include <caffe/caffe.hpp>

#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using ::boost::filesystem::path;

using namespace caffe;  // NOLINT(build/namespaces)

void DataAugmentation(vector<string> img_files, vector<string> anno_files);


#ifdef USE_OPENCV
int test(int argc, char** argv) {
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

	// Process image one by one.
	std::ifstream infile;
	string line;
	vector<std::string> fin, fan;
	// https://docs.oracle.com/cd/E19957-01/805-7889/z4000016dc674/index.html
	for (char** f = &argv[1]; *f; ++f) {
		infile.open(*f, ios::in);
		string line;
		while (std::getline(infile, line)) {
			std::cout << line << std::endl;
			path img_path(line);
			const string labelfile = (img_path.replace_extension(".xml")).string();

			fin.push_back(line);
			fan.push_back(labelfile);
		}
		infile.close();
	}
/*	string filename = argv[1];
	//std::ifstream infile(filename);
	// http://hwada.hatenablog.com/entry/20110611/1307781684
	path fullpath(filename);
	path rootdir = fullpath.root_directory();
	const string indir = rootdir.string();
	*/
	DataAugmentation(fin, fan);
	return 0;
}
#else
int main(int argc, char** argv) {
	LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
