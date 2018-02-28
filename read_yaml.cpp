//
// https://docs.opencv.org/3.0-beta/modules/core/doc/xml_yaml_persistence.html
// sources\samples\cpp\logistic_regression.cpp
// https://stackoverflow.com/questions/31381822/read-yaml-file-in-opencv?rq=1
//
#include <iostream>
#include <time.h>

#include <opencv2/core.hpp>
//#include <opencv2/ml.hpp>
//#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

//#define TEST1

int yaml_write() {
	FileStorage fs("test.yml", FileStorage::WRITE);

	fs << "frameCount" << 5;
	time_t rawtime; time(&rawtime);
	fs << "calibrationDate" << asctime(localtime(&rawtime));
	Mat cameraMatrix = (Mat_<double>(3, 3) << 1000, 0, 320, 0, 1000, 240, 0, 0, 1);
	Mat distCoeffs = (Mat_<double>(5, 1) << 0.1, 0.01, -0.001, 0, 0);
	fs << "cameraMatrix" << cameraMatrix << "distCoeffs" << distCoeffs;
	fs << "features" << "[";
	for (int i = 0; i < 3; i++)
	{
		int x = rand() % 640;
		int y = rand() % 480;
		uchar lbp = rand() % 256;

		fs << "{:" << "x" << x << "y" << y << "lbp" << "[:";
		for (int j = 0; j < 8; j++)
			fs << ((lbp >> j) & 1);
		fs << "]" << "}";
	}
	fs << "]";
	fs.release();

	return 0;
}

//frameCount: 5
//	transform_param :
//	mirror : true
//	mean_value : [104, 117, 123]
//	calibrationDate : "Tue Feb 27 22:00:51 2018\n"

int yaml_read() {
	FileStorage fs2("test.yml", FileStorage::READ);

	// first method: use (type) operator on FileNode.
	int frameCount = (int)fs2["frameCount"];

	string mirror = (string)fs2["transform_param"]["mirror"];
	//FileNode mean_value = fs2["transform_param"]["mean_value"];
	//FileNodeIterator it = mean_value.begin(), it_end = mean_value.end();
	//int idx = 0;
	//std::vector<int> lbpval;

	FileNode tm = fs2["transform_param"];

	vector<int> mean_value;
	FileNodeIterator ite = tm["mean_value"].begin();

	for (int k = 0; k < 3; k++, ++ite)
		mean_value.push_back((int)*ite);

	String date;
	// second method: use FileNode::operator >>
	fs2["calibrationDate"] >> date;

	Mat cameraMatrix2, distCoeffs2;
	fs2["cameraMatrix"] >> cameraMatrix2;
	fs2["distCoeffs"] >> distCoeffs2;

	cout << "frameCount: " << frameCount << endl
		<< "calibration date: " << date << endl
		<< "camera matrix: " << cameraMatrix2 << endl
		<< "distortion coeffs: " << distCoeffs2 << endl;

	FileNode features = fs2["features"];
	FileNodeIterator it = features.begin(), it_end = features.end();
	int idx = 0;
	std::vector<int> lbpval;

	// iterate through a sequence using FileNodeIterator
	for (; it != it_end; ++it, idx++)
	{
		cout << "feature #" << idx << ": ";
		cout << "x=" << (int)(*it)["x"] << ", y=" << (int)(*it)["y"] << ", lbp: (";
		// you can also easily read numerical arrays using FileNode >> std::vector operator.
		(*it)["lbp"] >> lbpval;
		for (int i = 0; i < (int)lbpval.size(); i++)
			cout << " " << (int)lbpval[i];
		cout << ")" << endl;
	}
	fs2.release();

	return 0;
}


int main()
{
	//yaml_write();
	yaml_read();

#ifdef TEST1
	const String filename = "../data/data01.xml";
	Mat data, labels;
	{
		cout << "loading the dataset...";
		FileStorage f;
		if (f.open(filename, FileStorage::READ))
		{
			f["datamat"] >> data;
			f["labelsmat"] >> labels;
			f.release();
		}
		else
		{
			cerr << "file can not be opened: " << filename << endl;
			return 1;
		}
	}
#else
	//FileStorage fs, fs2;
	FileStorage fs;
	fs.open("test2.yml", FileStorage::WRITE);
	fs << "features" << "[";
	for (unsigned int i = 0; i < 20; i++) {
		fs << 1.0 / (i + 1);
	}
	fs << "]";
	fs.release();

	FileStorage fs2;
	fs2.open("test2.yml", FileStorage::READ);
	vector<float> a;
	fs2["features"] >> a;
#endif
}