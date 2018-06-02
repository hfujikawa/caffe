//
// http://answers.opencv.org/question/182919/how-to-read-json-files-using-opencv-filestorage-with-c-program/
//
#include <iostream>
#include <time.h>

#include <opencv2/core.hpp>
//#include <opencv2/ml.hpp>
//#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;


struct stData {
	float a, b, c;
};

int main(int argc, char** argv)
{
	vector<vector<stData> > D; // use a dynamic container
	FileStorage fs("data.json", 0);
	FileNode root = fs["points"];
	for (int i = 0; i < root.size(); i++) {
		FileNode val1 = root[i]["val1"]; // we only want the content of val1
		vector<stData> row;
		for (int j = 0; j < val1.size(); j += 3) { // read 3 consecutive values
			stData  d;
			d.a = val1[j].REAL;
			d.b = val1[j + 1].REAL;
			d.c = val1[j + 2].REAL;
			row.push_back(d);
		}
		D.push_back(row);
		cout << "row: " << i << endl;
	}
}