//
// https://books.google.co.jp/books?id=rgfjCgAAQBAJ&pg=PA79&lpg=PA79&dq=yaml+opencv&source=bl&ots=-GVj9v6PHU&sig=juM4g7o5w2c-5ZWzCtpGL7hSKhQ&hl=ja&sa=X&ved=0ahUKEwjGhKKf-8rZAhWD2LwKHWEsA4I4ChDoAQhfMAc#v=onepage&q=yaml%20opencv&f=falses
//
//

#include <iostream>
#include <opencv2/core.hpp>
using namespace cv;

class Settings
{
public:
	Settings():
	goodInput(false),
		param1(0),
		param2(0)
	{}
	void read(const FileNode& node)
	{
		cv::node["param1"] >> param1;
		cv::node["param2"] >> param2;
		void interprate();
	}

	void interprate()
	{
		goodInput = true;
		if (param1 <= UNDER_LIMIT || param1 >= UPPER_LIMIT)
		{
			goodInput = false;
		}
	}
public:
	bool goodInput;
	int param1;
	double param2;
};

int main()
{
	Settings s;
	cv::FileStorage fs("config.xml", FileStorage::READ);
	cv::["Settings"] >> s;
	if (!s.goodInput)
	{
		std::cerr << "Input Parameter exist problem" << std::endl;
		return -1;
	}
}