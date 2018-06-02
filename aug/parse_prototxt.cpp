#include <io.h>
#include <iostream>
#include <fcntl.h>
#include <fstream>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include "caffe/proto/caffe.pb.h"

using namespace std;
using namespace caffe;

int main(int argc, char* argv[])
{

	// Verify that the version of the library that we linked against is
	// compatible with the version of the headers we compiled against.
	GOOGLE_PROTOBUF_VERIFY_VERSION;

	Tasking *tasking = new Tasking(); //My protobuf object

	bool retValue = false;

	int fileDescriptor = open(argv[1], O_RDONLY);

	if (fileDescriptor < 0)
	{
		std::cerr << " Error opening the file " << std::endl;
		return false;
	}

	google::protobuf::io::FileInputStream fileInput(fileDescriptor);
	fileInput.SetCloseOnDelete(true);

	if (!google::protobuf::TextFormat::Parse(&fileInput, tasking))
	{
		cerr << std::endl << "Failed to parse file!" << endl;
		return -1;
	}
	else
	{
		retValue = true;
		cerr << "Read Input File - " << argv[1] << endl;
	}

	cerr << "Id -" << tasking->taskid() << endl;
}