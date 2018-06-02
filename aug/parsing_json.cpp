//
// https://whatevericode.wordpress.com/2013/05/22/parsing-jason-nested-array-using-jsoncpp/
//
#include <iostream>
#include <fstream>
#include <string>
#include <json/reader.h>

using namespace std;

int main() {
	unsigned int i, j;
	char fileName[] = "abc.json";

	ifstream infile(fileName);
	infile.exceptions(ios::failbit | ios::badbit);
	try
	{
		infile.open(fileName);
	}
	catch (ifstream::failure& e)
	{
		cerr << "\n Exception opening file " << fileName << ": " << e.what();
		return 0;
	}

	string input;
	Json::Reader reader;
	Json::Value root;
	bool parsingSuccessful;

	try
	{
		while (getline(infile, input, '\n'))
		{
			if (false == (parsingSuccessful = reader.parse(input, root)))
			{
				cerr << "\nFailed to parse configuration:"
					<< reader.getFormatedErrorMessages();
				return 0;
			}
			else
			{

				int verbName = root.get("id", "").asInt();
				Json::Value data = root.get("tags", 0);
				cout << "size is " << data.size() << endl;

				//vector<float> freq (mWidth * mHeight);
				for (int i = 0; i < data.size(); i++) {
					int sizeint = data[i].size();
					cout << endl << "---------------" << endl;
					cout << "Size of individual is " << sizeint << endl;
					Json::Value dat1 = data[i];

					for (int j = 0; j< dat1.size(); j++) {
						cout << dat1[j].asDouble() << " ,";
					}

				}
			}
		}
	}
	catch (ifstream::failure& e)
	{

	}

	return 0;
}
