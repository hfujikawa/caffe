//
// https://boostjp.github.io/tips/ini.html
//
#include <iostream>
#include <string>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/optional.hpp>

using namespace boost::property_tree;

int main()
{
	ptree pt;
	read_ini("D:\\Develop\\MobileNet-SSD-windows\\examples\\ssd\\data.ini", pt);

	if (boost::optional<int> value = pt.get_optional<int>("Data.value")) {
		std::cout << "value : " << value.get() << std::endl;
	}
	else {
		std::cout << "value is nothing" << std::endl;
	}

	if (boost::optional<std::string> str = pt.get_optional<std::string>("Data.str")) {
		std::cout << "str : " << str.get() << std::endl;
	}
	else {
		std::cout << "str is nothing" << std::endl;
	}
}
