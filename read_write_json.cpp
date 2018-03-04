//
// https://stackoverflow.com/questions/2114466/creating-json-arrays-in-boost-using-property-trees
// http://zenol.fr/blog/boost-property-tree/en.html
// https://boostjp.github.io/tips/json.html
// https://boostjp.github.io/tips/foreach.html
//
#include <iostream>
#include <string>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>
#include <boost/optional.hpp>

using namespace boost::property_tree;

std::string json_file = "data_array.json";

int main()
{
	ptree pt;
	read_json(json_file, pt);

	// Get image info.
	int width = 0, height = 0;
	try {
		height = pt.get<int>("image.height");
		width = pt.get<int>("image.width");
	}
	catch (const ptree_error &e) {
		//LOG(WARNING) << "When parsing " << labelfile << ": " << e.what();
		//height = img_height;
		//width = img_width;
	}
}

int write_json()
{
	ptree targetTree;
	ptree arrayChild;

	//add array elements as desired, loop, whatever, for example
	{
		ptree arrayElement;
		arrayElement.put("min_scale", 1);
		arrayElement.put("max_trials", 1);
		arrayChild.push_back(std::make_pair("", arrayElement));
	}
	{
		ptree arrayElement;
		arrayElement.put("min_scale", 1);
		arrayElement.put("max_trials", 50);
		ptree sampler;
		sampler.put("sub1", 2);
		arrayElement.push_back(std::make_pair("sampler", sampler));
		arrayChild.push_back(std::make_pair("", arrayElement));
	}
	{
		ptree arrayElement;
		arrayElement.put("min_scale", 1);
		arrayElement.put("max_trials", 50);
		ptree sampler;
		sampler.put("sub2", 3);
		arrayElement.push_back(std::make_pair("sampler", sampler));
		arrayChild.push_back(std::make_pair("", arrayElement));
	}
	targetTree.put_child(ptree::path_type("batch_sampler"), arrayChild);
	write_json(json_file, targetTree);

/*	ptree pt;

	//pt.put("bt", 1);
	//pt.put("Data.value", 3);
	//pt.put("Data.str", "Hello");

	ptree child;
	{
		ptree info;
		info.put("id", 1);
		info.put("name", "Alice");
		child.push_back(std::make_pair("", info));
	}
	{
		ptree info;
		info.put("id", 2);
		info.put("name", "Millia");
		child.push_back(std::make_pair("", info));
	}
	pt.add_child("Data.info", child);

	write_json("data_out.json", pt);
	*/

	return 0;
}