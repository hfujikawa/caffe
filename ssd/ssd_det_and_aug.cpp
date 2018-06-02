// This is a demo code for using a SSD model to do detection.
// The code is modified from examples/cpp_classification/classification.cpp.
// Usage:
//    ssd_detect [FLAGS] model_file weights_file list_file
//
// where model_file is the .prototxt file defining the network architecture, and
// weights_file is the .caffemodel file containing the network parameters, and
// list_file contains a list of image files with the format as follows:
//    folder/img1.JPEG
//    folder/img2.JPEG
// list_file can also contain a list of video files with the format as follows:
//    folder/video1.mp4
//    folder/video2.mp4
//
#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#endif  // USE_OPENCV

#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "caffe/blob.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/bbox_util.hpp"
#include "caffe/util/im_transforms.hpp"
#include "caffe/util/sampler.hpp"

char* CLASSES[21] = { "__background__",
		   "aeroplane", "bicycle", "bird", "boat",
		   "bottle", "bus", "car", "cat", "chair",
		   "cow", "diningtable", "dog", "horse",
		   "motorbike", "person", "pottedplant",
		   "sheep", "sofa", "train", "tvmonitor" };
// https://stackoverflow.com/questions/8520560/get-a-file-name-from-a-path
using ::boost::filesystem::path;

struct MatchPathSeparator
{
	bool operator()(char ch) const
	{
		return ch == '\\' || ch == '/';
	}
};

std::string basename(std::string const& pathname)
{
	return std::string(
		std::find_if(pathname.rbegin(), pathname.rend(),
			MatchPathSeparator()).base(),
		pathname.end());
}
/*
// https://stackoverflow.com/questions/24734625/how-to-split-a-path-into-separate-strings
std::vector<std::string> SplitPath(const path &src) {
	std::vector<std::string> elements;
	for (const auto &p : src) {
		elements.emplace_back(p.filename());
	}
	return elements;
}
*/
#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)

class Augmentation {
public:
	Augmentation(const string& param_file,
		const string& mean_file,
		const string& mean_value);

	void InitRand();

	//void Transform(const Datum& datum, Blob<float>* transformed_blob);

	//void Transform(const vector<Datum> & datum_vector,
	//	Blob<float>* transformed_blob);

	//void Transform(const AnnotatedDatum& anno_datum,
	//	Blob<float>* transformed_blob,
	//	::google::protobuf::RepeatedPtrField<AnnotationGroup>* transformed_anno_vec);
	void TransformCrop(const AnnotatedDatum& anno_datum,
		Blob<float>* transformed_blob,
		//AnnotationGroup* transfomed_anno_datum,
		::google::protobuf::RepeatedPtrField<AnnotationGroup>* transformed_anno_vec,
		bool* do_mirror);
	//void Transform(const AnnotatedDatum& anno_datum,
	//	Blob<float>* transformed_blob,
	//	vector<AnnotationGroup>* transformed_anno_vec,
	//	bool* do_mirror);
	//void Transform(const AnnotatedDatum& anno_datum,
	//	Blob<float>* transformed_blob,
	//	vector<AnnotationGroup>* transformed_anno_vec);

	void TransformAnnotation(
		const AnnotatedDatum& anno_datum, const bool do_resize,
		const NormalizedBBox& crop_bbox, const bool do_mirror,
		//AnnotationGroup* transformed_anno_group);
		//AnnotatedDatum* transformed_anno_datum);
		::google::protobuf::RepeatedPtrField<AnnotationGroup>* transformed_anno_group_all);

	void CropImage(const Datum& datum, const NormalizedBBox& bbox,
		Datum* crop_datum);

	void CropImage(const AnnotatedDatum& anno_datum, const NormalizedBBox& bbox,
		AnnotatedDatum* cropped_anno_datum);

	void ExpandImage(const Datum& datum, const float expand_ratio,
		NormalizedBBox* expand_bbox, Datum* expanded_datum);

	void ExpandImage(const AnnotatedDatum& anno_datum,
		AnnotatedDatum* expanded_anno_datum);

	void DistortImage(const Datum& datum, Datum* distort_datum);

#ifdef USE_OPENCV
	//void Transform(const vector<cv::Mat> & mat_vector,
	//	Blob<float>* transformed_blob);

	//void Transform(const cv::Mat& cv_img, Blob<float>* transformed_blob,
	//	NormalizedBBox* crop_bbox, bool* do_mirror);
	//void Transform(const cv::Mat& cv_img, Blob<float>* transformed_blob);
	void CropImage(const cv::Mat& img, const NormalizedBBox& bbox,
		cv::Mat* crop_img);

	void ExpandImage(const cv::Mat& img, const float expand_ratio,
		NormalizedBBox* expand_bbox, cv::Mat* expand_img);
#endif  // USE_OPENCV
	//void Transform(Blob<float>* input_blob, Blob<float>* transformed_blob);

	//vector<int> InferBlobShape(const Datum& datum);
	//vector<int> InferBlobShape(const vector<Datum> & datum_vector);

#ifdef USE_OPENCV
	//vector<int> InferBlobShape(const vector<cv::Mat> & mat_vector);
	//vector<int> InferBlobShape(const cv::Mat& cv_img);
#endif  // USE_OPENCV

	virtual int Rand(int n);

	//void Transform(const Datum& datum, float* transformed_data,
	//	NormalizedBBox* crop_bbox, bool* do_mirror);
	//void Transform(const Datum& datum, float* transformed_data);

	void TransformCrop(const Datum& datum, Blob<float>* transformed_blob,
		NormalizedBBox* crop_bbox, bool* do_mirror);

	LayerParameter layer_param_;
	// Tranformation parameters
	TransformationParameter param_;
	vector<BatchSampler> batch_samplers_;

	shared_ptr<Caffe::RNG> rng_;
	Phase phase_;
	Blob<float> data_mean_;
	vector<float> mean_values_;
};


class Detector {
 public:
  Detector(const string& model_file,
           const string& weights_file,
           const string& mean_file,
           const string& mean_value,
		   const float confidence_threshold,
		   const float normalize_value);

  std::vector<vector<float> > Detect(const cv::Mat& img);

 private:
  void SetMean(const string& mean_file, const string& mean_value);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);
  void Preprocess(const cv::Mat& img,
	  std::vector<cv::Mat>* input_channels,double normalize_value);
 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
  float nor_val = 1.0;
};

Detector::Detector(const string& model_file,
                   const string& weights_file,
                   const string& mean_file,
                   const string& mean_value,
				   const float confidence_threshold,
				   const float normalize_value) {
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(weights_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  /* Load the binaryproto mean file. */
  SetMean(mean_file, mean_value);
  nor_val = normalize_value;
}

std::vector<vector<float> > Detector::Detect(const cv::Mat& img) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);
  if (nor_val != 1.0) {
	  Preprocess(img, &input_channels, nor_val);
  }
  else {
	  Preprocess(img, &input_channels);
  }

  net_->Forward();

  /* Copy the output layer to a std::vector */
  Blob<float>* result_blob = net_->output_blobs()[0];
  const float* result = result_blob->cpu_data();
  const int num_det = result_blob->height();
  vector<vector<float> > detections;
  for (int k = 0; k < num_det; ++k) {
    if (result[0] == -1) {
      // Skip invalid detection.
      result += 7;
      continue;
    }
    vector<float> detection(result, result + 7);
    detections.push_back(detection);
    result += 7;
  }
  return detections;
}

/* Load the mean file in binaryproto format. */
void Detector::SetMean(const string& mean_file, const string& mean_value) {
  cv::Scalar channel_mean;
  if (!mean_file.empty()) {
    CHECK(mean_value.empty()) <<
      "Cannot specify mean_file and mean_value at the same time";
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

    /* Convert from BlobProto to Blob<float> */
    Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);
    CHECK_EQ(mean_blob.channels(), num_channels_)
      << "Number of channels of mean file doesn't match input layer.";

    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
    std::vector<cv::Mat> channels;
    float* data = mean_blob.mutable_cpu_data();
    for (int i = 0; i < num_channels_; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
      channels.push_back(channel);
      data += mean_blob.height() * mean_blob.width();
    }

    /* Merge the separate channels into a single image. */
    cv::Mat mean;
    cv::merge(channels, mean);

    /* Compute the global mean pixel value and create a mean image
     * filled with this value. */
    channel_mean = cv::mean(mean);
    mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
  }
  if (!mean_value.empty()) {
    CHECK(mean_file.empty()) <<
      "Cannot specify mean_file and mean_value at the same time";
    stringstream ss(mean_value);
    vector<float> values;
    string item;
    while (getline(ss, item, ',')) {
      float value = std::atof(item.c_str());
      values.push_back(value);
    }
    CHECK(values.size() == 1 || values.size() == num_channels_) <<
      "Specify either 1 mean_value or as many as channels: " << num_channels_;

    std::vector<cv::Mat> channels;
    for (int i = 0; i < num_channels_; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1,
          cv::Scalar(values[i]));
      channels.push_back(channel);
    }
    cv::merge(channels, mean_);
  }
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Detector::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Detector::Preprocess(const cv::Mat& img,
	std::vector<cv::Mat>* input_channels) {
	/* Convert the input image to the input image format of the network. */
	cv::Mat sample;
	if (img.channels() == 3 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
	else if (img.channels() == 4 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
	else if (img.channels() == 4 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
	else if (img.channels() == 1 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
	else
		sample = img;

	cv::Mat sample_resized;
	if (sample.size() != input_geometry_)
		cv::resize(sample, sample_resized, input_geometry_);
	else
		sample_resized = sample;

	cv::Mat sample_float;
	if (num_channels_ == 3)
		sample_resized.convertTo(sample_float, CV_32FC3);
	else
		sample_resized.convertTo(sample_float, CV_32FC1);

	cv::Mat sample_normalized;
	cv::subtract(sample_float, mean_, sample_normalized);

	/* This operation will write the separate BGR planes directly to the
	* input layer of the network because it is wrapped by the cv::Mat
	* objects in input_channels. */
	cv::split(sample_normalized, *input_channels);

	CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
		== net_->input_blobs()[0]->cpu_data())
		<< "Input channels are not wrapping the input layer of the network.";
}

void Detector::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels, double normalize_value) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3, normalize_value);
  else
    sample_resized.convertTo(sample_float, CV_32FC1, normalize_value);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}


DEFINE_string(mean_file, "",
	"The mean file used to subtract from the input image.");
DEFINE_string(mean_value, "104,117,123",
	"If specified, can be one value or can be same as image channels"
	" - would subtract from the corresponding channel). Separated by ','."
	"Either mean_file or mean_value should be provided, not both.");
DEFINE_string(file_type, "image",
	"The file type in the list_file. Currently support image and video.");
DEFINE_string(out_file, "",
	"If provided, store the detection results in the out_file.");
DEFINE_double(confidence_threshold, 0.90,
	"Only store detections with score higher than the threshold.");
DEFINE_double(normalize_value, 1.0,
	"Normalize image to 0~1");
DEFINE_int32(wait_time, 1000,
	"cv imshow window waiting time ");

void DetectObject(cv::Mat img, string img_file) {
	// Print output to stderr (while still logging)
	FLAGS_alsologtostderr = 1;

	const string& model_file = "D:\\Develop\\MobileNet-SSD-windows\\models\\VGGNet\\VOC0712\\SSD_300x300\\deploy.prototxt";
	const string& weights_file = "D:\\Develop\\MobileNet-SSD-windows\\models\\VGGNet\\VOC0712\\SSD_300x300\\VGG_VOC0712_SSD_300x300_iter_120000.caffemodel";
	const string& mean_file = FLAGS_mean_file;
	const string& mean_value = FLAGS_mean_value;
	const string& out_file = FLAGS_out_file;
	const float& confidence_threshold = FLAGS_confidence_threshold;
	const float& normalize_value = FLAGS_normalize_value;
	const int& wait_time = FLAGS_wait_time;

	// Set the output mode.
	std::streambuf* buf = std::cout.rdbuf();
	std::ofstream outfile;
	if (!out_file.empty()) {
		outfile.open(out_file.c_str());
		if (outfile.good()) {
			buf = outfile.rdbuf();
		}
	}
	std::ostream out(buf);

	// Initialize the network.
	Detector detector(model_file, weights_file, mean_file, mean_value, confidence_threshold, normalize_value);

	// you probably want to do some preprocessing
	CHECK(!img.empty()) << "Unable to decode image " << img_file;
	std::vector<vector<float> > detections = detector.Detect(img);

	/* Print the detection results. */
	for (int i = 0; i < detections.size(); ++i) {
		const vector<float>& d = detections[i];
		// Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
		CHECK_EQ(d.size(), 7);
		const float score = d[2];
		if (score >= confidence_threshold) {
			out << img_file << " ";
			out << static_cast<int>(d[1]) << " ";
			out << score << " ";
			out << static_cast<int>(d[3] * img.cols) << " ";
			out << static_cast<int>(d[4] * img.rows) << " ";
			out << static_cast<int>(d[5] * img.cols) << " ";
			out << static_cast<int>(d[6] * img.rows) << std::endl;

			cv::Point pt1, pt2;
			pt1.x = (img.cols*d[3]);
			pt1.y = (img.rows*d[4]);
			pt2.x = (img.cols*d[5]);
			pt2.y = (img.rows*d[6]);
			cv::rectangle(img, pt1, pt2, cvScalar(0, 255, 0), 1, 8, 0);

			char label[100];
			sprintf(label, "%s,%f", CLASSES[static_cast<int>(d[1])], score);
			int baseline;
			cv::Size size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 0, &baseline);
			cv::Point pt3;
			pt3.x = pt1.x + size.width;
			pt3.y = pt1.y - size.height;
			cv::rectangle(img, pt1, pt3, cvScalar(0, 255, 0), -1);

			cv::putText(img, label, pt1, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
		}
	}
	cv::imshow("show", img);
	cv::waitKey(wait_time);
}

/*
 * Augmentation for SSD
 */
Augmentation::Augmentation(const string& param_file,
	const string& mean_file,
	const string& mean_value){

	// Set parameters
	NetParameter param;
	ReadNetParamsFromTextFileOrDie(param_file, &param);
	layer_param_ = param.layer(0);
	phase_ = layer_param_.phase();
	int batch_size = layer_param_.data_param().batch_size();
	const AnnotatedDataParameter& anno_data_param = layer_param_.annotated_data_param();
	for (int i = 0; i < anno_data_param.batch_sampler_size(); ++i) {
		batch_samplers_.push_back(anno_data_param.batch_sampler(i));
	}
	//string label_map_file_ = anno_data_param.label_map_file();
	// Make sure dimension is consistent within batch.
	param_ = layer_param_.transform_param();
	if (param_.has_resize_param()) {
		if (param_.resize_param().resize_mode() == ResizeParameter_Resize_mode_FIT_SMALL_SIZE) {
			CHECK_EQ(batch_size, 1)
				<< "Only support batch size of 1 for FIT_SMALL_SIZE.";
		}
	}
}

void Augmentation::InitRand() {
	const bool needs_rand = param_.mirror() ||
		(phase_ == TRAIN && param_.crop_size());
	if (needs_rand) {
		const unsigned int rng_seed = caffe_rng_rand();
		rng_.reset(new Caffe::RNG(rng_seed));
	}
	else {
		rng_.reset();
	}
}

int Augmentation::Rand(int n) {
	CHECK(rng_);
	CHECK_GT(n, 0);
	caffe::rng_t* rng =
		static_cast<caffe::rng_t*>(rng_->generator());
	return ((*rng)() % n);
}

void Augmentation::TransformAnnotation(
	const AnnotatedDatum& anno_datum, const bool do_resize,
	const NormalizedBBox& crop_bbox, const bool do_mirror,
	//AnnotationGroup* transformed_anno_group) {
	::google::protobuf::RepeatedPtrField<AnnotationGroup>* transformed_anno_group_all) {
	const int img_height = anno_datum.datum().height();
	const int img_width = anno_datum.datum().width();
	if (anno_datum.type() == AnnotatedDatum_AnnotationType_BBOX) {
		// Go through each AnnotationGroup.
		for (int g = 0; g < anno_datum.annotation_group_size(); ++g) {
			const AnnotationGroup& anno_group = anno_datum.annotation_group(g);
			AnnotationGroup transformed_anno_group;
			// Go through each Annotation.
			bool has_valid_annotation = false;
			for (int a = 0; a < anno_group.annotation_size(); ++a) {
				const Annotation& anno = anno_group.annotation(a);
				const NormalizedBBox& bbox = anno.bbox();
				// Adjust bounding box annotation.
				NormalizedBBox resize_bbox = bbox;
				if (do_resize && param_.has_resize_param()) {
					CHECK_GT(img_height, 0);
					CHECK_GT(img_width, 0);
					UpdateBBoxByResizePolicy(param_.resize_param(), img_width, img_height,
						&resize_bbox);
				}
				//if (param.has_emit_constraint() &&
				//	!MeetEmitConstraint(crop_bbox, resize_bbox,
				//		param.emit_constraint())) {
				//	continue;
				//}
				NormalizedBBox proj_bbox;
				if (ProjectBBox(crop_bbox, resize_bbox, &proj_bbox)) {
					has_valid_annotation = true;
					Annotation* transformed_anno =
						transformed_anno_group.add_annotation();
					transformed_anno->set_instance_id(anno.instance_id());
					NormalizedBBox* transformed_bbox = transformed_anno->mutable_bbox();
					transformed_bbox->CopyFrom(proj_bbox);
					if (do_mirror) {
						float temp = transformed_bbox->xmin();
						transformed_bbox->set_xmin(1 - transformed_bbox->xmax());
						transformed_bbox->set_xmax(1 - temp);
					}
					if (do_resize && param_.has_resize_param()) {
						ExtrapolateBBox(param_.resize_param(), img_height, img_width,
							crop_bbox, transformed_bbox);
					}
				}
			}
			// Save for output.
			if (has_valid_annotation) {
				transformed_anno_group.set_group_label(anno_group.group_label());
				
				transformed_anno_group_all->Add()->CopyFrom(transformed_anno_group);
			}
		}
	}
	else {
		LOG(FATAL) << "Unknown annotation type.";
	}
}
/*
void Augmentation::Transform(const Datum& datum,
	float* transformed_data) {
	NormalizedBBox crop_bbox;
	bool do_mirror;
	Transform(datum, transformed_data, &crop_bbox, &do_mirror);
}

void Augmentation::TransformCrop(const Datum& datum,
	Blob<float>* transformed_blob,
	NormalizedBBox* crop_bbox,
	bool* do_mirror) {
	// If datum is encoded, decoded and transform the cv::image.
	if (datum.encoded()) {
#ifdef USE_OPENCV
		CHECK(!(param_.force_color() && param_.force_gray()))
			<< "cannot set both force_color and force_gray";
		cv::Mat cv_img;
		if (param_.force_color() || param_.force_gray()) {
			// If force_color then decode in color otherwise decode in gray.
			cv_img = DecodeDatumToCVMat(datum, param_.force_color());
		}
		else {
			cv_img = DecodeDatumToCVMatNative(datum);
		}
		// Transform the cv::image into blob.
		return Transform(cv_img, transformed_blob, crop_bbox, do_mirror);
#else
		LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
	}
	else {
		if (param_.force_color() || param_.force_gray()) {
			LOG(ERROR) << "force_color and force_gray only for encoded datum";
		}
	}

	const int crop_size = param_.crop_size();
	const int datum_channels = datum.channels();
	const int datum_height = datum.height();
	const int datum_width = datum.width();

	// Check dimensions.
	const int channels = transformed_blob->channels();
	const int height = transformed_blob->height();
	const int width = transformed_blob->width();
	const int num = transformed_blob->num();

	CHECK_EQ(channels, datum_channels);
	CHECK_LE(height, datum_height);
	CHECK_LE(width, datum_width);
	CHECK_GE(num, 1);

	if (crop_size) {
		CHECK_EQ(crop_size, height);
		CHECK_EQ(crop_size, width);
	}
	else {
		CHECK_EQ(datum_height, height);
		CHECK_EQ(datum_width, width);
	}

	float* transformed_data = transformed_blob->mutable_cpu_data();
	Transform(datum, transformed_data, crop_bbox, do_mirror);
}
*/
void Augmentation::CropImage(const cv::Mat& img,
	const NormalizedBBox& bbox,
	cv::Mat* crop_img) {
	const int img_height = img.rows;
	const int img_width = img.cols;

	// Get the bbox dimension.
	NormalizedBBox clipped_bbox;
	ClipBBox(bbox, &clipped_bbox);
	NormalizedBBox scaled_bbox;
	ScaleBBox(clipped_bbox, img_height, img_width, &scaled_bbox);

	// Crop the image using bbox.
	int w_off = static_cast<int>(scaled_bbox.xmin());
	int h_off = static_cast<int>(scaled_bbox.ymin());
	int width = static_cast<int>(scaled_bbox.xmax() - scaled_bbox.xmin());
	int height = static_cast<int>(scaled_bbox.ymax() - scaled_bbox.ymin());
	cv::Rect bbox_roi(w_off, h_off, width, height);

	img(bbox_roi).copyTo(*crop_img);
}

void Augmentation::CropImage(const Datum& datum,
	const NormalizedBBox& bbox,
	Datum* crop_datum) {
	// If datum is encoded, decode and crop the cv::image.
	if (datum.encoded()) {
#ifdef USE_OPENCV
		CHECK(!(param_.force_color() && param_.force_gray()))
			<< "cannot set both force_color and force_gray";
		cv::Mat cv_img;
		if (param_.force_color() || param_.force_gray()) {
			// If force_color then decode in color otherwise decode in gray.
			cv_img = DecodeDatumToCVMat(datum, param_.force_color());
		}
		else {
			cv_img = DecodeDatumToCVMatNative(datum);
		}
		// Crop the image.
		cv::Mat crop_img;
		CropImage(cv_img, bbox, &crop_img);
		// Save the image into datum.
		EncodeCVMatToDatum(crop_img, "jpg", crop_datum);
		crop_datum->set_label(datum.label());
		return;
#else
		LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
	}
	else {
		if (param_.force_color() || param_.force_gray()) {
			LOG(ERROR) << "force_color and force_gray only for encoded datum";
		}
	}

	const int datum_channels = datum.channels();
	const int datum_height = datum.height();
	const int datum_width = datum.width();

	// Get the bbox dimension.
	NormalizedBBox clipped_bbox;
	ClipBBox(bbox, &clipped_bbox);
	NormalizedBBox scaled_bbox;
	ScaleBBox(clipped_bbox, datum_height, datum_width, &scaled_bbox);
	const int w_off = static_cast<int>(scaled_bbox.xmin());
	const int h_off = static_cast<int>(scaled_bbox.ymin());
	const int width = static_cast<int>(scaled_bbox.xmax() - scaled_bbox.xmin());
	const int height = static_cast<int>(scaled_bbox.ymax() - scaled_bbox.ymin());

	// Crop the image using bbox.
	crop_datum->set_channels(datum_channels);
	crop_datum->set_height(height);
	crop_datum->set_width(width);
	crop_datum->set_label(datum.label());
	crop_datum->clear_data();
	crop_datum->clear_float_data();
	crop_datum->set_encoded(false);
	const int crop_datum_size = datum_channels * height * width;
	const std::string& datum_buffer = datum.data();
	std::string buffer(crop_datum_size, ' ');
	for (int h = h_off; h < h_off + height; ++h) {
		for (int w = w_off; w < w_off + width; ++w) {
			for (int c = 0; c < datum_channels; ++c) {
				int datum_index = (c * datum_height + h) * datum_width + w;
				int crop_datum_index = (c * height + h - h_off) * width + w - w_off;
				buffer[crop_datum_index] = datum_buffer[datum_index];
			}
		}
	}
	crop_datum->set_data(buffer);
}

void Augmentation::CropImage(const AnnotatedDatum& anno_datum,
	const NormalizedBBox& bbox,
	AnnotatedDatum* cropped_anno_datum) {
	// Crop the datum.
	CropImage(anno_datum.datum(), bbox, cropped_anno_datum->mutable_datum());
	cropped_anno_datum->set_type(anno_datum.type());

	// Transform the annotation according to crop_bbox.
	const bool do_resize = false;
	const bool do_mirror = false;
	NormalizedBBox crop_bbox;
	ClipBBox(bbox, &crop_bbox);
	TransformAnnotation(anno_datum, do_resize, crop_bbox, do_mirror,
		cropped_anno_datum->mutable_annotation_group());
}

void Augmentation::TransformCrop(const Datum& datum,
	Blob<float>* transformed_data,
	NormalizedBBox* crop_bbox,
	bool* do_mirror) {
	const string& data = datum.data();
	const int datum_channels = datum.channels();
	const int datum_height = datum.height();
	const int datum_width = datum.width();

	const int crop_size = param_.crop_size();
	const float scale = param_.scale();
	*do_mirror = param_.mirror() && Rand(2);
	const bool has_mean_file = param_.has_mean_file();
	const bool has_uint8 = data.size() > 0;
	const bool has_mean_values = mean_values_.size() > 0;

	CHECK_GT(datum_channels, 0);
	CHECK_GE(datum_height, crop_size);
	CHECK_GE(datum_width, crop_size);

	float* mean = NULL;
	if (has_mean_file) {
		CHECK_EQ(datum_channels, data_mean_.channels());
		CHECK_EQ(datum_height, data_mean_.height());
		CHECK_EQ(datum_width, data_mean_.width());
		mean = data_mean_.mutable_cpu_data();
	}
	if (has_mean_values) {
		CHECK(mean_values_.size() == 1 || mean_values_.size() == datum_channels) <<
			"Specify either 1 mean_value or as many as channels: " << datum_channels;
		if (datum_channels > 1 && mean_values_.size() == 1) {
			// Replicate the mean_value for simplicity
			for (int c = 1; c < datum_channels; ++c) {
				mean_values_.push_back(mean_values_[0]);
			}
		}
	}

	int height = datum_height;
	int width = datum_width;

	int h_off = 0;
	int w_off = 0;
	if (crop_size) {
		height = crop_size;
		width = crop_size;
		// We only do random crop when we do training.
		if (phase_ == TRAIN) {
			h_off = Rand(datum_height - crop_size + 1);
			w_off = Rand(datum_width - crop_size + 1);
		}
		else {
			h_off = (datum_height - crop_size) / 2;
			w_off = (datum_width - crop_size) / 2;
		}
	}

	// Return the normalized crop bbox.
	crop_bbox->set_xmin(float(w_off) / datum_width);
	crop_bbox->set_ymin(float(h_off) / datum_height);
	crop_bbox->set_xmax(float(w_off + width) / datum_width);
	crop_bbox->set_ymax(float(h_off + height) / datum_height);

	float datum_element;
	int top_index, data_index;
/*	for (int c = 0; c < datum_channels; ++c) {
		for (int h = 0; h < height; ++h) {
			for (int w = 0; w < width; ++w) {
				data_index = (c * datum_height + h_off + h) * datum_width + w_off + w;
				if (*do_mirror) {
					top_index = (c * height + h) * width + (width - 1 - w);
				}
				else {
					top_index = (c * height + h) * width + w;
				}
				if (has_uint8) {
					datum_element =
						static_cast<float>(static_cast<uint8_t>(data[data_index]));
				}
				else {
					datum_element = datum.float_data(data_index);
				}
				if (has_mean_file) {
					transformed_data[top_index] =
						(datum_element - mean[data_index]) * scale;
				}
				else {
					if (has_mean_values) {
						transformed_data[top_index] =
							(datum_element - mean_values_[c]) * scale;
					}
					else {
						transformed_data[top_index] = datum_element * scale;
					}
				}
			}
		}
	} */
}

void Augmentation::TransformCrop(
	const AnnotatedDatum& anno_datum, Blob<float>* transformed_blob,
	//AnnotationGroup* transfomed_anno_group,
	::google::protobuf::RepeatedPtrField<AnnotationGroup>* transformed_anno_group_all,
	bool* do_mirror) {
	// Transform datum.
	const Datum& datum = anno_datum.datum();
	NormalizedBBox crop_bbox;
	TransformCrop(datum, transformed_blob, &crop_bbox, do_mirror);

	// Transform annotation.
	const bool do_resize = true;
	TransformAnnotation(anno_datum, do_resize, crop_bbox, *do_mirror,
		//transfomed_anno_group);
		transformed_anno_group_all);
}

void Augmentation::DistortImage(const Datum& datum,
	Datum* distort_datum) {
	if (!param_.has_distort_param()) {
		distort_datum->CopyFrom(datum);
		return;
	}
	// If datum is encoded, decode and crop the cv::image.
	if (datum.encoded()) {
#ifdef USE_OPENCV
		CHECK(!(param_.force_color() && param_.force_gray()))
			<< "cannot set both force_color and force_gray";
		cv::Mat cv_img;
		if (param_.force_color() || param_.force_gray()) {
			// If force_color then decode in color otherwise decode in gray.
			cv_img = DecodeDatumToCVMat(datum, param_.force_color());
		}
		else {
			cv_img = DecodeDatumToCVMatNative(datum);
		}
		// Distort the image.
		cv::Mat distort_img = ApplyDistort(cv_img, param_.distort_param());
		// Save the image into datum.
		caffe::EncodeCVMatToDatum(distort_img, "jpg", distort_datum);
		distort_datum->set_label(datum.label());
		return;
#else
		LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
	}
	else {
		LOG(ERROR) << "Only support encoded datum now";
	}
}

void Augmentation::ExpandImage(const cv::Mat& img,
	const float expand_ratio,
	NormalizedBBox* expand_bbox,
	cv::Mat* expand_img) {
	const int img_height = img.rows;
	const int img_width = img.cols;
	const int img_channels = img.channels();

	// Get the bbox dimension.
	int height = static_cast<int>(img_height * expand_ratio);
	int width = static_cast<int>(img_width * expand_ratio);
	float h_off, w_off;
	caffe_rng_uniform(1, 0.f, static_cast<float>(height - img_height), &h_off);
	caffe_rng_uniform(1, 0.f, static_cast<float>(width - img_width), &w_off);
	h_off = floor(h_off);
	w_off = floor(w_off);
	expand_bbox->set_xmin(-w_off / img_width);
	expand_bbox->set_ymin(-h_off / img_height);
	expand_bbox->set_xmax((width - w_off) / img_width);
	expand_bbox->set_ymax((height - h_off) / img_height);

	expand_img->create(height, width, img.type());
	expand_img->setTo(cv::Scalar(0));
	const bool has_mean_file = param_.has_mean_file();
	const bool has_mean_values = param_.mean_value_size() > 0;

/*	if (has_mean_file) {
		CHECK_EQ(img_channels, data_mean_.channels());
		CHECK_EQ(height, data_mean_.height());
		CHECK_EQ(width, data_mean_.width());
		Dtype* mean = data_mean_.mutable_cpu_data();
		for (int h = 0; h < height; ++h) {
			uchar* ptr = expand_img->ptr<uchar>(h);
			int img_index = 0;
			for (int w = 0; w < width; ++w) {
				for (int c = 0; c < img_channels; ++c) {
					int blob_index = (c * height + h) * width + w;
					ptr[img_index++] = static_cast<char>(mean[blob_index]);
				}
			}
		}
	} */
	if (has_mean_values) {
		vector<float> mean_values;
		CHECK(param_.mean_value_size() == 1 || param_.mean_value_size() == img_channels) <<
			"Specify either 1 mean_value or as many as channels: " << img_channels;
		for (int i = 0; i < param_.mean_value_size(); i++) {
			mean_values.push_back(param_.mean_value(i));
		}
		if (img_channels > 1 && param_.mean_value_size() == 1) {
			// Replicate the mean_value for simplicity
			for (int c = 1; c < img_channels; ++c) {
				mean_values.push_back(param_.mean_value(0));
			}
		}
		vector<cv::Mat> channels(img_channels);
		cv::split(*expand_img, channels);
		CHECK_EQ(channels.size(), param_.mean_value_size());
		for (int c = 0; c < img_channels; ++c) {
			channels[c] = mean_values[c];
		}
		cv::merge(channels, *expand_img);
	}

	cv::Rect bbox_roi(w_off, h_off, img_width, img_height);
	img.copyTo((*expand_img)(bbox_roi));
}

void Augmentation::ExpandImage(const Datum& datum, const float expand_ratio,
	NormalizedBBox* expand_bbox,
	Datum* expand_datum) {
	// If datum is encoded, decode and crop the cv::image.
	if (datum.encoded()) {
#ifdef USE_OPENCV
		CHECK(!(param_.force_color() && param_.force_gray()))
			<< "cannot set both force_color and force_gray";
		cv::Mat cv_img;
		if (param_.force_color() || param_.force_gray()) {
			// If force_color then decode in color otherwise decode in gray.
			cv_img = DecodeDatumToCVMat(datum, param_.force_color());
		}
		else {
			cv_img = DecodeDatumToCVMatNative(datum);
		}
		// Expand the image.
		cv::Mat expand_img;
		ExpandImage(cv_img, expand_ratio, expand_bbox, &expand_img);
		// Save the image into datum.
		EncodeCVMatToDatum(expand_img, "jpg", expand_datum);
		expand_datum->set_label(datum.label());
		return;
#else
		LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
	}
	else {
		if (param_.force_color() || param_.force_gray()) {
			LOG(ERROR) << "force_color and force_gray only for encoded datum";
		}
	}

	const int datum_channels = datum.channels();
	const int datum_height = datum.height();
	const int datum_width = datum.width();

	// Get the bbox dimension.
	int height = static_cast<int>(datum_height * expand_ratio);
	int width = static_cast<int>(datum_width * expand_ratio);
	float h_off, w_off;
	caffe_rng_uniform(1, 0.f, static_cast<float>(height - datum_height), &h_off);
	caffe_rng_uniform(1, 0.f, static_cast<float>(width - datum_width), &w_off);
	h_off = floor(h_off);
	w_off = floor(w_off);
	expand_bbox->set_xmin(-w_off / datum_width);
	expand_bbox->set_ymin(-h_off / datum_height);
	expand_bbox->set_xmax((width - w_off) / datum_width);
	expand_bbox->set_ymax((height - h_off) / datum_height);

	// Crop the image using bbox.
	expand_datum->set_channels(datum_channels);
	expand_datum->set_height(height);
	expand_datum->set_width(width);
	expand_datum->set_label(datum.label());
	expand_datum->clear_data();
	expand_datum->clear_float_data();
	expand_datum->set_encoded(false);
	const int expand_datum_size = datum_channels * height * width;
	const std::string& datum_buffer = datum.data();
	std::string buffer(expand_datum_size, ' ');
	for (int h = h_off; h < h_off + datum_height; ++h) {
		for (int w = w_off; w < w_off + datum_width; ++w) {
			for (int c = 0; c < datum_channels; ++c) {
				int datum_index =
					(c * datum_height + h - h_off) * datum_width + w - w_off;
				int expand_datum_index = (c * height + h) * width + w;
				buffer[expand_datum_index] = datum_buffer[datum_index];
			}
		}
	}
	expand_datum->set_data(buffer);
}

void Augmentation::ExpandImage(const AnnotatedDatum& anno_datum,
	AnnotatedDatum* expanded_anno_datum) {
	if (!param_.has_expand_param()) {
		expanded_anno_datum->CopyFrom(anno_datum);
		return;
	}
	const ExpansionParameter& expand_param = param_.expand_param();
	const float expand_prob = expand_param.prob();
	float prob;
	caffe_rng_uniform(1, 0.f, 1.f, &prob);
	if (prob > expand_prob) {
		expanded_anno_datum->CopyFrom(anno_datum);
		return;
	}
	const float max_expand_ratio = expand_param.max_expand_ratio();
	if (fabs(max_expand_ratio - 1.) < 1e-2) {
		expanded_anno_datum->CopyFrom(anno_datum);
		return;
	}
	float expand_ratio;
	caffe_rng_uniform(1, 1.f, max_expand_ratio, &expand_ratio);
	// Expand the datum.
	NormalizedBBox expand_bbox;
	ExpandImage(anno_datum.datum(), expand_ratio, &expand_bbox,
		expanded_anno_datum->mutable_datum());
	expanded_anno_datum->set_type(anno_datum.type());

	// Transform the annotation according to crop_bbox.
	const bool do_resize = false;
	const bool do_mirror = false;
	AnnotationGroup* expanded_anno_group = new AnnotationGroup();
	TransformAnnotation(anno_datum, do_resize, expand_bbox, do_mirror,
		//expanded_anno_group);
		expanded_anno_datum->mutable_annotation_group());
	//expanded_anno_datum->annotation_group();
}

void DrawBBoxOnImage(AnnotatedDatum datum, string filename) {
	cv::Mat img = DecodeDatumToCVMat(datum.datum(), true);
	for (int g = 0; g < datum.annotation_group_size(); ++g) {
		const AnnotationGroup& anno_group = datum.annotation_group(g);
		NormalizedBBox object_bbox;
		for (int a = 0; a < anno_group.annotation_size(); ++a) {
			const Annotation& anno = anno_group.annotation(a);
			object_bbox = anno.bbox();
			cv::Point pt1, pt2;
			pt1.x = (img.cols*object_bbox.xmin());
			pt1.y = (img.rows*object_bbox.ymin());
			pt2.x = (img.cols*object_bbox.xmax());
			pt2.y = (img.rows*object_bbox.ymax());
			cv::rectangle(img, pt1, pt2, cvScalar(0, 0, 255), 3, 8, 0);
		}
	}
	cv::imwrite(filename, img);
}


DEFINE_int32(min_dim, 0,
	"Minimum dimension images are resized to (keep same aspect ratio)");
DEFINE_int32(max_dim, 0,
	"Maximum dimension images are resized to (keep same aspect ratio)");

void DataAugmentation(cv::Mat img, string img_file) {
	const string& label_map_file = "D:\\Develop\\MobileNet-SSD-windows\\data\\VOC0712\\labelmap_voc.prototxt";
	const string& param_file = "D:\\Develop\\MobileNet-SSD-windows\\models\\VGGNet\\VOC0712\\SSD_300x300\\train.prototxt";
	const string& mean_file = FLAGS_mean_file;
	const string& mean_value = FLAGS_mean_value;


	// Initialize the network.
	Augmentation augment(param_file, mean_file, mean_value);

	// Read a label map
	LabelMap label_map;
	std::map<string, int>* name_to_label = new std::map<string, int>();
	CHECK(ReadProtoFromTextFile(label_map_file, &label_map))
		<< "Failed to read label map file.";
	CHECK(MapNameToLabel(label_map, false, name_to_label))
		<< "Failed to convert name to label.";

	// Set AnnotatedDatum for augmentation
	path img_path(img_file);
	const string labelfile = (img_path.replace_extension(".xml")).string();
	int height = img.rows;
	int width = img.cols;
	AnnotatedDatum anno_datum;
	ReadXMLToAnnotatedDatum(labelfile, height, width, *name_to_label, &anno_datum);
	// Read image to datum.
	int min_dim = std::max<int>(0, FLAGS_min_dim);
	int max_dim = std::max<int>(0, FLAGS_max_dim);
	bool status = ReadImageToDatum(img_file, -1, height, width,
		min_dim, max_dim, true, "jpg", anno_datum.mutable_datum());

	CPUTimer batch_timer;
	batch_timer.Start();
	double read_time = 0;
	double trans_time = 0;
	CPUTimer timer;

	// Store transformed annotation.
	map<int, vector<AnnotationGroup> > all_anno;
	int num_bboxes = 0;
	int item_id = 0;

	/*
	 * Color Distortion and Expand
	 */
	timer.Start();
	read_time += timer.MicroSeconds();
	timer.Start();
	AnnotatedDatum distort_datum;
	//AnnotatedDatum* expand_datum = NULL;
	AnnotatedDatum* expand_datum = new AnnotatedDatum();
	if (augment.param_.has_distort_param()) {
		distort_datum.CopyFrom(anno_datum);
		augment.DistortImage(anno_datum.datum(), distort_datum.mutable_datum());
		DrawBBoxOnImage(distort_datum, "distort.png");
		if (augment.param_.has_expand_param()) {
			augment.ExpandImage(distort_datum, expand_datum);
			DrawBBoxOnImage(*expand_datum, "expand.png");
		}
		else {
			expand_datum = &distort_datum;
		}
	}
	else {
		if (augment.param_.has_expand_param()) {
			augment.ExpandImage(anno_datum, expand_datum);
		}
		else {
			expand_datum = &anno_datum;
		}
	}

	/*
	 * Batch Sampler (Cropping)
	 */
	AnnotatedDatum* sampled_datum = NULL;
	bool has_sampled = false;
	if (augment.batch_samplers_.size() > 0) {
		// Generate sampled bboxes from expand_datum.
		vector<NormalizedBBox> sampled_bboxes;
		GenerateBatchSamples(*expand_datum, augment.batch_samplers_, &sampled_bboxes);
		if (sampled_bboxes.size() > 0) {
			// Randomly pick a sampled bbox and crop the expand_datum.
			int rand_idx = caffe_rng_rand() % sampled_bboxes.size();
			sampled_datum = new AnnotatedDatum();
			augment.CropImage(*expand_datum,
				sampled_bboxes[rand_idx],
				sampled_datum);
			has_sampled = true;
		}
		else {
			sampled_datum = expand_datum;
		}
	}
	else {
		sampled_datum = expand_datum;
	}
	DrawBBoxOnImage(*sampled_datum, "crop.png");
	CHECK(sampled_datum != NULL);


	/*
	* Resize and Noise
	*/
	timer.Start();
	// Apply data transformations (mirror, scale, crop...)
	vector<AnnotationGroup> transformed_anno_vec;
	Blob<float> transformed_data_;
	if (true) {
		AnnotatedDataParameter anno_data_param = augment.layer_param_.annotated_data_param();
		if (anno_data_param.has_anno_type()) {
			// Make sure all data have same annotation type.
			/*CHECK(sampled_datum->has_type()) << "Some datum misses AnnotationType.";
			if (augment.layer_param_. anno_data_param.has_anno_type()) {
				sampled_datum->set_type(anno_type_);
			}
			else {
				CHECK_EQ(anno_type_, sampled_datum->type()) <<
					"Different AnnotationType.";
			} */
			// Transform datum and annotation_group at the same time
			transformed_anno_vec.clear();
			//augment.Transform(*sampled_datum,
			//	&(transformed_data_),
			//	&transformed_anno_vec);
			//if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
				// Count the number of bboxes.
				for (int g = 0; g < transformed_anno_vec.size(); ++g) {
					num_bboxes += transformed_anno_vec[g].annotation_size();
				}
			//}
			//else {
			//	LOG(FATAL) << "Unknown annotation type.";
			//}
			all_anno[item_id] = transformed_anno_vec;
		}
	/*	else {
			Transform(sampled_datum->datum(),
				&(transformed_data_));
			// Otherwise, store the label from datum.
			CHECK(sampled_datum->datum().has_label()) << "Cannot find any label.";
			//top_label[item_id] = sampled_datum->datum().label();
		} */
	}
/*	else {
		Transform(sampled_datum->datum(),
			&(transformed_data_));
	} */

}


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


  // Process image one by one.
  string filename = argv[3];
  std::ifstream infile(filename);
  // http://hwada.hatenablog.com/entry/20110611/1307781684
  path fullpath(filename);
  path rootdir = fullpath.root_directory();
  //const string indir = basename(argv[3]);
  const string indir = rootdir.string();

  std::string img_file;
  string line;
  vector<std::string> fn;
  while (std::getline(infile, line)) {
	  fn.push_back(line);
  }
  //vector<cv::Mat> data;
  for (size_t k = 0; k < fn.size(); ++k)
  {
	  img_file = fn[k];
	  cv::Mat img = cv::imread(img_file);
	  if (img.empty()) continue; //only proceed if sucsessful

	  //DetectObject(img, img_file);
	  DataAugmentation(img, img_file);

	  //data.push_back(img);
  }
  return 0;
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
