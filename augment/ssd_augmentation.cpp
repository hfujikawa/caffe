// This is a demo code for using a SSD model to do detection.
// The code is modified from examples/ssd/ssd_detctsion.cpp.
// Usage:
//    ssd_augmentation [FLAGS] list_file
//
// wherelist_file contains a list of image files with the format as follows:
//    folder/img1.JPEG
//    folder/img2.JPEG
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

// https://stackoverflow.com/questions/8520560/get-a-file-name-from-a-path
using ::boost::filesystem::path;

using namespace caffe;  // NOLINT(build/namespaces)

class Augmentation {
public:
	Augmentation(const string& param_file,
		const string& mean_file,
		const string& mean_value);

	void InitRand();

	void Transform(const Datum& datum,
		Blob<float>* transformed_blob);

	void Transform(const Datum& datum,
		float* transformed_data);
	
	void Transform(const Datum& datum,
		float* transformed_data,
		NormalizedBBox* crop_bbox,
		bool* do_mirror);

	void Transform(const Datum& datum, Blob<float>* transformed_blob,
		NormalizedBBox* crop_bbox, bool* do_mirror);

	void Transform(
		const AnnotatedDatum& anno_datum, Blob<float>* transformed_blob,
		::google::protobuf::RepeatedPtrField<AnnotationGroup>* transformed_anno_group_all,
		bool* do_mirror);

	void Transform(const AnnotatedDatum& anno_datum,
		Blob<float>* transformed_blob,
		vector<AnnotationGroup>* transformed_anno_vec,
		bool* do_mirror);

	void Transform(const AnnotatedDatum& anno_datum,
		Blob<float>* transformed_blob,
		vector<AnnotationGroup>* transformed_anno_vec);

	void TransformAnnotation(
		const AnnotatedDatum& anno_datum, const bool do_resize,
		const NormalizedBBox& crop_bbox, const bool do_mirror,
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
	void Transform(const cv::Mat& cv_img, Blob<float>* transformed_blob,
		NormalizedBBox* crop_bbox, bool* do_mirror);

	void CropImage(const cv::Mat& img, const NormalizedBBox& bbox,
		cv::Mat* crop_img);

	void ExpandImage(const cv::Mat& img, const float expand_ratio,
		NormalizedBBox* expand_bbox, cv::Mat* expand_img);
#endif  // USE_OPENCV

	virtual int Rand(int n);

	LayerParameter layer_param_;
	TransformationParameter param_;
	vector<BatchSampler> batch_samplers_;

	shared_ptr<Caffe::RNG> rng_;
	Phase phase_;
	Blob<float> data_mean_;
	vector<float> mean_values_;
};


DEFINE_string(mean_file, "",
	"The mean file used to subtract from the input image.");
DEFINE_string(mean_value, "104,117,123",
	"If specified, can be one value or can be same as image channels"
	" - would subtract from the corresponding channel). Separated by ','."
	"Either mean_file or mean_value should be provided, not both.");

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

void Augmentation::Transform(const Datum& datum,
	Blob<float>* transformed_blob) {
	NormalizedBBox crop_bbox;
	bool do_mirror;
	Transform(datum, transformed_blob, &crop_bbox, &do_mirror);
}

void Augmentation::Transform(const Datum& datum,
	float* transformed_data) {
	NormalizedBBox crop_bbox;
	bool do_mirror;
	Transform(datum, transformed_data, &crop_bbox, &do_mirror);
}

void Augmentation::Transform(const Datum& datum,
	float* transformed_data,
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
		CHECK(param_.mean_value_size() == 1 || param_.mean_value_size() == datum_channels) <<
			"Specify either 1 mean_value or as many as channels: " << datum_channels;
		if (datum_channels > 1 && param_.mean_value_size() == 1) {
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
	for (int c = 0; c < datum_channels; ++c) {
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
	}
}

void Augmentation::Transform(const Datum& datum,
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


void Augmentation::Transform(
	const AnnotatedDatum& anno_datum, Blob<float>* transformed_blob,
	::google::protobuf::RepeatedPtrField<AnnotationGroup>* transformed_anno_group_all,
	bool* do_mirror) {
	// Transform datum.
	const Datum& datum = anno_datum.datum();
	NormalizedBBox crop_bbox;
	Transform(datum, transformed_blob, &crop_bbox, do_mirror);

	// Transform annotation.
	const bool do_resize = true;
	TransformAnnotation(anno_datum, do_resize, crop_bbox, *do_mirror,
		transformed_anno_group_all);
}

void Augmentation::Transform(
	const AnnotatedDatum& anno_datum, Blob<float>* transformed_blob,
	vector<AnnotationGroup>* transformed_anno_vec, bool* do_mirror) {
	::google::protobuf::RepeatedPtrField<AnnotationGroup> transformed_anno_group_all;
	Transform(anno_datum, transformed_blob, &transformed_anno_group_all, do_mirror);
	for (int g = 0; g < transformed_anno_group_all.size(); ++g) {
		transformed_anno_vec->push_back(transformed_anno_group_all.Get(g));
	}
}

void Augmentation::Transform(
	const AnnotatedDatum& anno_datum, Blob<float>* transformed_blob,
	vector<AnnotationGroup>* transformed_anno_vec) {
	bool do_mirror;
	Transform(anno_datum, transformed_blob, transformed_anno_vec, &do_mirror);
}

void Augmentation::TransformAnnotation(
	const AnnotatedDatum& anno_datum, const bool do_resize,
	const NormalizedBBox& crop_bbox, const bool do_mirror,
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

void Augmentation::Transform(const cv::Mat& cv_img,
	Blob<float>* transformed_blob,
	NormalizedBBox* crop_bbox,
	bool* do_mirror) {
	// Check dimensions.
	const int img_channels = cv_img.channels();
	//const int channels = transformed_blob->channels();
	//const int height = transformed_blob->height();
	//const int width = transformed_blob->width();
	//const int num = transformed_blob->num();
	const int channels = 3;
	const int height = 300;
	const int width = 300;
	const int num = 1;

	CHECK_GT(img_channels, 0);
	CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";
	CHECK_EQ(channels, img_channels);
	CHECK_GE(num, 1);

	const int crop_size = param_.crop_size();
	const float scale = param_.scale();
	*do_mirror = param_.mirror() && Rand(2);
	const bool has_mean_file = param_.has_mean_file();
	const bool has_mean_values = param_.mean_value_size() > 0;

	float* mean = NULL;
	if (has_mean_file) {
		CHECK_EQ(img_channels, data_mean_.channels());
		mean = data_mean_.mutable_cpu_data();
	}
	if (has_mean_values) {
		CHECK(param_.mean_value_size() == 1 || param_.mean_value_size() == img_channels) <<
			"Specify either 1 mean_value or as many as channels: " << img_channels;
		if (img_channels > 1 && param_.mean_value_size() == 1) {
			// Replicate the mean_value for simplicity
			for (int c = 1; c < img_channels; ++c) {
				mean_values_.push_back(param_.mean_value(0));
			}
		}
	}

	int crop_h = param_.crop_h();
	int crop_w = param_.crop_w();
	if (crop_size) {
		crop_h = crop_size;
		crop_w = crop_size;
	}

	cv::Mat cv_resized_image, cv_noised_image, cv_cropped_image;
	if (param_.has_resize_param()) {
		cv_resized_image = ApplyResize(cv_img, param_.resize_param());
	}
	else {
		cv_resized_image = cv_img;
	}
	if (param_.has_noise_param()) {
		cv_noised_image = ApplyNoise(cv_resized_image, param_.noise_param());
	}
	else {
		cv_noised_image = cv_resized_image;
	}
	int img_height = cv_noised_image.rows;
	int img_width = cv_noised_image.cols;
	CHECK_GE(img_height, crop_h);
	CHECK_GE(img_width, crop_w);

	int h_off = 0;
	int w_off = 0;
	if ((crop_h > 0) && (crop_w > 0)) {
		CHECK_EQ(crop_h, height);
		CHECK_EQ(crop_w, width);
		// We only do random crop when we do training.
		if (layer_param_.phase() == TRAIN) {
			h_off = Rand(img_height - crop_h + 1);
			w_off = Rand(img_width - crop_w + 1);
		}
		else {
			h_off = (img_height - crop_h) / 2;
			w_off = (img_width - crop_w) / 2;
		}
		cv::Rect roi(w_off, h_off, crop_w, crop_h);
		cv_cropped_image = cv_noised_image(roi);
	}
	else {
		cv_cropped_image = cv_noised_image;
	}

	// Return the normalized crop bbox.
	crop_bbox->set_xmin(float(w_off) / img_width);
	crop_bbox->set_ymin(float(h_off) / img_height);
	crop_bbox->set_xmax(float(w_off + width) / img_width);
	crop_bbox->set_ymax(float(h_off + height) / img_height);

	if (has_mean_file) {
		CHECK_EQ(cv_cropped_image.rows, data_mean_.height());
		CHECK_EQ(cv_cropped_image.cols, data_mean_.width());
	}
	CHECK(cv_cropped_image.data);
	cv::Mat dst_image;
	if (*do_mirror)
	{
		cv::flip(cv_cropped_image, dst_image, 1);
	}
	else
		dst_image = cv_cropped_image;
	cv::imwrite("resized.png", dst_image);

/*	float* transformed_data = transformed_blob->mutable_cpu_data();
	int top_index;
	for (int h = 0; h < height; ++h) {
		const uchar* ptr = cv_cropped_image.ptr<uchar>(h);
		int img_index = 0;
		int h_idx = h;
		for (int w = 0; w < width; ++w) {
			int w_idx = w;
			if (*do_mirror) {
				w_idx = (width - 1 - w);
			}
			int h_idx_real = h_idx;
			int w_idx_real = w_idx;
			for (int c = 0; c < img_channels; ++c) {
				top_index = (c * height + h_idx_real) * width + w_idx_real;
				float pixel = static_cast<float>(ptr[img_index++]);
				if (has_mean_file) {
					int mean_index = (c * img_height + h_off + h_idx_real) * img_width
						+ w_off + w_idx_real;
					transformed_data[top_index] =
						(pixel - mean[mean_index]) * scale;
				}
				else {
					if (has_mean_values) {
						transformed_data[top_index] =
							(pixel - mean_values_[c]) * scale;
					}
					else {
						transformed_data[top_index] = pixel * scale;
					}
				}
			}
		}
	} */
}

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
		expanded_anno_datum->mutable_annotation_group());
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

void DrawBBoxImage(cv::Mat img, vector<AnnotationGroup> annotation_group, string filename) {
	for (int g = 0; g < annotation_group.size(); ++g) {
		const AnnotationGroup& anno_group = annotation_group[g];
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
	augment.InitRand();

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
	//AnnotatedDatum* sampled_datum = NULL;
	AnnotatedDatum sampled_datum;
	bool has_sampled = false;
	if (augment.batch_samplers_.size() > 0) {
		// Generate sampled bboxes from expand_datum.
		vector<NormalizedBBox> sampled_bboxes;
		GenerateBatchSamples(*expand_datum, augment.batch_samplers_, &sampled_bboxes);
		if (sampled_bboxes.size() > 0) {
			// Randomly pick a sampled bbox and crop the expand_datum.
			int rand_idx = caffe_rng_rand() % sampled_bboxes.size();
			//sampled_datum = new AnnotatedDatum();
			augment.CropImage(*expand_datum,
				sampled_bboxes[rand_idx],
				&sampled_datum);
			has_sampled = true;
		}
		else {
			sampled_datum = *expand_datum;
		}
	}
	else {
		sampled_datum = *expand_datum;
	}
	DrawBBoxOnImage(sampled_datum, "crop.png");
	//CHECK(sampled_datum != NULL);

	/*
	* Resize, Mirror, vertical Flip and Noise
	*/
	timer.Start();
	// Apply data transformations (mirror, scale, crop...)
	vector<AnnotationGroup> transformed_anno_vec;
	Blob<float> transformed_data_;
	if (true) {
		AnnotatedDataParameter anno_data_param = augment.layer_param_.annotated_data_param();
		//if (anno_data_param.has_anno_type()) {
		if (true) {
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
			augment.Transform(sampled_datum,
				&(transformed_data_),
				&transformed_anno_vec);
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
		else {
			augment.Transform(sampled_datum.datum(),
				&(transformed_data_));
			// Otherwise, store the label from datum.
			CHECK(sampled_datum.datum().has_label()) << "Cannot find any label.";
			//top_label[item_id] = sampled_datum->datum().label();
		}
	}
	else {
		augment.Transform(sampled_datum.datum(),
			&(transformed_data_));
	}
	cv::Mat result_image = cv::imread("resized.png");
	DrawBBoxImage(result_image, transformed_anno_vec, "resized.png");
	for (int i = 0; i < transformed_anno_vec.size(); i++) {
		for (int j = 0; j < transformed_anno_vec[i].annotation_size() ; j++) {
			;
		}
	}
	
}

