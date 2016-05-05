// Sponge Detection by Richard chan
// 2016 spring - University of Southern California
// Machine learning on medical application research
// =======================================================
// The program will read ground truth images from 'groundtruth' foldera and train 
// SVM with normalized images in 'normalized' folder and use images inside 'img_test' to verify
// the model's accuracy
// Compiled with Microsoft Visual Studio Express 2013 with OpenCV library
// =========================================================
#include "stdafx.h"
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\photo\photo.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\nonfree\features2d.hpp>
#include <opencv2\flann\flann.hpp>
#include <opencv2\ml\ml.hpp>
#include <opencv2\opencv_modules.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <iostream>
#include <ctime>
#include <Windows.h>

using namespace cv;
using namespace std;


// Algorithms
SurfFeatureDetector detector(200);//300
SurfDescriptorExtractor extractor;

//SiftFeatureDetector detector;
//SiftDescriptorExtractor extractor;

// Parameters
string contour_tag;
string detector_tag; 

int MAX_KERNEL_LENGTH = 8; //7
float contrast_alpha = 1; //0.9
int contrast_beta = 0; //0
int drawing_counter_max = 5; // Count how many images can be drew to decrese speed.
int drawing_counter = 0; // Count how many images can be drew to decrese speed.
float matcher_min_distance_threshold = 0.12; // 0.06  //0.15 TOO HIGH //0.11 for canny surf // 50,100 for sift
int canny_thresh = 50; // 80 ok

void setTag(string *contour, string *detector){
	*contour = "Canny";
	*detector = "SURF";
}

vector<string> get_all_files_names_within_folder(string folder)
{
	vector<string> names;
	char search_path[200];
	std::sprintf(search_path, "%s/*.jpg", folder.c_str());
	WIN32_FIND_DATA fd;
	HANDLE hFind = ::FindFirstFile(search_path, &fd);
	if (hFind != INVALID_HANDLE_VALUE) {
		do {
			// read all (real) files in current folder
			// , delete '!' read other 2 default folder . and ..
			if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
				names.push_back(fd.cFileName);
			}
		} while (::FindNextFile(hFind, &fd));
		::FindClose(hFind);
	}
	return names;
}

void contour_processing(Mat *src){
	src->convertTo(*src, -1, contrast_alpha, contrast_beta); // Adjust the contrast and brightness
	//Laplacian
	//Laplacian(*src, *src, CV_16S, 3);
	//convertScaleAbs(*src, *src);
	//blur(*src, *src, Size(3, 3));
	for (int j = 1; j < MAX_KERNEL_LENGTH; j = j + 2)
	{
		GaussianBlur(*src, *src, Size(j, j), 0, 0);
	}
	Canny(*src, *src, canny_thresh, canny_thresh * 2, 3);
}


void detect_extract_feature(Mat *src, vector<KeyPoint> *kp, Mat *descriptorOut, string file_name, int *counter, int *max){
	// Detect feature
	std::cout << "Detecting Feature..." << endl;
	detector.detect(*src, *kp);
	// Extract feature
	std::cout << kp->size() << " feature detected " << endl;
	std::cout << "Extracting Descriptor..." ;
	extractor.compute(*src, *kp, *descriptorOut);
	std::cout << "Done" << endl;
	if (*counter < *max){
		// Print out answer images
		Mat keypoint_image;
		drawKeypoints(*src, *kp, keypoint_image, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		std::cout << "Printing keypoint image" << endl;
		imwrite(file_name, keypoint_image);
	}
}

void DeleteAllFiles(char* folderPath)
{
	char fileFound[256];
	WIN32_FIND_DATA info;
	HANDLE hp;
	sprintf(fileFound, "%s\\*.*", folderPath);
	hp = FindFirstFile(fileFound, &info);
	do
	{
		sprintf(fileFound, "%s\\%s", folderPath, info.cFileName);
		DeleteFile(fileFound);

	} while (FindNextFile(hp, &info));
	FindClose(hp);
}


int _tmain(int argc, _TCHAR* argv[])
{
	if (argc == 1){
		std::cout << "Train from normalize folder" << endl;
	}
	else if (argc == 3){
		std::cout << "Training with\n\t" << argv[1] << "\n\t" << argv[2] << endl;
	}
	else {
		std::cout << "Parameter error!\n   Usage: ./EXE CLASSIFIER[s|b] TRAINING_DATA TRAINING_LABEL" << endl;
	}
	setTag(&contour_tag, &detector_tag);
	clock_t begin = clock();
	// Locate image file
	Mat original_image;
	Mat answer_image;
	// Create output directory
	CreateDirectory("output", NULL);
	DeleteAllFiles("output");
	CreateDirectory("matched", NULL);
	DeleteAllFiles("matched");
	CreateDirectory("tested", NULL);
	DeleteAllFiles("tested");
	CreateDirectory("normalized", NULL);
	CreateDirectory("answer", NULL);
	DeleteAllFiles("answer");
	CreateDirectory("img_test", NULL);


	string image_reading_folder = "./normalized/";
	string groundtruth_folder = "./groundtruth/";
	string image_testing_folder = "./img_test/";
	// Traning data
	Mat training_data;
	Mat training_labels;
	int number_of_row_training_data = 0;

	// Answer data
	vector<KeyPoint> keypoints_answer;
	Mat answer_descriptor;
	char ans_en = 1; // Skip answer switch
	if (ans_en == 1){
		vector<string> all_answer = get_all_files_names_within_folder(groundtruth_folder);
		for (auto i = all_answer.begin(); i != all_answer.end(); i++){
			string input_image_name = groundtruth_folder + *i;
			string current_file_name = *i;
			// Feature detection - true answer
			std::cout << "Loading GroundTruth..." << current_file_name << endl;
			answer_image = imread(input_image_name, 0);	//Load the image
			if (!answer_image.data) // Check for invalid input
			{
				std::cout << "Could not open or find the image:" << std::endl;
				return -1;
			}

			// Pre-processing ground truth
			// equalizeHist(answer_image, answer_image);
			// answer_image.convertTo(answer_image, -1, contrast_alpha, contrast_beta); // Adjust the contrast and brightness

			// Perform contour processing
			contour_processing(&answer_image);
			string original_name = "./answer/original_"  + *i;
			imwrite(original_name, answer_image);
			// Detect feature
			std::cout << "Detecting Feature..." << endl;
			detector.detect(answer_image, keypoints_answer);
			// Extract feature
			std::cout << keypoints_answer.size() << "feature detected " << endl;
			std::cout << "Extracting Descriptor...";
			Mat answer_descriptor_inner;
			extractor.compute(answer_image, keypoints_answer, answer_descriptor_inner);
			cout << "Size of descriptor: " << answer_descriptor_inner.size() << endl;
			answer_descriptor.push_back(answer_descriptor_inner);
			std::cout << "Done" << endl;
			if (drawing_counter < drawing_counter_max){
				// Print out answer images
				Mat keypoint_image;
				drawKeypoints(answer_image, keypoints_answer, keypoint_image, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
				std::cout << "Printing keypoint image" << endl;
				string answer_name = "./answer/keypoint_" + contour_tag + "_" + *i;
				imwrite(answer_name, keypoint_image);
			}
		}
	}
	cout << "Size of the real answer_descriptor: " << answer_descriptor.size() << endl;
	if (argc == 1){ // When training data sets are provided, skip this
		// Start Image Pre-processing for each image in 'image_reading_folder'
		vector<string> all_images = get_all_files_names_within_folder(image_reading_folder);
		for (auto i = all_images.begin(); i != all_images.end(); i++){
			// Iterate through images
			string input_image_name = image_reading_folder + *i;
			string current_file_name = *i;

			// Assume all images' histogram are normalized
			original_image = imread(input_image_name, 0); // Load image
			original_image.convertTo(original_image, -1, contrast_alpha, contrast_beta); // Adjust the contrast and brightness
			std::cout << "============= Input_image_name:" << input_image_name << "==========" << endl;

			// Training data processing
			number_of_row_training_data++;
			if (current_file_name[0] == 'x' || current_file_name[0] == 'X'){
				std::cout << "Current file is false case" << endl;
				training_labels.push_back(0);
			}
			else {
				std::cout << "Current file is true case" << endl;
				training_labels.push_back(1);
			}

			// Image-preprocess
			contour_processing(&original_image);

			// Draw Image
			if (drawing_counter < drawing_counter_max){
				string my_image_name = "./output/" + contour_tag + "_" + *i;
				imwrite(my_image_name, original_image);
			}

			// Feature detection and extraction
			vector<KeyPoint> keypoints_input;
			Mat current_image_descriptor;
			string keypoints_adjust_name = "./output/keypoint" + *i;

			detect_extract_feature(&original_image, &keypoints_input, &current_image_descriptor, keypoints_adjust_name, &drawing_counter, &drawing_counter_max);


			FlannBasedMatcher matcher;
			vector<DMatch> matches;
			std::cout << "Start matching...";
			matcher.match(answer_descriptor, current_image_descriptor, matches);
			std::cout << "Done" << endl;
			std::cout << "Answer_descriptor.row:" << answer_descriptor.rows << " Current_descriptor.row:" << current_image_descriptor.rows << endl;
			std::cout << "Number of matches from descriptor:" << matches.size() << endl;

			double max_dist = 0; double min_dist = 1000;
			for (int j = 0; j < answer_descriptor.rows; j++)
			{
				double dist = matches[j].distance;
				if (dist < min_dist) min_dist = dist;
				if (dist > max_dist) max_dist = dist;
			}
			printf("-- Max dist : %f \t", max_dist);
			printf("-- Min dist : %f \n", min_dist);

			vector<DMatch> good_matches;
			for (int j = 0; j < answer_descriptor.rows; j++)
			{
				if (matches[j].distance <= max(min_dist * 2, matcher_min_distance_threshold)) // Normal SIFT

				{
					good_matches.push_back(matches[j]);
					training_data.push_back(1);
				}
				else {
					training_data.push_back(0);
				}
			}
			std::cout << "Matched:" << good_matches.size() << "/" << matches.size() << endl;
			std::cout << "Training data size:[col , row]" << training_data.size() << endl;

			//-- Draw only "good" matches

			if (drawing_counter < 0){
					Mat img_matches;
				drawMatches(answer_image, keypoints_answer, original_image, keypoints_input,
					good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
					vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
				string matched_image_name = "./matched/" + *i;
				imwrite(matched_image_name, img_matches);
				std::cout << "Wrote to: ./matched/" << *i << endl;
			}
			drawing_counter++;
		} // Done with image pre-processing
	}
	if (argc > 1){
		// Read directly from training file previously stored
		std::cout << "Reading... " << argv[1] << endl;
		FileStorage yml_data(argv[1], FileStorage::READ);
		FileStorage yml_label(argv[2], FileStorage::READ);
		yml_data["training_data"] >> training_data;
		yml_label["training_label"] >> training_labels;
		yml_data.release();
		yml_label.release();
	}
	else if (argc == 1){
		// Reshape data 
		training_data = training_data.reshape(1, number_of_row_training_data);
	}
	std::cout << "Training data size:[col , row]" << training_data.size() << endl;
	std::cout << "Training label size:[col , row]" << training_labels.size() << endl;

	// Setup parameter for SVM
	std::cout << "Setting training parameters" << endl;
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

	// Convert to floating point
	training_data.convertTo(training_data, CV_32F); 
	training_labels.convertTo(training_labels, CV_32F);

	CvSVM SVM;
	// Train SVM
	std::cout << "Start Training SVM" << endl;
	SVM.train(training_data, training_labels, Mat(), Mat(), params);
	std::cout << "Finished training" << endl;

	// Get all test image file names
	Mat testing_image;
	// Counter for accuracy test
	float count_true_test_image = 0;
	float count_false_test_image = 0;

	// Every counter reserve one spot for future use
	float true_positive[2] = { 0 }; 
	float true_negative[2] = { 0 };
	float false_negative[2] = { 0 };
	float false_positive[2] = { 0 };
	// Reset drawing counter to draw some tested images
	drawing_counter = 0;

	// ======================================================   Perform testing ==========================================================
	vector<string> final_result;
	vector<string> all_testing_images = get_all_files_names_within_folder(image_testing_folder);
	for (auto i = all_testing_images.begin(); i != all_testing_images.end(); i++){
		// Read image
		string input_image_name = image_testing_folder + *i;
		std::cout << "============= Testing image:" << input_image_name << "==========" << endl;
		testing_image = imread(input_image_name, 0); // Load image

		// Count images
		bool true_answer;
		string filename = *i;
		if (filename[0] == 'x' || filename[0] == 'X'){
			// False Case
			count_false_test_image++;
			true_answer = 0;
		}
		else {
			count_true_test_image++;
			true_answer = 1;
		}
		std::cout << "Current sequence: " << (count_false_test_image + count_true_test_image) << endl;

		// Pre-process image
		equalizeHist(testing_image, testing_image);

		contour_processing(&testing_image);

		if (drawing_counter < drawing_counter_max){
			// Draw Image
			string outputimage_name = "./tested/" + *i;
			imwrite(outputimage_name, testing_image);
		}
		// Feature detection & extraction
		Mat current_image_descriptor;
		vector<KeyPoint> keypoints_adjust;
		string output_keypoint = "./tested/key_" + *i;

		detect_extract_feature(&testing_image, &keypoints_adjust, &current_image_descriptor, output_keypoint, &drawing_counter, &drawing_counter_max);


		FlannBasedMatcher matcher;
		vector<DMatch> matches;
		matcher.match(answer_descriptor, current_image_descriptor, matches);
		std::cout << "Answer_descriptor.row:" << answer_descriptor.rows << "Current_descriptor.row:" << current_image_descriptor.rows << endl;


		double max_dist = 0; double min_dist = 1000;
		for (int j = 0; j < answer_descriptor.rows; j++)
		{
			double dist = matches[j].distance;
			if (dist < min_dist) min_dist = dist;
			if (dist > max_dist) max_dist = dist;
		}
		printf("-- Max dist : %f \t", max_dist);
		printf("-- Min dist : %f \n", min_dist);

		vector<DMatch> good_matches;
		Mat testing_data;
		for (int j = 0; j < answer_descriptor.rows; j++)
		{
			if (matches[j].distance <= max(min_dist * 2, matcher_min_distance_threshold))
			{
				good_matches.push_back(matches[j]);
				testing_data.push_back((float)1);
			}
			else {
				testing_data.push_back((float)0);
			}
		}
		std::cout << "Matched:" << good_matches.size() << "/" << matches.size() << endl;
		std::cout << "Training data size:[col , row]" << testing_data.size() << endl;


		//-- Draw only "good" matches

		//if (drawing_counter < drawing_counter_max){
		if (drawing_counter < 0){
				Mat img_matches;
			drawMatches(answer_image, keypoints_answer, testing_image, keypoints_adjust,
				good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
				vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
			string matched_image_name = "./tested/match__" + *i;
			imwrite(matched_image_name, img_matches);
			std::cout << "Wrote to: ./tested/match__" << *i << endl;
		}

		testing_data.convertTo(testing_data, CV_32F);
		float answer[2];

		answer[0] = SVM.predict(testing_data);
		cout << "Done predicting" << endl;
		for (int z = 0; z < 1; z++){
			// Count predicted label
			if (answer[z] == 0 && true_answer == 0){
				// Correctly predicted false
				true_negative[z]++;
			}
			else if (answer[z] == 1 && true_answer == 1)  {
				// Predicted True
				true_positive[z]++;
			}
			else if (answer[z] == 1 && true_answer == 0)  {
				// Predicted True
				false_positive[z]++;
			}
			else if (answer[z] == 0 && true_answer == 1)  {
				// Predicted True
				false_negative[z]++;
			}
		}
		string result = "File: " + *i + "= " + to_string(answer[0]);
		std::cout << result << endl;
		final_result.push_back(result);
		drawing_counter++;
	} // End of testing

	std::cout << endl << endl;
	for (vector<string>::const_iterator i = final_result.begin(); i != final_result.end(); ++i) {
		// process i
		std::cout << *i << endl; // this will print all the contents of final_result
	}

	float sensitivity[2], specificity[2], accuracy[2];

	std::cout << endl << "=========================" << endl;
	printf("Alpha:%.2f\tBeta:%d\tBlur Strength:%d\tMatcher Threshold:%.3f\n", contrast_alpha, contrast_beta, MAX_KERNEL_LENGTH, matcher_min_distance_threshold);
	for (int z = 0; z < 1; z++){ // Currently only one detector, loop execute once
		// Print out Sensitivity (Recall), Specificity
		sensitivity[z] = true_positive[z] / (true_positive[z] + false_negative[z]); // recall
		specificity[z] = true_negative[z] / (false_positive[z] + true_negative[z]);
		accuracy[z] = (true_positive[z] + true_negative[z]) / (count_true_test_image + count_false_test_image);
		string classifier_name = (z == 0) ? "SVM" : "Knn";
		cout << classifier_name << endl;
		printf("True_positive:%.1f\tTrue_negative:%.1f\tFalse_positive:%.1f\tFalse_negative:%.1f\n", true_positive[z], true_negative[z], false_positive[z], false_negative[z]);
		printf("Sensitivity: %.2f\tSpecificity:%.2f\tAccuray:%.2f\n", sensitivity[z], specificity[z], accuracy[z]);
	}
	
	printf("Total Images tested:%.1f\t True cases/False cases:[%.1f/%.1f]", count_true_test_image + count_false_test_image, count_true_test_image, count_false_test_image);
	std::cout << endl << "=========================" << endl;

	// Save the training data to file
	char matrix_data_name[254];
	char matrix_label_name[254];
	sprintf(matrix_data_name, "./A%.2f_CA%.2f_CB%d_BLUR%d_training_data.yml" , accuracy[0], contrast_alpha, contrast_beta, MAX_KERNEL_LENGTH);
	sprintf(matrix_label_name, "./A%.2f_CA%.2f_CB%d_BLUR%d_training_label.yml", accuracy[0], contrast_alpha, contrast_beta, MAX_KERNEL_LENGTH);
	FileStorage output_data(matrix_data_name, FileStorage::WRITE);
	FileStorage output_label(matrix_label_name, FileStorage::WRITE);
	output_data << "training_data" << training_data;
	output_label << "training_label" << training_labels;
	cout << "Image Process: " << contour_tag << "\t Extractor: " << detector_tag << endl;
	cout << "Filed Save : " << matrix_data_name << endl;
	output_data.release();
	output_label.release();
	std::cout << "Done" << endl;
	clock_t end = clock();
	double elapsed_time = double(end - begin) / CLOCKS_PER_SEC;
	std::cout << "Runtime: " << floor(elapsed_time / 60) << " Mins " << int(elapsed_time) % 60 << " Seconds" << endl;
	return 0;
}


