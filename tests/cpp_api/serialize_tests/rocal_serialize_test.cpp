/*
MIT License

Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "opencv2/opencv.hpp"
#include "rocal_api.h"
using namespace cv;

#if USE_OPENCV_4
#define CV_LOAD_IMAGE_COLOR IMREAD_COLOR
#define CV_BGR2GRAY COLOR_BGR2GRAY
#define CV_GRAY2RGB COLOR_GRAY2RGB
#define CV_RGB2BGR COLOR_RGB2BGR
#define CV_FONT_HERSHEY_SIMPLEX FONT_HERSHEY_SIMPLEX
#define CV_FILLED FILLED
#define CV_WINDOW_AUTOSIZE WINDOW_AUTOSIZE
#endif

#define DISPLAY 0

using namespace std::chrono;

int test(int test_case, int reader_type, const char *path, const char *outName, int rgb, int gpu, int width, int height, int num_of_classes, int display_all);
int main(int argc, const char **argv) {
    // check command-line usage
    const int MIN_ARG_COUNT = 2;
    if (argc < MIN_ARG_COUNT) {
        printf("Usage: unit_tests reader-type <image-dataset-folder> output_image_name <width> <height> test_case gpu=1/cpu=0 rgb=1/grayscale=0 one_hot_labels=num_of_classes/0  display_all=0(display_last_only)1(display_all)\n");
        return -1;
    }

    int argIdx = 0;
    int reader_type = atoi(argv[++argIdx]);
    const char *path = argv[++argIdx];
    const char *outName = argv[++argIdx];
    int width = atoi(argv[++argIdx]);
    int height = atoi(argv[++argIdx]);
    int display_all = 0;

    int rgb = 1;  // process color images
    bool gpu = 1;
    int test_case = 3;  // For Rotate
    int num_of_classes = 0;

    if (argc >= argIdx + MIN_ARG_COUNT)
        test_case = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        gpu = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        rgb = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        num_of_classes = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        display_all = atoi(argv[++argIdx]);

    test(test_case, reader_type, path, outName, rgb, gpu, width, height, num_of_classes, display_all);

    return 0;
}

int test(int test_case, int reader_type, const char *path, const char *outName, int rgb, int gpu, int width, int height, int num_of_classes, int display_all) {
    size_t num_threads = 1;
    const unsigned int input_batch_size = 2;
    int decode_max_width = width;
    int decode_max_height = height;
    int pipeline_type = -1;
    std::cout << "Test case " << test_case << std::endl;
    std::cout << "Running on " << (gpu ? "GPU" : "CPU") << " , " << (rgb ? " Color " : " Grayscale ") << std::endl;

    RocalImageColor color_format = (rgb != 0) ? RocalImageColor::ROCAL_COLOR_RGB24
                                              : RocalImageColor::ROCAL_COLOR_U8;

    auto handle = rocalCreate(input_batch_size,
                              gpu ? RocalProcessMode::ROCAL_PROCESS_GPU : RocalProcessMode::ROCAL_PROCESS_CPU, 0,
                              1);

    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "Could not create the Rocal contex\n";
        return -1;
    }

    /*>>>>>>>>>>>>>>>> Getting the path for MIVisionX-data  <<<<<<<<<<<<<<<<*/

    std::string rocal_data_path;
    if (std::getenv("ROCAL_DATA_PATH"))
        rocal_data_path = std::getenv("ROCAL_DATA_PATH");

    /*>>>>>>>>>>>>>>>> Creating Rocal parameters  <<<<<<<<<<<<<<<<*/

    rocalSetSeed(0);

    // // Creating uniformly distributed random objects to override some of the default augmentation parameters
    RocalIntParam color_temp_adj = rocalCreateIntParameter(-50);
    RocalIntParam mirror = rocalCreateIntParameter(1);

    /*>>>>>>>>>>>>>>>>>>> Graph description <<<<<<<<<<<<<<<<<<<*/

    RocalTensor decoded_output;
    RocalTensorLayout output_tensor_layout = (rgb != 0) ? RocalTensorLayout::ROCAL_NHWC : RocalTensorLayout::ROCAL_NCHW;
    RocalTensorOutputType output_tensor_dtype = RocalTensorOutputType::ROCAL_UINT8;
    // The jpeg file loader can automatically select the best size to decode all images to that size
    // User can alternatively set the size or change the policy that is used to automatically find the size
    rocalCreateLabelReader(handle, path);
    if (decode_max_height <= 0 || decode_max_width <= 0)
        decoded_output = rocalJpegFileSource(handle, path, color_format, num_threads, false, true);
    else
        decoded_output = rocalJpegFileSource(handle, path, color_format, num_threads, false, false, false, ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED, decode_max_width, decode_max_height);

    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "JPEG source could not initialize : " << rocalGetErrorMessage(handle) << std::endl;
        return -1;
    }

    int resize_w = width, resize_h = height;  // height and width

    RocalTensor input = decoded_output;
    // RocalTensor input = rocalResize(handle, decoded_output, resize_w, resize_h, false); // uncomment when processing images of different size
    // RocalTensor output = rocalBrightness(handle, input, true);
    RocalTensor output = rocalBrightnessFixed(handle, input, 0.5, 0.5, true);


    // Calling the API to verify and build the augmentation graph
    rocalVerify(handle);
    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "Could not verify the augmentation graph " << rocalGetErrorMessage(handle);
        return -1;
    }

    auto number_of_outputs = rocalGetAugmentationBranchCount(handle);
    std::cout << "\n\nAugmented copies count " << number_of_outputs << "\n";

    if (number_of_outputs != 1) {
        std::cout << "More than 1 output set in the pipeline";
        return -1;
    }

    /*>>>>>>>>>>>>>> Serialize the pipeline <<<<<<<<<<<<<<<<<<*/
    size_t str_size;
    rocalSerialize(handle, str_size);
    std::cerr << "String size : ------------------------------>>>>>>>>>>>>> " << str_size << "\n";
    std::string serialized_pipe_string(str_size, '\0');
    rocalGetSerializedString(handle, serialized_pipe_string.c_str());
    std::cerr << "==================================================================\n";
    std::cerr << serialized_pipe_string << "\n";
    std::cerr << "==================================================================\n";

    // perform deserialize
    auto pipe_params = RocalPipelineParams();

    auto second_handle = rocalDeserialize(serialized_pipe_string.c_str(), str_size, pipe_params);
    rocalSetSeed(pipe_params.seed.value());
    
    // Calling the API to verify and build the augmentation graph
    rocalVerify(second_handle);
    if (rocalGetStatus(second_handle) != ROCAL_OK) {
        std::cout << "Could not verify the augmentation graph " << rocalGetErrorMessage(handle);
        return -1;
    }

    auto number_of_output = rocalGetAugmentationBranchCount(second_handle);
    std::cout << "\n\nAugmented copies count " << number_of_outputs << "\n";

    if (number_of_output != 1) {
        std::cout << "More than 1 output set in the pipeline";
        return -1;
    }

    /*>>>>>>>>>>>>>>>>>>> Diplay using OpenCV <<<<<<<<<<<<<<<<<*/
    int h = rocalGetAugmentationBranchCount(second_handle) * rocalGetOutputHeight(second_handle) * input_batch_size;
    int w = rocalGetOutputWidth(second_handle);
    int p = ((color_format == RocalImageColor::ROCAL_COLOR_RGB24) ? 3 : 1);
    const unsigned number_of_cols = 1;  // 1920 / w;
    auto cv_color_format = ((color_format == RocalImageColor::ROCAL_COLOR_RGB24) ? CV_8UC3 : CV_8UC1);
    cv::Mat mat_output(h, w, cv_color_format);
    cv::Mat mat_input(h, w, cv_color_format);
    cv::Mat mat_color;
    int col_counter = 0;
    if (DISPLAY)
        cv::namedWindow("output", CV_WINDOW_AUTOSIZE);
    printf("Remaining images %lu \n", rocalGetRemainingImages(second_handle));
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    int index = 0;

// >>>>>>>>>>>>>>>>>>> Fist handle case

    while (rocalGetRemainingImages(handle) >= input_batch_size) {
        index++;
        if (rocalRun(handle) != 0)
            break;
        int image_name_length[input_batch_size];
        /*switch (pipeline_type) {
            case 1: {   // classification pipeline
                RocalTensorList labels = rocalGetImageLabels(handle);
                int *label_id = reinterpret_cast<int *>(labels->at(0)->buffer());  // The labels are present contiguously in memory
                int img_size = rocalGetImageNameLen(handle, image_name_length);
                std::vector<char> img_name(img_size);
                std::vector<int> label_one_hot_encoded(input_batch_size * num_of_classes);
                rocalGetImageName(handle, img_name.data());
                if (num_of_classes != 0) {
                    rocalGetOneHotImageLabels(handle, label_one_hot_encoded.data(), num_of_classes, RocalOutputMemType::ROCAL_MEMCPY_HOST);
                }
                std::cerr << "\nImage name:" << img_name.data() << "\n";
                for (unsigned int i = 0; i < input_batch_size; i++) {
                    std::cerr << "Label id: " << label_id[i] << std::endl;
                    if(num_of_classes != 0)
                    {
                        std::cout << "One Hot Encoded labels:"<<"\t";
                        for (int j = 0; j < num_of_classes; j++)
                        {
                            int idx_value = label_one_hot_encoded[(i*num_of_classes)+j];
                            if(idx_value == 0)
                                std::cout << idx_value << "\t";
                            else
                            {
                                std::cout << idx_value << "\t";
                            }
                        }
                    }
                    std::cout << "\n";
                }
            } break;
            case 2: {   // detection pipeline
                int img_size = rocalGetImageNameLen(handle, image_name_length);
                std::vector<char> img_name(img_size);
                rocalGetImageName(handle, img_name.data());
                std::cerr << "\nImage name:" << img_name.data();
                RocalTensorList bbox_labels = rocalGetBoundingBoxLabel(handle);
                RocalTensorList bbox_coords = rocalGetBoundingBoxCords(handle);
                for (unsigned i = 0; i < bbox_labels->size(); i++) {
                    int *labels_buffer = reinterpret_cast<int *>(bbox_labels->at(i)->buffer());
                    float *bbox_buffer = reinterpret_cast<float *>(bbox_coords->at(i)->buffer());
                    std::cerr << "\nBBOX Labels : ";
                    for (unsigned j = 0; j < bbox_labels->at(i)->dims().at(0); j++)
                        std::cerr << labels_buffer[j] << " ";
                    std::cerr << "\nBBOX Count: " << bbox_coords->at(i)->dims().at(0) << "\n";
                    for (unsigned j = 0, j4 = 0; j < bbox_coords->at(i)->dims().at(0); j++, j4 = j * 4)
                        std::cerr << bbox_buffer[j4] << " " << bbox_buffer[j4 + 1] << " " << bbox_buffer[j4 + 2] << " " << bbox_buffer[j4 + 3] << "\n";
                }
                int img_sizes_batch[input_batch_size * 2];
                rocalGetImageSizes(handle, img_sizes_batch);
                for (int i = 0; i < (int)input_batch_size; i++) {
                    std::cout << "\nwidth:" << img_sizes_batch[i * 2];
                    std::cout << "\nHeight:" << img_sizes_batch[(i * 2) + 1];
                }
            } break;
            case 3: {   // keypoints pipeline
                int size = input_batch_size;
                RocalJointsData *joints_data;
                rocalGetJointsDataPtr(handle, &joints_data);
                for (int i = 0; i < size; i++) {
                    std::cout << "ImageID: " << joints_data->image_id_batch[i] << std::endl;
                    std::cout << "AnnotationID: " << joints_data->annotation_id_batch[i] << std::endl;
                    std::cout << "ImagePath: " << joints_data->image_path_batch[i] << std::endl;
                    std::cout << "Center: " << joints_data->center_batch[i][0] << " " << joints_data->center_batch[i][1] << std::endl;
                    std::cout << "Scale: " << joints_data->scale_batch[i][0] << " " << joints_data->scale_batch[i][1] << std::endl;
                    std::cout << "Score: " << joints_data->score_batch[i] << std::endl;
                    std::cout << "Rotation: " << joints_data->rotation_batch[i] << std::endl;

                    for (int k = 0; k < 17; k++) {
                        std::cout << "x : " << joints_data->joints_batch[i][k][0] << " , y : " << joints_data->joints_batch[i][k][1] << " , v : " << joints_data->joints_visibility_batch[i][k][0] << std::endl;
                    }
                }
            } break;
            case 4: {   // webdataset pipeline
                int img_size = rocalGetImageNameLen(handle, image_name_length);
                std::vector<char> img_name(img_size);
                rocalGetImageName(handle, img_name.data());
                std::cout << "\n Image name: " << img_name.data() << "\n \n";
                RocalMetaData ascii_sample_contents = rocalGetAsciiDatas(handle);
                std::vector<std::vector<std::vector<uint8_t>>> ext_componenet_list;
                for(uint ext = 0; ext < ascii_sample_contents->size(); ext++) {
                    RocalTensorList ext_ascii_values_batch = ascii_sample_contents->at(ext);
                    std::vector<std::vector<uint8_t>> component_list;
                    std::vector<uint8_t> ascii_components_array;
                    for (uint i = 0; i < ext_ascii_values_batch->size(); i++) {
                        if (ext_ascii_values_batch->at(i)->buffer() !=  nullptr) {
                            uint8_t* buffer = reinterpret_cast<uint8_t*>(ext_ascii_values_batch->at(i)->buffer());
                            size_t length = ext_ascii_values_batch->at(i)->dims().at(0);
                            ascii_components_array.assign(buffer, buffer + length);
                        } else {
                            ascii_components_array = std::vector<uint8_t>{};
                        }
                        component_list.push_back(ascii_components_array);
                    }
                    ext_componenet_list.push_back(component_list);
                }
                for (size_t i = 0; i < ext_componenet_list.size(); ++i) {
                    std::cout << " Meta Data Component " << i + 1 << ":" << std::endl;
                    for (size_t j = 0; j < ext_componenet_list[i].size(); ++j) {
                        std::cout << "  Value " << j + 1 << ": ";
                        for (const auto& value : ext_componenet_list[i][j]) {
                            std::cout << static_cast<uint8_t>(value) << " ";
                        }
                        std::cout << std::endl;
                    }
                }
            } break;
            default: {
                std::cout << "Not a valid pipeline type ! Exiting!\n";
                return -1;
            }
        }*/
        auto last_colot_temp = rocalGetIntValue(color_temp_adj);
        rocalUpdateIntParameter(last_colot_temp + 1, color_temp_adj);

        rocalCopyToOutput(handle, mat_input.data, h * w * p);

        std::vector<int> compression_params;
        compression_params.push_back(IMWRITE_PNG_COMPRESSION);
        compression_params.push_back(9);

        mat_input.copyTo(mat_output(cv::Rect(col_counter * w, 0, w, h)));
        std::string out_filename = std::string(outName) + ".png";  // in case the user specifies non png filename
        if (display_all)
            out_filename = std::string(outName) + "_p1" + std::to_string(index) + ".png";  // in case the user specifies non png filename

        if (color_format == RocalImageColor::ROCAL_COLOR_RGB24) {
            cv::cvtColor(mat_output, mat_color, CV_RGB2BGR);
            if (DISPLAY)
                cv::imshow("output", mat_output);
            else
                cv::imwrite(out_filename, mat_color, compression_params);
        } else {
            if (DISPLAY)
                cv::imshow("output", mat_output);
            else
                cv::imwrite(out_filename, mat_output, compression_params);
        }
        col_counter = (col_counter + 1) % number_of_cols;
    }
    

// >>>>>>>>>>>>>> Second handle case


    while (rocalGetRemainingImages(second_handle) >= input_batch_size) {
        index++;
        if (rocalRun(second_handle) != 0)
            break;
        int image_name_length[input_batch_size];
        /*switch (pipeline_type) {
            case 1: {   // classification pipeline
                RocalTensorList labels = rocalGetImageLabels(second_handle);
                int *label_id = reinterpret_cast<int *>(labels->at(0)->buffer());  // The labels are present contiguously in memory
                int img_size = rocalGetImageNameLen(second_handle, image_name_length);
                std::vector<char> img_name(img_size);
                std::vector<int> label_one_hot_encoded(input_batch_size * num_of_classes);
                rocalGetImageName(second_handle, img_name.data());
                if (num_of_classes != 0) {
                    rocalGetOneHotImageLabels(second_handle, label_one_hot_encoded.data(), num_of_classes, RocalOutputMemType::ROCAL_MEMCPY_HOST);
                }
                std::cerr << "\nImage name:" << img_name.data() << "\n";
                for (unsigned int i = 0; i < input_batch_size; i++) {
                    std::cerr << "Label id: " << label_id[i] << std::endl;
                    if(num_of_classes != 0)
                    {
                        std::cout << "One Hot Encoded labels:"<<"\t";
                        for (int j = 0; j < num_of_classes; j++)
                        {
                            int idx_value = label_one_hot_encoded[(i*num_of_classes)+j];
                            if(idx_value == 0)
                                std::cout << idx_value << "\t";
                            else
                            {
                                std::cout << idx_value << "\t";
                            }
                        }
                    }
                    std::cout << "\n";
                }
            } break;
            case 2: {   // detection pipeline
                int img_size = rocalGetImageNameLen(second_handle, image_name_length);
                std::vector<char> img_name(img_size);
                rocalGetImageName(second_handle, img_name.data());
                std::cerr << "\nImage name:" << img_name.data();
                RocalTensorList bbox_labels = rocalGetBoundingBoxLabel(second_handle);
                RocalTensorList bbox_coords = rocalGetBoundingBoxCords(second_handle);
                for (unsigned i = 0; i < bbox_labels->size(); i++) {
                    int *labels_buffer = reinterpret_cast<int *>(bbox_labels->at(i)->buffer());
                    float *bbox_buffer = reinterpret_cast<float *>(bbox_coords->at(i)->buffer());
                    std::cerr << "\nBBOX Labels : ";
                    for (unsigned j = 0; j < bbox_labels->at(i)->dims().at(0); j++)
                        std::cerr << labels_buffer[j] << " ";
                    std::cerr << "\nBBOX Count: " << bbox_coords->at(i)->dims().at(0) << "\n";
                    for (unsigned j = 0, j4 = 0; j < bbox_coords->at(i)->dims().at(0); j++, j4 = j * 4)
                        std::cerr << bbox_buffer[j4] << " " << bbox_buffer[j4 + 1] << " " << bbox_buffer[j4 + 2] << " " << bbox_buffer[j4 + 3] << "\n";
                }
                int img_sizes_batch[input_batch_size * 2];
                rocalGetImageSizes(second_handle, img_sizes_batch);
                for (int i = 0; i < (int)input_batch_size; i++) {
                    std::cout << "\nwidth:" << img_sizes_batch[i * 2];
                    std::cout << "\nHeight:" << img_sizes_batch[(i * 2) + 1];
                }
            } break;
            case 3: {   // keypoints pipeline
                int size = input_batch_size;
                RocalJointsData *joints_data;
                rocalGetJointsDataPtr(second_handle, &joints_data);
                for (int i = 0; i < size; i++) {
                    std::cout << "ImageID: " << joints_data->image_id_batch[i] << std::endl;
                    std::cout << "AnnotationID: " << joints_data->annotation_id_batch[i] << std::endl;
                    std::cout << "ImagePath: " << joints_data->image_path_batch[i] << std::endl;
                    std::cout << "Center: " << joints_data->center_batch[i][0] << " " << joints_data->center_batch[i][1] << std::endl;
                    std::cout << "Scale: " << joints_data->scale_batch[i][0] << " " << joints_data->scale_batch[i][1] << std::endl;
                    std::cout << "Score: " << joints_data->score_batch[i] << std::endl;
                    std::cout << "Rotation: " << joints_data->rotation_batch[i] << std::endl;

                    for (int k = 0; k < 17; k++) {
                        std::cout << "x : " << joints_data->joints_batch[i][k][0] << " , y : " << joints_data->joints_batch[i][k][1] << " , v : " << joints_data->joints_visibility_batch[i][k][0] << std::endl;
                    }
                }
            } break;
            case 4: {   // webdataset pipeline
                int img_size = rocalGetImageNameLen(second_handle, image_name_length);
                std::vector<char> img_name(img_size);
                rocalGetImageName(second_handle, img_name.data());
                std::cout << "\n Image name: " << img_name.data() << "\n \n";
                RocalMetaData ascii_sample_contents = rocalGetAsciiDatas(second_handle);
                std::vector<std::vector<std::vector<uint8_t>>> ext_componenet_list;
                for(uint ext = 0; ext < ascii_sample_contents->size(); ext++) {
                    RocalTensorList ext_ascii_values_batch = ascii_sample_contents->at(ext);
                    std::vector<std::vector<uint8_t>> component_list;
                    std::vector<uint8_t> ascii_components_array;
                    for (uint i = 0; i < ext_ascii_values_batch->size(); i++) {
                        if (ext_ascii_values_batch->at(i)->buffer() !=  nullptr) {
                            uint8_t* buffer = reinterpret_cast<uint8_t*>(ext_ascii_values_batch->at(i)->buffer());
                            size_t length = ext_ascii_values_batch->at(i)->dims().at(0);
                            ascii_components_array.assign(buffer, buffer + length);
                        } else {
                            ascii_components_array = std::vector<uint8_t>{};
                        }
                        component_list.push_back(ascii_components_array);
                    }
                    ext_componenet_list.push_back(component_list);
                }
                for (size_t i = 0; i < ext_componenet_list.size(); ++i) {
                    std::cout << " Meta Data Component " << i + 1 << ":" << std::endl;
                    for (size_t j = 0; j < ext_componenet_list[i].size(); ++j) {
                        std::cout << "  Value " << j + 1 << ": ";
                        for (const auto& value : ext_componenet_list[i][j]) {
                            std::cout << static_cast<uint8_t>(value) << " ";
                        }
                        std::cout << std::endl;
                    }
                }
            } break;
            default: {
                std::cout << "Not a valid pipeline type ! Exiting!\n";
                return -1;
            }
        }*/
        auto last_colot_temp = rocalGetIntValue(color_temp_adj);
        rocalUpdateIntParameter(last_colot_temp + 1, color_temp_adj);

        rocalCopyToOutput(second_handle, mat_input.data, h * w * p);

        std::vector<int> compression_params;
        compression_params.push_back(IMWRITE_PNG_COMPRESSION);
        compression_params.push_back(9);

        mat_input.copyTo(mat_output(cv::Rect(col_counter * w, 0, w, h)));
        std::string out_filename = std::string(outName) + ".png";  // in case the user specifies non png filename
        if (display_all)
            out_filename = std::string(outName) + std::to_string(index) + ".png";  // in case the user specifies non png filename

        if (color_format == RocalImageColor::ROCAL_COLOR_RGB24) {
            cv::cvtColor(mat_output, mat_color, CV_RGB2BGR);
            if (DISPLAY)
                cv::imshow("output", mat_output);
            else
                cv::imwrite(out_filename, mat_color, compression_params);
        } else {
            if (DISPLAY)
                cv::imshow("output", mat_output);
            else
                cv::imwrite(out_filename, mat_output, compression_params);
        }
        col_counter = (col_counter + 1) % number_of_cols;
    }
    
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto dur = duration_cast<microseconds>(t2 - t1).count();
    auto rocal_timing = rocalGetTimingInfo(handle);
    std::cout << "Load     time " << rocal_timing.load_time << std::endl;
    std::cout << "Decode   time " << rocal_timing.decode_time << std::endl;
    std::cout << "Process  time " << rocal_timing.process_time << std::endl;
    std::cout << "Transfer time " << rocal_timing.transfer_time << std::endl;
    std::cout << "Total Elapsed Time " << dur / 1000000 << " sec " << dur % 1000000 << " us " << std::endl;
    rocalRelease(handle);
    rocalRelease(second_handle);
    // mat_input.release();
    // mat_output.release();
    if (!output)
        return -1;
    return 0;
}
