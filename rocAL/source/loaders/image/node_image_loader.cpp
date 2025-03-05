/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

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

#include "loaders/image/node_image_loader.h"

#include "pipeline/exception.h"

ImageLoaderNode::ImageLoaderNode(Tensor *output, void *device_resources) : Node({}, {output}) {
    _loader_module = std::make_shared<ImageLoaderSharded>(device_resources);
}

void ImageLoaderNode::init(unsigned internal_shard_count, unsigned cpu_num_threads, const std::string &source_path, const std::string &json_path, const std::map<std::string, std::string> feature_key_map, StorageType storage_type, DecoderType decoder_type,
                           bool shuffle, bool loop, size_t load_batch_count, RocalMemType mem_type, std::shared_ptr<MetaDataReader> meta_data_reader, bool decoder_keep_orig, const ShardingInfo& sharding_info, const char *file_prefix, unsigned sequence_length, 
                           unsigned step, unsigned stride, ExternalSourceFileMode external_file_mode, const std::string &index_path) {
    if (!_loader_module)
        THROW("ERROR: loader module is not set for ImageLoaderNode, cannot initialize")
    if (internal_shard_count < 1)
        THROW("Shard count should be greater than or equal to one")
    _loader_module->set_output(_outputs[0]);
    // Set reader and decoder config accordingly for the ImageLoaderNode
    auto reader_cfg = ReaderConfig(storage_type, source_path, json_path, feature_key_map, shuffle, loop);
    reader_cfg.set_shard_count(internal_shard_count);
    reader_cfg.set_cpu_num_threads(cpu_num_threads);
    reader_cfg.set_batch_count(load_batch_count);
    reader_cfg.set_file_prefix(file_prefix);
    reader_cfg.set_meta_data_reader(meta_data_reader);
    //  sequence_length, step and stride parameters used only for SequenceReader
    reader_cfg.set_sequence_length(sequence_length);
    reader_cfg.set_frame_step(step);
    reader_cfg.set_frame_stride(stride);
    reader_cfg.set_external_filemode(external_file_mode);
    reader_cfg.set_index_path(index_path);
    reader_cfg.set_sharding_info(sharding_info);


    // Convert map to vector of strings
    std::vector<std::string> feature_vector;
    if (!feature_key_map.empty()) {
        for (const auto& pair : feature_key_map) {
            feature_vector.push_back(pair.first);  // Push key
            feature_vector.push_back(pair.second); // Push value
        }
    }

    // Add all arguments as part of the operator
    this->_args.push_back(Argument("internal_shard_count", "unsigned", internal_shard_count));
    this->_args.push_back(Argument("cpu_num_threads", "unsigned", cpu_num_threads));
    this->_args.push_back(Argument("source_path", "string", source_path));
    this->_args.push_back(Argument("json_path", "string", json_path));
    // Feature key Map
    this->_args.push_back(Argument("feature_key_map", "map_string", feature_vector));
    this->_args.push_back(Argument("storage_type", "int", "StorageType", static_cast<int>(storage_type)));
    this->_args.push_back(Argument("decoder_type", "int", "DecoderType", static_cast<int>(decoder_type)));
    this->_args.push_back(Argument("shuffle", "bool", shuffle));
    this->_args.push_back(Argument("loop", "bool", loop));
    this->_args.push_back(Argument("load_batch_count", "size_t", load_batch_count));
    this->_args.push_back(Argument("meta_data_reader", "shared_ptr", 0));
    this->_args.push_back(Argument("decoder_keep_orig", "bool", decoder_keep_orig));
    // this->_args.push_back(Argument("sharding_info", "int", "ShardingInfo", static_cast<int>(sharding_info)));
    // ShardingInfo to be added
    this->_args.push_back(Argument("last_batch_policy", "int", "RocalBatchPolicy", static_cast<int>(sharding_info.last_batch_policy)));
    this->_args.push_back(Argument("pad_last_batch_repeated", "bool", sharding_info.pad_last_batch_repeated));
    this->_args.push_back(Argument("stick_to_shard", "bool", sharding_info.stick_to_shard));
    this->_args.push_back(Argument("shard_size", "int", sharding_info.shard_size));


    this->_args.push_back(Argument("file_prefix", "char_str", std::string(file_prefix)));
    this->_args.push_back(Argument("sequence_length", "unsigned", sequence_length));
    this->_args.push_back(Argument("step", "unsigned", step));
    this->_args.push_back(Argument("stride", "unsigned", stride));
    this->_args.push_back(Argument("external_file_mode", "int", "ExternalSourceFileMode", static_cast<int>(external_file_mode)));
    this->_args.push_back(Argument("index_path", "string", index_path));

    _loader_module->initialize(reader_cfg, DecoderConfig(decoder_type),
                               mem_type,
                               _batch_size, decoder_keep_orig);
    _loader_module->start_loading();
}

std::shared_ptr<LoaderModule> ImageLoaderNode::get_loader_module() {
    if (!_loader_module)
        WRN("ImageLoaderNode's loader module is null, not initialized")
    return _loader_module;
}

ImageLoaderNode::~ImageLoaderNode() {
    _loader_module = nullptr;
}
