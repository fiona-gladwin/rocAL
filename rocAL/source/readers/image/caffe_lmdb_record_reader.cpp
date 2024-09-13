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

#include "readers/image/caffe_lmdb_record_reader.h"

#include "pipeline/commons.h"
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <math.h>

using namespace std;
using caffe_protos::Datum;

CaffeLMDBRecordReader::CaffeLMDBRecordReader()
{
    _sub_dir = nullptr;
    _curr_file_idx = 0;
    _current_file_size = 0;
    _loop = false;
    _shuffle = false;
    _last_rec = false;
    _file_count_all_shards = 0;
}

unsigned CaffeLMDBRecordReader::count_items() {
    int ret = 0;
    if (_shard_size == -1) {                                     // When shard_size is set to -1, The shard_size variable is not used
        if (_loop) return largest_shard_size_without_padding();  // Return the size of the largest shard amongst all the shard's size
        int size = std::max(largest_shard_size_without_padding(), _batch_size);
        ret = (size - _read_counter);
        // Formula used to calculate - [_last_batch_padded_size = _batch_size - (_shard_size % _batch_size) ]
        // Since "size" doesnt involve padding - we add the count of padded samples to the number of remaining elements
        // which equals to the shard size with padding
        if (_last_batch_info.last_batch_policy == RocalBatchPolicy::PARTIAL || _last_batch_info.last_batch_policy == RocalBatchPolicy::FILL) {
            ret += _last_batch_padded_size;
        } else if (_last_batch_info.last_batch_policy == RocalBatchPolicy::DROP &&
                   _last_batch_info.pad_last_batch_repeated == true) {  // When pad_last_batch_repeated is False - Enough
                                                       // number of samples would not be present in the last batch - hence
                                                       // dropped by condition handled in the loader
            ret -= _batch_size;
        }
    } else if (_shard_size > 0) {
        auto largest_shard_size_with_padding =
            _shard_size + (_batch_size - (_shard_size % _batch_size));  // The shard size used here is padded
        if (_loop)
            return largest_shard_size_with_padding;
        int size = std::max(largest_shard_size_with_padding, _batch_size);
        ret = (size - _read_counter);
        if (_last_batch_info.last_batch_policy == RocalBatchPolicy::DROP)  // The shard size is padded at the beginning of the condition, hence dropping the last batch
            ret -= _batch_size;
    }
    return ((ret < 0) ? 0 : ret);
}

Reader::Status CaffeLMDBRecordReader::initialize(ReaderConfig desc) {
    auto ret = Reader::Status::OK;
    _folder_path = desc.path();
    _path = desc.path();
    _shard_id = desc.get_shard_id();
    _shard_count = desc.get_shard_count();
    _batch_size = desc.get_batch_size();
    _loop = desc.loop();
    _shuffle = desc.shuffle();
    _meta_data_reader = desc.meta_data_reader();
    _last_batch_info = desc.get_sharding_info();
    _pad_last_batch_repeated = _last_batch_info.pad_last_batch_repeated;
    _stick_to_shard = _last_batch_info.stick_to_shard;
    _shard_size = _last_batch_info.shard_size;
    ret = folder_reading();
    _curr_file_idx = _shard_start_idx_vector[_shard_id]; // shard's start_idx would vary for every shard in the vector
    // shuffle dataset if set
    std::random_shuffle(_all_shard_file_names_padded.begin() + _shard_start_idx_vector[_shard_id],
                        _all_shard_file_names_padded.begin() + _shard_start_idx_vector[_shard_id] + actual_shard_size_without_padding());

    return ret;
}

void CaffeLMDBRecordReader::increment_curr_file_idx() {
    // The condition satisfies for both pad_last_batch = True (or) False
    if (_stick_to_shard == false) {  // The elements of each shard rotate in a round-robin fashion once the elements in particular shard is exhausted
        _curr_file_idx = (_curr_file_idx + 1) % _file_names.size();
    } else {  // Stick to only elements from the current shard
        if (_curr_file_idx >= _shard_start_idx_vector[_shard_id] &&
            _curr_file_idx < _shard_end_idx_vector[_shard_id])  // checking if current-element lies within the shard size [begin_idx, last_idx -1]
            _curr_file_idx = (_curr_file_idx + 1);
        else
            _curr_file_idx = _shard_start_idx_vector[_shard_id];
    }
}

void CaffeLMDBRecordReader::incremenet_read_ptr() {
    _read_counter++;
    increment_curr_file_idx();
}
size_t CaffeLMDBRecordReader::open() {
    auto file_path = _all_shard_file_names_padded[_curr_file_idx];  // Get next file name
    _last_id = file_path;
    _current_file_size = _file_size[_all_shard_file_names_padded[_curr_file_idx]];
    return _current_file_size;
}

size_t CaffeLMDBRecordReader::read_data(unsigned char *buf, size_t read_size) {
    read_image(buf, _all_shard_file_names_padded[_curr_file_idx]);
    incremenet_read_ptr();
    return read_size;
}

int CaffeLMDBRecordReader::close() {
    return release();
}

CaffeLMDBRecordReader::~CaffeLMDBRecordReader() {
    _open_env = 0;
    mdb_txn_abort(_read_mdb_txn);
    mdb_close(_read_mdb_env, _read_mdb_dbi);
    mdb_env_close(_read_mdb_env);
    _read_mdb_txn = nullptr;
    _read_mdb_env = nullptr;
    release();
}

int CaffeLMDBRecordReader::release() {
    mdb_cursor_close(_mdb_cursor);
    mdb_txn_abort(_mdb_txn);
    mdb_close(_mdb_env, _mdb_dbi);

    mdb_env_close(_mdb_env);
    _mdb_cursor = nullptr;
    _mdb_txn = nullptr;
    _mdb_env = nullptr;
    return 0;
}

void CaffeLMDBRecordReader::reset() {
    if (_shuffle)
        std::random_shuffle(_all_shard_file_names_padded.begin() + _shard_start_idx_vector[_shard_id],
                            _all_shard_file_names_padded.begin() + _shard_start_idx_vector[_shard_id] + actual_shard_size_without_padding());

    if (_stick_to_shard == false)  // Pick elements from the next shard - hence increment shard_id
        increment_shard_id();      // Should work for both single and multiple shards

    _read_counter = 0;

    if (_last_batch_info.last_batch_policy == RocalBatchPolicy::DROP) {  // Skipping the dropped batch in next epoch
        for (uint i = 0; i < _batch_size; i++)
            increment_curr_file_idx();
    }
}

void CaffeLMDBRecordReader::increment_shard_id() {
    _shard_id = (_shard_id + 1) % _shard_count;
}

Reader::Status CaffeLMDBRecordReader::folder_reading() {
    if ((_sub_dir = opendir(_folder_path.c_str())) == nullptr)
        THROW("CaffeLMDBRecordReader ShardID [" + TOSTR(_shard_id) + "] ERROR: Failed opening the directory at " + _folder_path);

    std::string _full_path = _folder_path;
    auto ret = Reader::Status::OK;
    if (Caffe_LMDB_reader() != Reader::Status::OK)
        WRN("CaffeLMDBRecordReader ShardID [" + TOSTR(_shard_id) + "] CaffeLMDBRecordReader cannot access the storage at " + _folder_path);

    if (!_file_names.empty())
        std::cout << "CaffeLMDBRecordReader ShardID [" << TOSTR(_shard_id) << "] Total of " << TOSTR(_file_names.size()) << " images loaded from " << _full_path << std::endl;
    
    auto dataset_size = _file_count_all_shards;
    // Pad the _file_names with last element of the shard in the vector when _pad_last_batch_repeated is True
    _padded_samples = ((_shard_size > 0) ? _shard_size : largest_shard_size_without_padding()) % _batch_size;
    _last_batch_padded_size = (_batch_size > 1) ? (_batch_size - _padded_samples) : 0;
    if (_pad_last_batch_repeated == true) { 
        // pad the last sample when the dataset_size is not divisible by
        // the number of shard's (or) when the shard's size is not
        // divisible by the batch size making each shard having equal
        // number of samples
        for (uint shard_id = 0; shard_id < _shard_count; shard_id++) {
            uint start_idx = (dataset_size * shard_id) / _shard_count;
            uint actual_shard_size_without_padding = std::floor((shard_id + 1) * dataset_size / _shard_count) - floor(shard_id * dataset_size / _shard_count);
            uint largest_shard_size = std::ceil(dataset_size * 1.0 / _shard_count);
            auto start = _file_names.begin() + start_idx;
            auto end = _file_names.begin() + start_idx + actual_shard_size_without_padding;
            if (start != end && start <= _file_names.end() &&
                end <= _file_names.end()) {
                _all_shard_file_names_padded.insert(_all_shard_file_names_padded.end(), start, end);
            }
            if (largest_shard_size % _batch_size) {
                _num_padded_samples = (largest_shard_size - actual_shard_size_without_padding) + _batch_size - (largest_shard_size % _batch_size);
                _file_count_all_shards += _num_padded_samples;
                _all_shard_file_names_padded.insert(_all_shard_file_names_padded.end(), _num_padded_samples, _all_shard_file_names_padded.back());
            }
        }
    } else {
        _all_shard_file_names_padded = _file_names;
    }
    _last_file_name = _all_shard_file_names_padded[_all_shard_file_names_padded.size() - 1];
    compute_start_and_end_idx_of_all_shards();
    closedir(_sub_dir);
    return ret;
}


Reader::Status CaffeLMDBRecordReader::Caffe_LMDB_reader() {
    _open_env = 0;
    string tmp1 = _folder_path + "/data.mdb";
    string tmp2 = _folder_path + "/lock.mdb";
    uint file_size, file_size1;

    ifstream in_file(tmp1, ios::binary);
    in_file.seekg(0, ios::end);
    file_size = in_file.tellg();
    ifstream in_file1(tmp2, ios::binary);
    in_file1.seekg(0, ios::end);
    file_size1 = in_file1.tellg();
    _file_byte_size = file_size + file_size1;
    read_image_names();
    return Reader::Status::OK;
}

void CaffeLMDBRecordReader::read_image_names() {
    int rc;
    // Creating an LMDB environment handle
    CHECK_LMDB_RETURN_STATUS(mdb_env_create(&_mdb_env));
    // Setting the size of the memory map to use for this environment.
    // The size of the memory map is also the maximum size of the database.
    CHECK_LMDB_RETURN_STATUS(mdb_env_set_mapsize(_mdb_env, _file_byte_size));
    // Opening an environment handle.
    CHECK_LMDB_RETURN_STATUS(mdb_env_open(_mdb_env, _path.c_str(), MDB_RDONLY, 0664));
    // Creating a transaction for use with the environment
    CHECK_LMDB_RETURN_STATUS(mdb_txn_begin(_mdb_env, NULL, MDB_RDONLY, &_mdb_txn));
    // Opening a database in the environment.
    CHECK_LMDB_RETURN_STATUS(mdb_open(_mdb_txn, NULL, 0, &_mdb_dbi));
    // Creating a cursor handle.
    // A cursor is associated with a specific transaction and database
    CHECK_LMDB_RETURN_STATUS(mdb_cursor_open(_mdb_txn, _mdb_dbi, &_mdb_cursor));

    // Retrieve by cursor. It retrieves key/data pairs from the database
    while ((rc = mdb_cursor_get(_mdb_cursor, &_mdb_key, &_mdb_value, MDB_NEXT)) == 0) {
        Datum datum;
        caffe_protos::AnnotatedDatum annotatedDatum_protos;
        annotatedDatum_protos.ParseFromArray((char *)_mdb_value.mv_data, _mdb_value.mv_size);
        // Checking image Datum
        int check_image_datum = annotatedDatum_protos.has_datum();

        if (check_image_datum)
            datum = annotatedDatum_protos.datum();  // parse datum for detection
        else
            datum.ParseFromArray((const void *)_mdb_value.mv_data, _mdb_value.mv_size);  // parse datum for classification
        if ((!_meta_data_reader || _meta_data_reader->exists(string((char *)_mdb_key.mv_data).c_str()))) {
            string image_key = string((char *)_mdb_key.mv_data);
            _file_names.push_back(image_key.c_str());
            _last_file_name = image_key.c_str();
            _file_count_all_shards++;
            _last_file_size = datum.data().size();
            _file_size.insert(pair<std::string, unsigned int>(_last_file_name, _last_file_size));
        }
    }

    release();
}


void CaffeLMDBRecordReader::open_env_for_read_image() {
    // Creating an LMDB environment handle
    CHECK_LMDB_RETURN_STATUS(mdb_env_create(&_read_mdb_env));
    // Setting the size of the memory map to use for this environment.
    // The size of the memory map is also the maximum size of the database.
    CHECK_LMDB_RETURN_STATUS(mdb_env_set_mapsize(_read_mdb_env, _file_byte_size));
    // Opening an environment handle.
    CHECK_LMDB_RETURN_STATUS(mdb_env_open(_read_mdb_env, _path.c_str(), MDB_RDONLY, 0664));
    // Creating a transaction for use with the environment
    CHECK_LMDB_RETURN_STATUS(mdb_txn_begin(_read_mdb_env, NULL, MDB_RDONLY, &_read_mdb_txn));
    // Opening a database in the environment.
    CHECK_LMDB_RETURN_STATUS(mdb_open(_read_mdb_txn, NULL, 0, &_read_mdb_dbi));
    _open_env = 1;
}

void CaffeLMDBRecordReader::read_image(unsigned char *buff, std::string file_name) {
    if (_open_env == 0) {
        open_env_for_read_image();
    }

    // Creating a cursor handle.
    // A cursor is associated with a specific transaction and database
    CHECK_LMDB_RETURN_STATUS(mdb_cursor_open(_read_mdb_txn, _read_mdb_dbi, &_read_mdb_cursor));

    string checkedStr = string((char *)file_name.c_str());
    string newStr = checkedStr.substr(0, checkedStr.find(".")) + ".JPEG";

    _read_mdb_key.mv_size = newStr.size();
    _read_mdb_key.mv_data = (char *)newStr.c_str();

    int _mdb_status = mdb_cursor_get(_read_mdb_cursor, &_read_mdb_key, &_read_mdb_value, MDB_SET_RANGE);
    if (_mdb_status == MDB_NOTFOUND) {
        THROW("\nKey Not found");
    } else {
        Datum datum;
        caffe_protos::AnnotatedDatum annotatedDatum_protos;
        annotatedDatum_protos.ParseFromArray((char *)_read_mdb_value.mv_data, _read_mdb_value.mv_size);

        // Checking image Datum
        int check_image_datum = annotatedDatum_protos.has_datum();

        if (check_image_datum)
            datum = annotatedDatum_protos.datum();  // parse datum for detection
        else
            datum.ParseFromArray((const void *)_read_mdb_value.mv_data, _read_mdb_value.mv_size);  // parse datum for classification

        memcpy(buff, datum.data().c_str(), datum.data().size());
    }

    mdb_cursor_close(_read_mdb_cursor);
    _read_mdb_cursor = nullptr;
}

size_t CaffeLMDBRecordReader::last_batch_padded_size() {
    return _last_batch_padded_size;
}

void CaffeLMDBRecordReader::compute_start_and_end_idx_of_all_shards() {
    for (uint shard_id = 0; shard_id < _shard_count; shard_id++) {
        auto start_idx_of_shard = (_file_count_all_shards * shard_id) / _shard_count;
        auto end_idx_of_shard = start_idx_of_shard + actual_shard_size_without_padding() - 1;
        _shard_start_idx_vector.push_back(start_idx_of_shard);
        _shard_end_idx_vector.push_back(end_idx_of_shard);
     
    }
}

size_t CaffeLMDBRecordReader::get_start_idx() {
    _shard_start_idx = (get_dataset_size() * _shard_id) / _shard_count;
    return _shard_start_idx;
}

size_t CaffeLMDBRecordReader::get_dataset_size() {
    return _file_count_all_shards;
}


size_t CaffeLMDBRecordReader::actual_shard_size_without_padding() {
    return std::floor((_shard_id + 1) * _file_count_all_shards / _shard_count) - floor(_shard_id * _file_count_all_shards / _shard_count);
}

size_t CaffeLMDBRecordReader::largest_shard_size_without_padding() {
  return std::ceil(_file_count_all_shards * 1.0 / _shard_count);
}
