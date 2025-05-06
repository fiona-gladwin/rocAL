/*
Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc. All rights reserved.

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

#pragma once
#include <memory>
#include <set>
#include <any>
#include <type_traits>
#include <typeindex>
#include <unordered_map>
#include <map>

#include "pipeline/graph.h"
#include "meta_data/meta_data_graph.h"
#include "pipeline/tensor.h"
#include "parameters/parameter_factory.h"
#include "pipeline/commons.h"
#include "decoders/image/decoder.h"
#include "readers/image/image_reader.h"

// Argument class stores the details of each argument in the Node
class Argument {
   public:
    std::string arg_name;   // Name of the argument
    std::string type_name;  // Denotes the data type of the argument
    std::string enum_type_name; // Denotes the name of the enum <arg_name_enum>
    // TODO - Make an enum
    bool is_vector = false;
    bool is_parameter = false;
    bool is_null_ptr = false;
    std::vector<std::any> values;   // Can change to std::variant later
    pParamCore param_core;
    
    // unordered map, mapping the type with the string
    std::unordered_map<std::type_index, std::string> type_names = {
        {typeid(int), "int"},
        {typeid(unsigned), "unsigned"},
        {typeid(size_t), "size_t"},
        {typeid(float), "float"},
        {typeid(double), "double"},
        {typeid(bool), "bool"},
        {typeid(std::string), "string"},
        {typeid(char *), "char_str"},
        {typeid(const char *), "char_str"},
        {typeid(DecoderType), "DecoderType"},
        {typeid(StorageType), "StorageType"},
        {typeid(ExternalSourceFileMode), "ExternalSourceFileMode"},
        {typeid(RocalBatchPolicy), "RocalBatchPolicy"},
        {typeid(RocalResizeInterpolationType), "RocalResizeInterpolationType"},
    };

    template <typename T>
    explicit inline Argument(const std::string& name, const T&& val)
        : arg_name(name) {
        if constexpr (std::is_enum<T>::value) {
            type_name = "int"; // Enum types are stored as integers by default
            
            auto it = type_names.find(typeid(std::decay_t<T>));
            if (it != type_names.end()) {
                enum_type_name = it->second;
                values.push_back(static_cast<int>(val));
            } else {
                std::cout << "Type: Unknown" << arg_name << std::endl;
            }
        } else {
            auto it = type_names.find(typeid(std::decay_t<T>));
            if (it != type_names.end()) {
                type_name = it->second;

                if constexpr (std::is_same<std::decay_t<T>, const char *>::value) {
                    values.push_back(std::string(val));
                } else {
                    values.push_back(static_cast<std::decay_t<T>>(val));
                }
            } else {
                std::cout << "Type: Unknown" << arg_name << std::endl;
            }
        }
    }

    // Used to store the feature key map
    explicit inline Argument(const std::string& name, const std::map<std::string, std::string>&& val)
        : arg_name(name) {
        type_name = "map_string";
        is_vector = true;
        if (!val.empty()) {
            for (const auto& pair : val) {
                values.push_back(static_cast<std::string>(pair.first));  // Push key
                values.push_back(static_cast<std::string>(pair.second)); // Push value
            }
        }
    }

    // Used to store the shared_ptr
    template <typename T>
    explicit inline Argument(const std::string& name, const std::shared_ptr<T>&& val)
        : arg_name(name) {
        type_name = "shared_ptr";

        // For MetadataReader case store an empty value
        // During deserialization the MetadataReader should be created and passed from the MasterGraph.
        if (name == "meta_data_reader") {
            values.push_back(static_cast<int>(0));
        }
    }

    // Contructor to initialize the arguments of in-built data types
    template <typename T>
    explicit inline Argument(std::string name, std::string type, T val)
        : arg_name(name), type_name(type) {
        values.push_back(val);
    }

    // Constructor for vector type arguments
    template <typename T>
    explicit inline Argument(std::string name, std::string type, std::vector<T> &val)
        : arg_name(name), type_name(type) {
        is_vector = true;
        for (const auto& v : val) {
            values.push_back(v);  // Store std::string as std::any
        }
    }

    // Constructor for enum type arguments
    template <typename T>
    explicit inline Argument(std::string name, std::string type, std::string enum_name, T val)
        : arg_name(name), type_name(type), enum_type_name(enum_name) {
        values.push_back(val);
    }

    // Deduces the type of parameter of the argument
    inline void extract_param(const RocalParameterType param_type, pParamCore param) {
        if (param_type == RocalParameterType::DETERMINISTIC) {
            enum_type_name = "SimpleParameter";
        } else if (param_type == RocalParameterType::RANDOM_UNIFORM) {
            enum_type_name = "UniformRand";
        } else if (param_type == RocalParameterType::RANDOM_CUSTOM) {
            enum_type_name = "CustomRand";
        }
        param_core = param;
        is_parameter = true;
    }

    // Constructor for FloatParam arguments
    explicit inline Argument(std::string name, FloatParam* param)
        : arg_name(name) {
        type_name = "float";
        if (param == nullptr) {
            is_null_ptr = true;
            type_name = "nullptr";
            return;
        }
        extract_param(param->type, pParamCore(core(param)));
    }

    // Constructor for IntParam arguments
    explicit inline Argument(std::string name, IntParam* param)
        : arg_name(name) {
        type_name = "int";
        if (param == nullptr) {
            type_name = "nullptr";
            is_null_ptr = true;
            return;
        }
        extract_param(param->type, pParamCore(core(param)));
    }
};

class Node {
   public:
    Node(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) : _inputs(inputs),
                                                                                      _outputs(outputs),
                                                                                      _batch_size(outputs[0]->info().batch_size()) {}
    virtual ~Node();
    void create(std::shared_ptr<Graph> graph);
    void update_parameters();
    std::vector<Tensor *> input() { return _inputs; };
    std::vector<Tensor *> output() { return _outputs; };
    void add_next(const std::shared_ptr<Node> &node) {}      // To be implemented
    void add_previous(const std::shared_ptr<Node> &node) {}  // To be implemented
    std::shared_ptr<Graph> graph() { return _graph; }
    void set_meta_data(pMetaDataBatch meta_data_info) { _meta_data_info = meta_data_info; }
    bool _is_ssd = false;
    const Roi2DCords *get_src_roi() { return _inputs[0]->info().roi().get_2D_roi(); }
    const Roi2DCords *get_dst_roi() { return _outputs[0]->info().roi().get_2D_roi(); }
    virtual std::string node_name() { return ""; }
    std::vector<Argument> get_args_list() { return _args; }

   protected:
    virtual void create_node() = 0;
    virtual void update_node() = 0;
    const std::vector<Tensor *> _inputs;
    const std::vector<Tensor *> _outputs;
    std::shared_ptr<Graph> _graph = nullptr;
    vx_node _node = nullptr;
    size_t _batch_size;
    pMetaDataBatch _meta_data_info;
    std::vector<Argument> _args;

    template <size_t N, size_t... Indices, typename... Args>
    void set_node_arguments(std::array<std::string, N>& arg_names, std::index_sequence<Indices ...>, Args... args) {
        // Fold expression to create Argument object for each argument in the node
        (this->_args.push_back(Argument(arg_names[Indices], std::forward<Args>(args))), ...);
    }
};
