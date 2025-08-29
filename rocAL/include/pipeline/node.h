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
#include <tuple>

#include "pipeline/graph.h"
#include "loaders/loader_module.h"
// #include "meta_data/meta_data_graph.h"
#include "pipeline/tensor.h"
#include "parameters/parameter_factory.h"
#include "pipeline/commons.h"
#include "decoders/image/decoder.h"
#include "readers/image/image_reader.h"

// Custom type traits to check the vector types
template <typename T>
struct is_vector_type : std::false_type {};

template <typename T, typename Alloc>
struct is_vector_type<std::vector<T, Alloc>> : std::true_type {};

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
    pParam param;

    template <typename T>
    T Get() const {
        std::cerr << "Type name -> " << type_name << "\n";

        // Compile-time check for parameter types
        if constexpr (std::is_same_v<T, FloatParam*> || std::is_same_v<T, IntParam*>) {
            if (is_null_ptr) {
                return nullptr;
            }
            return std::any_cast<T>(param);
        } else {
            if (!is_null_ptr) {
                if (!is_vector) {
                    return std::any_cast<T>(values[0]);
                } else if (is_vector) {
                    return std::any_cast<T>(values);
                }
            } else {
                THROW("Undefined type passed")
            }
        }
    }

    template<>
    std::map<std::string, std::string> Get<std::map<std::string, std::string>>() const {
        std::map<std::string, std::string> feature_map;
        for (int i = 0; i < values.size(); i+=2) {
            feature_map[std::any_cast<std::string>(values[i])] = std::any_cast<std::string>(values[i + 1]);
        }
        return feature_map;
    }

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
        {typeid(RocalResizeScalingMode), "RocalResizeScalingMode"},
        {typeid(RocalMemType), "RocalMemType"},
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
        } else if constexpr (is_vector_type<T>::value) {
            using ElementType = typename std::decay_t<T>::value_type;
            auto it = type_names.find(typeid(ElementType));
            if (it != type_names.end()) {
                type_name = it->second;                
                is_vector = true;
                for (const auto& v : val) {
                    // values.push_back(v);  // Store std::string as std::any
                    values.push_back(static_cast<ElementType>(v));
                }
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
    inline void extract_param(const RocalParameterType param_type, pParam parameter) {
        if (param_type == RocalParameterType::DETERMINISTIC) {
            enum_type_name = "SimpleParameter";
        } else if (param_type == RocalParameterType::RANDOM_UNIFORM) {
            enum_type_name = "UniformRand";
        } else if (param_type == RocalParameterType::RANDOM_CUSTOM) {
            enum_type_name = "CustomRand";
        }
        param = parameter;
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
        extract_param(param->type, param);
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
        extract_param(param->type, param);
    }

    Argument() {}
};

template <typename... Args, std::size_t... I>
std::tuple<Args...> unpack_arguments_impl(const std::vector<Argument>& arguments, std::index_sequence<I...>) {
    return std::make_tuple(arguments[I].Get<Args>()...);
}

// Helper: extract arguments into a tuple using index sequence
template <typename... Args>
std::tuple<Args...> unpack_arguments(const std::vector<Argument>& arguments) {
    return unpack_arguments_impl<Args...>(arguments, std::index_sequence_for<Args...>{});
}

template <typename NodeType, typename... Args>
bool try_init_with(NodeType* node, const std::vector<Argument>& arguments) {
    if (arguments.size() != sizeof...(Args)) return false;

    try {
        // Unpack arguments with type-check and casting
        // For C++ >= 20
        // std::tuple<Args...> unpacked_args = [&]<std::size_t... I>(std::index_sequence<I...>) {
        //     return std::make_tuple(arguments[I].Get<Args>()...);
        // }(std::index_sequence_for<Args...>{});

        auto unpacked_args = unpack_arguments<Args...>(arguments);

        std::apply([&](Args... unpacked) {
            node->init(std::forward<Args>(unpacked)...);
        }, unpacked_args);

        return true;
    } catch (const std::exception& e) {
        return false; // Type mismatch
    }
}

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
    void add_next(const std::shared_ptr<Node> &node);   // Adds the Node next to the current Node
    void add_previous(const std::shared_ptr<Node> &node);   // Adds the Node preceding the current Node
    void release();
    std::shared_ptr<Graph> graph() { return _graph; }
    void set_meta_data(pMetaDataBatch meta_data_info) { _meta_data_info = meta_data_info; }
    bool _is_ssd = false;
    const Roi2DCords *get_src_roi() { return _inputs[0]->info().roi().get_2D_roi(); }
    const Roi2DCords *get_dst_roi() { return _outputs[0]->info().roi().get_2D_roi(); }
    void set_graph_id(int id) { _graph_id = id; }
    int get_graph_id() { return _graph_id; }
    virtual std::string node_name() { return ""; }
    std::vector<Argument> get_args_list() { return _args; }
    virtual std::shared_ptr<LoaderModule> get_loader_module() { THROW("Not Implemented") }
    virtual void initalize_args(std::vector<Argument> &arguments, std::shared_ptr<MetaDataReader> meta_data_reader) { THROW("Not Implemented") }
    virtual void initalize_args(std::vector<Argument> &arguments) { THROW("Not Implemented") }

   protected:
    virtual void create_node() = 0;
    virtual void update_node() = 0;
    const std::vector<Tensor *> _inputs;
    const std::vector<Tensor *> _outputs;
    std::shared_ptr<Graph> _graph = nullptr;
    vx_node _node = nullptr;
    size_t _batch_size;
    pMetaDataBatch _meta_data_info;
    std::vector<std::shared_ptr<Node>> _next;   // Stores the reference to a list of next Nodes
    std::vector<std::shared_ptr<Node>> _prev;   // Stores the reference to a list of previous Nodes
    int _graph_id = -1;
    std::vector<Argument> _args;
    template <size_t N, size_t... Indices, typename... Args>
    void set_node_arguments(std::array<std::string, N>& arg_names, std::index_sequence<Indices ...>, Args... args) {
        // Fold expression to create Argument object for each argument in the node
        (this->_args.push_back(Argument(arg_names[Indices], std::forward<Args>(args))), ...);
    }
};

class NodeFactory {
public:
    using LoaderCreator = std::function<std::shared_ptr<Node>(Tensor*, void*)>;
    using AugmentationCreator = std::function<std::shared_ptr<Node>(const std::vector<Tensor *>&, const std::vector<Tensor *>&)>;

    static NodeFactory& instance() {
        static NodeFactory factory;
        return factory;
    }

    void register_loader_node(const std::string& name, LoaderCreator creator) {
        _loader_node_registry[name] = std::move(creator);
    }

    void register_node(const std::string& name, AugmentationCreator creator) {
        _node_registry[name] = std::move(creator);
    }

    std::shared_ptr<Node> create_loader_node(const std::string& name, Tensor* output_tensor, void *dev_resource) const {
        auto it = _loader_node_registry.find(name);
        if (it != _loader_node_registry.end()) {
            return it->second(output_tensor, dev_resource);
        } else {
            THROW("The given node not found in the registry" + name)
        }
    }

    std::shared_ptr<Node> create_node(const std::string& name, const std::vector<Tensor *>& inputs, const std::vector<Tensor *>& outputs) const {
        auto it = _node_registry.find(name);
        if (it != _node_registry.end()) {
            return it->second(inputs, outputs);
        } else {
            THROW("The given node not found in the registry" + name)
        }
    }

private:
    std::map<std::string, LoaderCreator> _loader_node_registry;
    std::map<std::string, AugmentationCreator> _node_registry;
};

// template<typename T>
// class NodeRegistrar {
// public:
//     NodeRegistrar(const std::string& name) {
//         NodeFactory::instance().register_node(name, []() -> std::shared_ptr<Node> {
//             return std::make_shared<T>();
//         });
//     }
// };

// Macro to define static registrar for the class
// #define REGISTER_NODE(CLASS_NAME) \
//     static struct NodeRegistrar<CLASS_NAME> _##CLASS_NAME##_registrar(#CLASS_NAME);

#define REGISTER_LOADER_NODE(CLASS_NAME) \
    static struct CLASS_NAME##_NodeRegistrar { \
        CLASS_NAME##_NodeRegistrar() { \
            NodeFactory::instance().register_loader_node(#CLASS_NAME, [](Tensor *output, void *dev_resources) { \
                return std::make_shared<CLASS_NAME>(output, dev_resources); \
            }); \
        } \
    } _##CLASS_NAME##_registrar;

#define REGISTER_NODE(CLASS_NAME) \
    static struct CLASS_NAME##_NodeRegistrar { \
        CLASS_NAME##_NodeRegistrar() { \
            NodeFactory::instance().register_node(#CLASS_NAME, [](const std::vector<Tensor *>& inputs, const std::vector<Tensor *>& outputs) { \
                return std::make_shared<CLASS_NAME>(inputs, outputs); \
            }); \
        } \
    } _##CLASS_NAME##_registrar;
