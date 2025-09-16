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
#include <any>
#include <type_traits>
#include <typeindex>
#include <unordered_map>
#include <map>
#include <vector>
#include <string>
#include <stdexcept>

#include "pipeline/argument_types.h"
#include "pipeline/enum_registry.h"
#include "parameters/parameter_factory.h"
#include "pipeline/commons.h"

/**
 * @brief Argument class stores the details of each argument in the Node
 * 
 * This class encapsulates argument information for pipeline nodes, supporting
 * various data types including basic types, enums, vectors, maps, and parameters. 
 */
class Argument {
   public:
    std::string arg_name;       ///< Name of the argument
    std::string type_name;      ///< Denotes the data type of the argument
    std::string enum_type_name; ///< Denotes the name of the enum <arg_name_enum>
    // TODO - Make an enum for type_name instead of string
    bool is_vector = false;     ///< True if the argument contains vector data
    bool is_parameter = false;  ///< True if the argument is a parameter object
    bool is_null_ptr = false;   ///< True if the argument represents a null pointer
    std::vector<std::any> values; ///< Storage for argument values (can change to std::variant later)
    pParam param;               ///< Parameter core for parameter-type arguments
    
   private:
    // Helper method to get type name from registry or built-in types
    template<typename T>
    std::string getTypeName() const {
        using DecayedType = std::decay_t<T>;
        
        if constexpr (std::is_enum_v<DecayedType>) {
            // For enum types, check the registry first
            std::string enum_name = EnumRegistry::getInstance().getEnumName<DecayedType>();
            return enum_name.empty() ? "unknown_enum" : enum_name;
        } else {
            // Use the enhanced type name resolution from argument_types.h
            return std::string(get_type_name<DecayedType>());
        }
    }

   public:
    Argument() {}
    template <typename T>
    T Get() const {
        std::cerr << "Type name -> " << type_name << "\n";

        // Compile-time check for parameter types
        if constexpr (std::is_same_v<T, FloatParam*> || std::is_same_v<T, IntParam*>) {
            if (is_null_ptr) {
                return nullptr;
            }
            if constexpr (std::is_same_v<T, FloatParam*>)
                return std::get<FloatParam*>(param);
            else if constexpr (std::is_same_v<T, IntParam*>)
                return std::get<IntParam*>(param);
        } else {
            if (is_null_ptr || is_parameter)
                THROW("Undefined type passed")

            if constexpr (is_vector_type<std::decay_t<T>>::value) {
                std::cerr << "Vector type\n";
                using ElementType = typename std::decay_t<T>::value_type;

                std::vector<ElementType> result;
                for (const auto& v : values) {
                    result.push_back(std::any_cast<ElementType>(v));
                    std::cerr << "Print val : " << std::any_cast<ElementType>(v) << "\n";
                }
                return result;
            } else if (!is_vector) {
                std::cerr << "Non Vector type detected - \t" << values.size() <<"\n";
                return std::any_cast<T>(values[0]);
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

    template <typename T>
    explicit inline Argument(const std::string name, T&& val)
        : arg_name(std::move(name)) {
        if constexpr (std::is_enum_v<std::decay_t<T>>) {
            type_name = "enum"; // Enum types are stored as integers by default
            
            enum_type_name = getTypeName<T>();
            if (enum_type_name != "unknown_enum") {
                values.push_back(static_cast<int>(val));
            } else {
                THROW("Unknown enum type for argument " + arg_name)
            }
        } else if constexpr (is_vector_type_v<std::decay_t<T>>) {
            using ElementType = typename std::decay_t<T>::value_type;
            std::string element_type_name = getTypeName<ElementType>();
            if (element_type_name != "unknown") {
                type_name = element_type_name;                
                is_vector = true;
                values.reserve(val.size()); // Pre-allocate for better performance
                for (auto&& v : std::forward<T>(val)) {
                    values.push_back(static_cast<ElementType>(std::forward<decltype(v)>(v)));
                }
            } else {
                THROW("Unknown vector element type for argument " + arg_name)
            }
        } else {
            type_name = getTypeName<T>();
            if (type_name != "unknown") {
                if constexpr (std::is_same_v<std::decay_t<T>, const char*>) {
                    values.push_back(std::string(val));
                } else {
                    values.push_back(static_cast<std::decay_t<T>>(std::forward<T>(val)));
                }
            } else {
                THROW("Unknown type " + std::string(typeid(T).name()) + " for argument " + arg_name)
            }
        }
    }

    // Used to store the feature key map
    explicit inline Argument(std::string name, std::map<std::string, std::string> val)
        : arg_name(std::move(name)) {
        type_name = "map_string";
        is_vector = true;
        if (!val.empty()) {
            values.reserve(val.size() * 2); // Pre-allocate for key-value pairs
            for (auto&& pair : std::move(val)) {
                values.push_back(std::move(pair.first));   // Push key
                values.push_back(std::move(pair.second));  // Push value
            }
        }
    }

    // Used to store the shared_ptr
    template <typename T>
    explicit inline Argument(std::string name, std::shared_ptr<T> val)
        : arg_name(std::move(name)) {
        type_name = "shared_ptr";

        // For MetadataReader case store an empty value
        // During deserialization the MetadataReader should be created and passed from the MasterGraph.
        if (arg_name == "meta_data_reader") {
            values.push_back(static_cast<int>(0));
        }
        // Could store additional shared_ptr metadata here if needed
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
        : arg_name(std::move(name)) {
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
        : arg_name(std::move(name)) {
        type_name = "int";
        if (param == nullptr) {
            type_name = "nullptr";
            is_null_ptr = true;
            return;
        }
        extract_param(param->type, param);
    }
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
bool init_args(NodeType* node, const std::vector<Argument>& arguments) {
    if (arguments.size() != sizeof...(Args)) return false;

    try {
        // Unpack arguments with type-check and casting
        // For C++ >= 20
        // std::tuple<Args...> unpacked_args = [&]<std::size_t... I>(std::index_sequence<I...>) {
        //     return std::make_tuple(arguments[I].Get<Args>()...);
        // }(std::index_sequence_for<Args...>{});

        auto unpacked_args = unpack_arguments<Args...>(arguments);
        std::cerr << "Arguments unpacked\t" << std::tuple_size<decltype(unpacked_args)>::value << "\n";

        std::apply([&](Args... unpacked) {
            node->init(std::forward<Args>(unpacked)...);
        }, unpacked_args);

        return true;
    } catch (const std::exception& e) {
        std::cerr << "[ERR] Exception during init_args: " << e.what() << "\n";
        return false; // Type mismatch
    }
}