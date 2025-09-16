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
#include <iostream>

#include "pipeline/enum_registry.h"
#include "parameters/parameter_factory.h"
#include "pipeline/commons.h"

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
    pParamCore param_core;
    
private:
    // Helper method to get type name from registry or built-in types
    template<typename T>
    std::string getTypeName() const {
        using DecayedType = std::decay_t<T>;
        
        // Check built-in types first
        if constexpr (std::is_same_v<DecayedType, int>) return "int";
        else if constexpr (std::is_same_v<DecayedType, unsigned>) return "unsigned";
        else if constexpr (std::is_same_v<DecayedType, size_t>) return "size_t";
        else if constexpr (std::is_same_v<DecayedType, float>) return "float";
        else if constexpr (std::is_same_v<DecayedType, double>) return "double";
        else if constexpr (std::is_same_v<DecayedType, bool>) return "bool";
        else if constexpr (std::is_same_v<DecayedType, std::string>) return "string";
        else if constexpr (std::is_same_v<DecayedType, char*> || std::is_same_v<DecayedType, const char*>) return "char_str";
        else if constexpr (std::is_enum_v<DecayedType>) {
            // For enum types, check the registry
            std::string enum_name = EnumRegistry::getInstance().getEnumName<DecayedType>();
            return enum_name.empty() ? "unknown_enum" : enum_name;
        }
        else {
            return "unknown";
        }
    }

public:

    template <typename T>
    explicit inline Argument(const std::string& name, const T&& val)
        : arg_name(name) {
        if constexpr (std::is_enum<T>::value) {
            type_name = "enum"; // Enum types are stored as integers by default
            
            enum_type_name = getTypeName<T>();
            if (enum_type_name != "unknown_enum") {
                values.push_back(static_cast<int>(val));
            } else {
                std::cout << "Type: Unknown enum " << arg_name << std::endl;
            }
        } else if constexpr (is_vector_type<T>::value) {
            using ElementType = typename std::decay_t<T>::value_type;
            std::string element_type_name = getTypeName<ElementType>();
            if (element_type_name != "unknown") {
                type_name = element_type_name;                
                is_vector = true;
                for (const auto& v : val) {
                    values.push_back(static_cast<ElementType>(v));
                }
            } else {
                std::cout << "Type: Unknown vector element type " << arg_name << std::endl;
            }
        } else {
            type_name = getTypeName<T>();
            if (type_name != "unknown") {
                if constexpr (std::is_same<std::decay_t<T>, const char *>::value) {
                    values.push_back(std::string(val));
                } else {
                    values.push_back(static_cast<std::decay_t<T>>(val));
                }
            } else {
                std::cout << "Type: Unknown " << arg_name << std::endl;
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
