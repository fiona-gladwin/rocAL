/*
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

#pragma once

#include <any>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeindex>
#include <unordered_map>
#include <vector>

#include "pipeline/argument_types.h"
#include "pipeline/commons.h"
#include "pipeline/enum_registry.h"
#include "parameters/parameter_factory.h"

/**
 * @brief Argument class stores the details of each argument in the Node
 * 
 * This class encapsulates argument information for pipeline nodes, supporting
 * various data types including basic types, enums, vectors, maps, and parameters.
 * 
 * The class provides type-safe storage and retrieval of arguments with support for:
 * - Basic types (int, float, string, etc.)
 * - Enum types with registry lookup
 * - Vector containers
 * - String-to-string maps
 * - Shared pointers
 * - Parameter objects (FloatParam, IntParam)
 */
class Argument {
public:
    // Public member variables
    std::string arg_name;                 ///< Name of the argument
    std::string type_name;                ///< Denotes the data type of the argument
    std::string enum_type_name;           ///< Denotes the name of the enum <arg_name_enum>
    bool is_vector = false;               ///< True if the argument contains vector data
    bool is_parameter = false;            ///< True if the argument is a parameter object
    bool is_null_ptr = false;             ///< True if the argument represents a null pointer
    std::vector<std::any> values;         ///< Storage for argument values (can change to std::variant later)
    pParam param;                         ///< Parameter stored for parameter-type arguments
    std::string tensor_name;              ///< Name of the tensor for tensor reference arguments

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
        } else if constexpr (std::is_pointer_v<T> && std::is_same_v<std::remove_pointer_t<T>, class Tensor>) {
            // Handle Tensor* retrieval
            if (is_tensor) {
                if (is_null_ptr) {
                    return nullptr;
                }
                // For tensor arguments, the actual pointer resolution should be done externally
                // This method should not be called directly for tensors during deserialization
                // Instead, use GetTensorName() to get the tensor name and resolve it externally
                if (!values.empty()) {
                    return static_cast<T>(std::any_cast<void*>(values[0]));
                }
                THROW("Tensor argument has no stored pointer")
            }
            THROW("Attempting to get Tensor* from non-tensor argument")
        } else {
            if (is_null_ptr || is_parameter || is_tensor)
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

    // Method to get tensor name for external resolution
    std::string GetTensorName() const {
        if (!is_tensor) {
            THROW("Argument is not a tensor type")
        }
        return tensor_name;
    }

    template<>
    std::map<std::string, std::string> Get<std::map<std::string, std::string>>() const {
        std::map<std::string, std::string> feature_map;
        for (int i = 0; i < values.size(); i+=2) {
            feature_map[std::any_cast<std::string>(values[i])] = std::any_cast<std::string>(values[i + 1]);
        }
        return feature_map;
    }

    // Constructors

    /**
     * @brief Unified template constructor for all data types
     * @tparam T The type of the value being stored
     * @param name The name of the argument
     * @param val The value to store
     * @throws std::runtime_error if the type is unknown or unsupported
     */
    Argument() {}

    template <typename T>
    explicit Argument(std::string name, T&& val) : arg_name(std::move(name)) {

        if constexpr (std::is_enum_v<std::decay_t<T>>) {
            constructFromEnum(std::forward<T>(val));
        } else if constexpr (is_vector_type_v<std::decay_t<T>>) {
            constructFromVector(std::forward<T>(val));
        } else if constexpr (is_shared_ptr_v<std::decay_t<T>>) {
            constructFromSharedPtr(std::forward<T>(val));
        } else if constexpr (is_string_map_v<std::decay_t<T>>) {
            constructFromMap(std::forward<T>(val));
        } else if constexpr (is_param_type_v<std::decay_t<T>>) {
            constructFromParam(std::forward<T>(val));
        } else {
            constructFromBasicType(std::forward<T>(val));
        }
    }


private:
    /**
     * @brief Helper method to get type name from registry or built-in types
     * @tparam T The type to get the name for
     * @return The string representation of the type name
     */
    template<typename T>
    std::string getTypeName() const {
        using DecayedType = std::decay_t<T>;
        
        if constexpr (std::is_enum_v<DecayedType>) {
            // For enum types, check the registry first
            std::string enum_name = EnumRegistry::getInstance().getEnumName<DecayedType>();
            return enum_name.empty() ? "unknown_enum" : enum_name;
        } else {
            // Use the type name resolution from argument_types.h
            return std::string(get_type_name<DecayedType>());
        }
    }

    /**
     * @brief Constructs argument from enum type
     * @tparam T The enum type
     * @param val The enum value
     */
    template<typename T>
    void constructFromEnum(T&& val) {
        static_assert(std::is_enum_v<std::decay_t<T>>, "T must be an enum type");
        
        type_name = "enum";
        enum_type_name = getTypeName<T>();
        
        if (enum_type_name != "unknown_enum") {
            values.push_back(static_cast<int>(val));
        } else {
            THROW("Unknown enum type for argument " + arg_name);
        }
    }

    /**
     * @brief Constructs argument from vector type
     * @tparam T The vector type
     * @param val The vector value
     */
    template<typename T>
    void constructFromVector(T&& val) {
        static_assert(is_vector_type_v<std::decay_t<T>>, "T must be a vector type");
        
        using ElementType = typename std::decay_t<T>::value_type;
        const std::string element_type_name = getTypeName<ElementType>();
        
        if (element_type_name != "unknown") {
            type_name = element_type_name;
            is_vector = true;
            values.reserve(val.size());
            
            for (auto&& v : std::forward<T>(val)) {
                values.push_back(static_cast<ElementType>(std::forward<decltype(v)>(v)));
            }
        } else {
            THROW("Unknown vector element type for argument " + arg_name);
        }
    }

    /**
     * @brief Constructs argument from basic type
     * @tparam T The basic type
     * @param val The value
     */
    template<typename T>
    void constructFromBasicType(T&& val) {
        type_name = getTypeName<T>();
        
        if (type_name != "unknown") {
            if constexpr (std::is_same_v<std::decay_t<T>, const char*>) {
                values.push_back(std::string(val));
            } else {
                values.push_back(static_cast<std::decay_t<T>>(std::forward<T>(val)));
            }
        } else {
            THROW("Unknown type " + std::string(typeid(T).name()) + " for argument " + arg_name);
        }
    }

    /**
     * @brief Constructs argument from shared pointer type
     * @tparam T The shared_ptr type
     * @param val The shared_ptr value
     */
    template<typename T>
    void constructFromSharedPtr(T&& val) {
        static_assert(is_shared_ptr_v<std::decay_t<T>>, "T must be a shared_ptr type");
        
        type_name = "shared_ptr";
        // For MetadataReader case store an empty value
        // During deserialization the MetadataReader should be created and passed from the MasterGraph.
        if (arg_name == "meta_data_reader") {
            values.push_back(static_cast<int>(0));
        } else {
            THROW("Unsupported shared_ptr type for argument " + arg_name);
        }
    }

    /**
     * @brief Constructs argument from string-to-string map type
     * @tparam T The map type
     * @param val The map value
     */
    template<typename T>
    void constructFromMap(T&& val) {
        static_assert(is_string_map_v<std::decay_t<T>>, "T must be a string-to-string map type");
        
        type_name = "map_string";
        is_vector = true;
        
        if (!val.empty()) {
            values.reserve(val.size() * 2); // Pre-allocate for key-value pairs
            for (auto&& pair : std::forward<T>(val)) {
                values.push_back(std::move(pair.first));   // Push key
                values.push_back(std::move(pair.second));  // Push value
            }
        }
    }

    /**
     * @brief Constructs argument from parameter type (FloatParam* or IntParam*)
     * @tparam T The parameter pointer type
     * @param val The parameter pointer value
     */
    template<typename T>
    void constructFromParam(T&& val) {
        static_assert(is_param_type_v<std::decay_t<T>>, "T must be a parameter pointer type");
        
        using DecayedType = std::decay_t<T>;
        
        if constexpr (std::is_same_v<DecayedType, FloatParam*>) {
            type_name = "float";
        } else if constexpr (std::is_same_v<DecayedType, IntParam*>) {
            type_name = "int";
        }
        
        if (val == nullptr) {
            is_null_ptr = true;
            type_name = "nullptr";
            return;
        }
        
        extractParam(val->type, val);
    }

    /**
     * @brief Deduces the type of parameter of the argument
     * @param param_type The type of the parameter
     * @param parameter The parameter object
     */
    void extractParam(RocalParameterType param_type, pParam parameter) {
        switch (param_type) {
            case RocalParameterType::DETERMINISTIC:
                enum_type_name = "SimpleParameter";
                break;
            case RocalParameterType::RANDOM_UNIFORM:
                enum_type_name = "UniformRand";
                break;
            case RocalParameterType::RANDOM_CUSTOM:
                enum_type_name = "CustomRand";
                break;
            default:
                THROW("Unknown parameter type for argument " + arg_name);
        }
        param = parameter;
        is_parameter = true;
    }

    // Constructor for FloatParam arguments
    // Constructor for Tensor* arguments - stores tensor name for later resolution
    // template<typename T>
    // explicit inline Argument(std::string name, T* tensor_ptr, 
    //                        typename std::enable_if_t<std::is_same_v<T, class Tensor>>* = nullptr)

    explicit inline Argument(std::string name, Tensor* tensor_ptr)
        : arg_name(std::move(name)) {
        type_name = "tensor";
        is_tensor = true;
        if (tensor_ptr == nullptr) {
            is_null_ptr = true;
            type_name = "nullptr";
            return;
        }
        // Store the tensor name for later resolution during deserialization
        // The actual tensor pointer will be resolved from MasterGraph's _pipeline_tensors map
        tensor_name = tensor_ptr->tensor_name(); // This will be set during serialization with the actual tensor name
        values.push_back(static_cast<Tensor*>(tensor_ptr)); // Store the pointer temporarily
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

        std::apply([&](Args&... unpacked) {
            node->init(unpacked...);
        }, unpacked_args);

        return true;
    } catch (const std::exception& e) {
        std::cerr << "[ERR] Exception during init_args: " << e.what() << "\n";
        return false; // Type mismatch
    }
}
