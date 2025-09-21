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

#include "pipeline/pipeline_serializer.h"

void PipelineSerializer::serialize_to_string(std::string& serialized_string) {
    serialized_string = _pipeline.SerializeAsString();
}

void PipelineSerializer::serialize_pipeline_config(size_t num_threads, size_t batch_size, int device_id, RocalMemType device_type, size_t prefetch_queue_depth) {
    _pipeline.set_num_threads(num_threads);
    _pipeline.set_batch_size(batch_size);
    _pipeline.set_device_id(device_id);
    // _pipeline.set_seed(seed);
    _pipeline.set_rocal_cpu(device_type == RocalMemType::HOST ? true : false);
    _pipeline.set_prefetch_queue_depth(prefetch_queue_depth);
}

void PipelineSerializer::serialize_operators(std::vector<std::shared_ptr<PipelineOperator>>& operators) {
    // Serialize all operators
    for (auto &pipe_op : operators) {
        rocal_proto::OperatorDef *op = _pipeline.add_operators();
        op->set_name(pipe_op->operator_name);
        op->set_module_name(pipe_op->module_name);
        // Add support to add each argument in the operator
        pipe_op->serialize_pipeop_args_to_protobuf(op);
        pipe_op->serialize_pipeop_inputs_and_outputs_to_protobuf(op);
    }
}

void PipelineSerializer::serialize_output_tensors(TensorList& output_tensors_list) {

    // Serialize the pipeline outputs
    for (size_t idx = 0; idx < output_tensors_list.size(); idx++) {
        rocal_proto::InputOutput *output = _pipeline.add_pipe_outputs();
        auto pipe_output = output_tensors_list[idx];
        output->set_name(pipe_output->tensor_name());
        output->set_device(static_cast<int>(pipe_output->info().mem_type()));
        output->set_dtype(static_cast<int>(pipe_output->info().data_type()));
        output->set_layout(static_cast<int>(pipe_output->info().layout()));
        output->set_color_format(static_cast<int>(pipe_output->info().color_format()));
        for (auto& dim : pipe_output->info().dims())
            output->add_dims(dim);
        output->set_num_dims(pipe_output->info().num_of_dims());
        output->set_is_argument_input(false);
    }
}

RocalStatus PipelineSerializer::deserialize_args_from_protobuf(const rocal_proto::OperatorDef& opdef, std::vector<Argument>& arguments) {
    for (const auto& proto_arg : opdef.args()) {
        Argument arg;
        arg.arg_name = proto_arg.name();
        arg.type_name = proto_arg.has_type() ? proto_arg.type() : "";
        arg.enum_type_name = proto_arg.has_instance_name() ? proto_arg.instance_name() : "";
        arg.is_vector = proto_arg.is_vector();
        arg.is_parameter = proto_arg.has_param();

        // Handle parameters
        if (arg.type_name == "enum") {
            const auto& enum_val = proto_arg.enum_value();
            arg.enum_type_name = enum_val.name();
            if (EnumRegistry::getInstance().isEnumRegistered(arg.enum_type_name)) {
                // Use new std::any-based approach
                std::any enum_value = EnumRegistry::getInstance().convertIntToEnum(arg.enum_type_name, enum_val.value());
                arg.values.push_back(enum_value);
            } else {
                THROW("Invalid instance name set to the argument: " + arg.enum_type_name);
            }
        } else if (arg.is_parameter) {
            const auto& param = proto_arg.param();
            if (arg.type_name == "int") {
                if (arg.enum_type_name == "SimpleParameter") {
                    arg.param = static_cast<IntParam*>(ParameterFactory::instance()->create_single_value_int_param(param.param_val_int(0)));
                } else if (arg.enum_type_name == "UniformRand") {
                    arg.param = static_cast<IntParam*>(ParameterFactory::instance()->create_uniform_int_rand_param(param.param_val_int(0), param.param_val_int(1)));
                } else if (arg.enum_type_name == "CustomRand") {
                    std::vector<int> values(param.param_val_int().begin(), param.param_val_int().end());
                    std::vector<double> freqs(param.frequency().begin(), param.frequency().end());
                    arg.param = static_cast<IntParam*>(ParameterFactory::instance()->create_custom_int_rand_param(values.data(),
                                                                      freqs.data(),
                                                                      values.size()));
                }
            } else if (arg.type_name == "float") {
                if (arg.enum_type_name == "SimpleParameter") {
                    arg.param = static_cast<FloatParam*>(ParameterFactory::instance()->create_single_value_float_param(param.param_val_float(0)));
                } else if (arg.enum_type_name == "UniformRand") {
                    arg.param = static_cast<FloatParam*>(ParameterFactory::instance()->create_uniform_float_rand_param(param.param_val_float(0), param.param_val_float(1)));
                } else if (arg.enum_type_name == "CustomRand") {
                    std::vector<float> values(param.param_val_float().begin(), param.param_val_float().end());
                    std::vector<double> freqs(param.frequency().begin(), param.frequency().end());
                    arg.param = static_cast<FloatParam*>(ParameterFactory::instance()->create_custom_float_rand_param(values.data(),
                                                                      freqs.data(),
                                                                      values.size()));
                }
            } else {
                arg.is_null_ptr = true;
            }
        } else if (arg.is_vector) {
            // Handle vector deserialization based on type
            if (arg.type_name == "int" || arg.type_name == "unsigned" || arg.type_name == "size_t" || arg.type_name == "shared_ptr") {
                // Deserialize integer vectors - expect exactly one vector
                if (proto_arg.int_vectors_size() != 1) {
                    THROW("Expected exactly one int vector for argument " + arg.arg_name + ", but found " + std::to_string(proto_arg.int_vectors_size()));
                }
                const auto& int_vec = proto_arg.int_vectors(0);
                for (auto val : int_vec.values()) {
                    if (arg.type_name == "unsigned") {
                        arg.values.push_back(static_cast<unsigned>(val));
                    } else if (arg.type_name == "size_t") {
                        arg.values.push_back(static_cast<size_t>(val));
                    } else if (arg.type_name == "shared_ptr") {
                        arg.values.push_back(static_cast<int>(val));
                    } else { // int
                        arg.values.push_back(static_cast<int>(val));
                    }
                }
            } else if (arg.type_name == "float") {
                // Deserialize float vectors - expect exactly one vector
                if (proto_arg.float_vectors_size() != 1) {
                    THROW("Expected exactly one float vector for argument " + arg.arg_name + ", but found " + std::to_string(proto_arg.float_vectors_size()));
                }
                const auto& float_vec = proto_arg.float_vectors(0);
                for (auto val : float_vec.values()) {
                    arg.values.push_back(val);
                }
            } else if (arg.type_name == "char_str" || arg.type_name == "string" || arg.type_name == "map_string") {
                // Deserialize string vectors - expect exactly one vector
                if (proto_arg.string_vectors_size() != 1) {
                    THROW("Expected exactly one string vector for argument " + arg.arg_name + ", but found " + std::to_string(proto_arg.string_vectors_size()));
                }
                const auto& string_vec = proto_arg.string_vectors(0);
                for (const auto& val : string_vec.values()) {
                    arg.values.push_back(val);
                }
            } else {
                THROW("Vector type not supported during deserialization for Argument " + arg.arg_name + " with type " + arg.type_name);
            }
        }

        // Handle non-parameter arguments
        else if (arg.type_name == "int" || arg.type_name == "shared_ptr") {
            for (auto i : proto_arg.ints()) {
                arg.values.push_back(static_cast<int>(i));
            }
        } else if (arg.type_name == "float") {
            for (auto f : proto_arg.floats()) {
                arg.values.push_back(f);
            }
        } else if (arg.type_name == "char_str" || arg.type_name == "string") {
            for (const auto& s : proto_arg.strings()) {
                arg.values.push_back(s);
            }
        } else if (arg.type_name == "bool") {
            for (auto b : proto_arg.bools()) {
                arg.values.push_back(b);
            }
        } else if (arg.type_name == "unsigned") {
            for (auto u : proto_arg.uints()) {
                arg.values.push_back(static_cast<unsigned>(u));
            }
        } else if (arg.type_name == "size_t") {
            for (auto u : proto_arg.uints()) {
                arg.values.push_back(static_cast<size_t>(u));
            }
        } else if (arg.type_name == "nullptr") {
            arg.is_null_ptr = true;
        } else {
            THROW("Invalid or unsupported type while deserializing: " + arg.type_name);
        }

        arguments.push_back(std::move(arg));
    }
    return ROCAL_OK;
}
