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

#include "pipeline/pipe_op.h"

void PipelineOperator::serialize_pipeop_inputs_and_outputs_to_protobuf(rocal_proto::OperatorDef *opdef) {
    if (this->module_name == "loader") {
        for (auto& node_output : this->node->output()) {
            rocal_proto::InputOutput *output = opdef->add_outputs();
            output->set_name(node_output->tensor_name());
            output->set_device(static_cast<int>(node_output->info().mem_type()));
            output->set_dtype(static_cast<int>(node_output->info().data_type()));
            output->set_layout(static_cast<int>(node_output->info().layout()));
            output->set_color_format(static_cast<int>(node_output->info().color_format()));
            for (auto& dim : node_output->info().dims())
                output->add_dims(dim);
            output->set_num_dims(node_output->info().num_of_dims());
        }
    } else if (this->module_name != "reader") {
        for (auto& node_input : this->node->input()) {
            rocal_proto::InputOutput *input = opdef->add_inputs();
            input->set_name(node_input->tensor_name());
            input->set_device(static_cast<int>(node_input->info().mem_type()));
            input->set_dtype(static_cast<int>(node_input->info().data_type()));
            input->set_layout(static_cast<int>(node_input->info().layout()));
            input->set_color_format(static_cast<int>(node_input->info().color_format()));
            for (auto& dim : node_input->info().dims())
                input->add_dims(dim);
            input->set_num_dims(node_input->info().num_of_dims());
        }
        for (auto& node_output : this->node->output()) {
            rocal_proto::InputOutput *output = opdef->add_outputs();
            output->set_name(node_output->tensor_name());
            output->set_device(static_cast<int>(node_output->info().mem_type()));
            output->set_dtype(static_cast<int>(node_output->info().data_type()));
            output->set_layout(static_cast<int>(node_output->info().layout()));
            output->set_color_format(static_cast<int>(node_output->info().color_format()));
            for (auto& dim : node_output->info().dims())
                output->add_dims(dim);
            output->set_num_dims(node_output->info().num_of_dims());
        }
    }
}

void serialize_parameter_to_protobuf(rocal_proto::Parameter *parameter, Argument &op_arg) {
    if (op_arg.enum_type_name == "SimpleParameter") {
        if (op_arg.type_name == "int") {
            auto param_core = std::get<Parameter<int>*>(op_arg.param_core);
            auto simple_param = dynamic_cast<SimpleParameter<int> *>(param_core);
            // Fetch and add values to the parameter class
            parameter->add_param_val_int(simple_param->get());
        } else if (op_arg.type_name == "float") {
            auto param_core = std::get<Parameter<float>*>(op_arg.param_core);
            auto simple_param = dynamic_cast<SimpleParameter<float> *>(param_core);
            parameter->add_param_val_float(simple_param->get());
        }
    } else if (op_arg.enum_type_name == "UniformRand") {
        if (op_arg.type_name == "int") {
            auto param_core = std::get<Parameter<int>*>(op_arg.param_core);
            auto uniform_param = dynamic_cast<UniformRand<int> *>(param_core);
            // Fetch and add values to the parameter class
            auto uniform_range = uniform_param->get_start_and_end();
            parameter->add_param_val_int(uniform_range.first);
            parameter->add_param_val_int(uniform_range.second);
        } else if (op_arg.type_name == "float") {
            auto param_core = std::get<Parameter<float>*>(op_arg.param_core);
            auto uniform_param = dynamic_cast<UniformRand<float> *>(param_core);
            // Fetch and add values to the parameter class
            auto uniform_range = uniform_param->get_start_and_end();
            parameter->add_param_val_float(uniform_range.first);
            parameter->add_param_val_float(uniform_range.second);
        }
    } else if (op_arg.enum_type_name == "CustomRand") {
        if (op_arg.type_name == "int") {
            auto param_core = std::get<Parameter<int>*>(op_arg.param_core);
            auto random_param = dynamic_cast<CustomRand<int> *>(param_core);
            // Fetch and add values to the parameter class
            auto values_vec = random_param->get_values();
            // Get values
            for (auto& val : values_vec) {
                parameter->add_param_val_int(val);
            }
            auto frequency_vec = random_param->get_frequencies();
            for (auto& val : frequency_vec) {
                parameter->add_frequency(val);
            }
            parameter->set_size(random_param->size());
        } else if (op_arg.type_name == "float") {
            auto param_core = std::get<Parameter<float>*>(op_arg.param_core);
            auto random_param = dynamic_cast<CustomRand<float> *>(param_core);
            // Fetch and add values to the parameter class
            auto values_vec = random_param->get_values();
            // Get values
            for (auto& val : values_vec) {
                parameter->add_param_val_float(val);
            }
            auto frequency_vec = random_param->get_frequencies();
            for (auto& val : frequency_vec) {
                parameter->add_frequency(val);
            }
            parameter->set_size(random_param->size());
        }
    }
}

void PipelineOperator::serialize_pipeop_args_to_protobuf(rocal_proto::OperatorDef *opdef) {
    
    std::vector<Argument> arguments_list;
    
    if (this->module_name == "reader") {
        arguments_list = this->arguments;
    } else {
        arguments_list = this->node->get_args_list();
    }
    // Iterate through each argument to store in the protobuffers
    for (auto& op_arg : arguments_list) {
        rocal_proto::Arguments *arg = opdef->add_args();
        arg->set_name(op_arg.arg_name);
        arg->set_type(op_arg.type_name);

        if (op_arg.type_name == "nullptr") continue; // TODOSER - During deserialize Nullptr needs to be handled
        if (op_arg.enum_type_name != "")
            arg->set_instance_name(op_arg.enum_type_name);

        if (op_arg.is_parameter) {
            rocal_proto::Parameter *param = arg->mutable_param();
            serialize_parameter_to_protobuf(param, op_arg);
        }
        // Add each value to the arg based on the type
        else if (!op_arg.is_vector && op_arg.values.size() == 1) {
            std::cerr << op_arg.arg_name << " - " << op_arg.type_name << "<<\n";
            if (op_arg.type_name == "int" || op_arg.type_name == "shared_ptr") {
                arg->add_ints(std::any_cast<int>(op_arg.values[0]));
            } else if (op_arg.type_name == "float") {
                arg->add_floats(std::any_cast<float>(op_arg.values[0]));
            } else if (op_arg.type_name == "char_str" || op_arg.type_name == "string") {
                arg->add_strings(std::any_cast<std::string>(op_arg.values[0]));
            } else if (op_arg.type_name == "bool") {
                arg->add_bools(std::any_cast<bool>(op_arg.values[0]));
            } else if (op_arg.type_name == "unsigned") {
                arg->add_uints(std::any_cast<unsigned>(op_arg.values[0])); // Use unsigned int instead of uint
            } else if (op_arg.type_name == "size_t") {
                arg->add_uints(std::any_cast<size_t>(op_arg.values[0])); // Use unsigned int instead of uint
            } 
            else {
                THROW("Invalid type specified for the Argument " + op_arg.arg_name);
            }
        } else if (op_arg.is_vector) {
            // TODO - VECTOR BASED PROCESSING
            for (auto& v : op_arg.values) {
                if (op_arg.type_name == "int" || op_arg.type_name == "shared_ptr") {
                    arg->add_ints(std::any_cast<int>(v));
                } else if (op_arg.type_name == "float") {
                    arg->add_floats(std::any_cast<float>(v));
                } else if (op_arg.type_name == "char_str" || op_arg.type_name == "string") {
                    arg->add_strings(std::any_cast<std::string>(v));
                } else if (op_arg.type_name == "bool") {
                    arg->add_bools(std::any_cast<bool>(v));
                } else if (op_arg.type_name == "unsigned") {
                    arg->add_uints(std::any_cast<unsigned>(v)); // Use unsigned int instead of uint
                } else if (op_arg.type_name == "size_t") {
                    arg->add_uints(std::any_cast<size_t>(v)); // Use unsigned int instead of uint
                } 
                else {
                    THROW("Invalid type specified for the Argument " + op_arg.arg_name);
                }    
            }
        }
    }
}