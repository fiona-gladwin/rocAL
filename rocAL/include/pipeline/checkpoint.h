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

#include <vector>
#include <random>
#include <any>
#include <map>

class OperatorCheckpoint {
   public:
    OperatorCheckpoint(std::string name) : _operator_name(name) {
    }
   private:
    const std::string _operator_name;
    std::any _state;
};

class Checkpoint {
   public:
    OperatorCheckpoint& AddOperatorCheckpoint(std::string op_name) {
        _name_to_id[op_name] = _op_cpts.size();
        _op_cpts.emplace_back(OperatorCheckpoint(op_name));
        return _op_cpts.back();
    }

    OperatorCheckpoint& GetOperatorCheckpoint(std::string op_name) {
        return _op_cpts[_name_to_id[op_name]];
    }
   private:
    std::vector<OperatorCheckpoint> _op_cpts;
    std::map<std::string, int, std::less<>> _name_to_id;
    // size_t _iteration_number = 0;
};
