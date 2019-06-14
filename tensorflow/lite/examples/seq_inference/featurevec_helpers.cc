/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include <unistd.h>  // NOLINT(build/include_order)

#include "tensorflow/lite/examples/seq_inference/featurevec_helpers.h"

#define LOG(x) std::cerr

namespace tflite {
namespace seq_inference {

std::vector<double> read_featurevec(const std::string& featurevec_name, int* width,
                              int* height, Settings* s)
{
    string buff;
    int i = 0;

    std::ifstream file(featurevec_name, std::ios::in);
    if (!file) {
        LOG(FATAL) << "input file " << featurevec_name << " not found\n";
        exit(-1);
    }

    //get sequence length and feature length from 1st two lines
    getline(file, buff);
    *height = atoi(buff.c_str());
    getline(file, buff);
    *width = atoi(buff.c_str());

    std::vector<double> feature_array((*height)*(*width));

    while(getline(file, buff)) {
        feature_array[i] = atof(buff.c_str());
        i++;
        if(i >= (*height)*(*width)) {
            std::cout << "buffer oversize!" << "\n";
            break;
        }
    }
    return feature_array;
}

}  // namespace seq_inference
}  // namespace tflite
