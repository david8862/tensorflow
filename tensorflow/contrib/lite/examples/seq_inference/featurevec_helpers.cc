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

#include <unistd.h>  // NOLINT(build/include_order)

#include "tensorflow/contrib/lite/examples/seq_inference/featurevec_helpers.h"

#define LOG(x) std::cerr

namespace tflite {
namespace seq_inference {

std::vector<double> read_featurevec(const std::string& input_npy_name, int width,
                              int height, Settings* s)
{
    t_npy_header header;

    std::ifstream file(input_npy_name, std::ios::in | std::ios::binary);
    if (!file) {
        LOG(FATAL) << "input file " << input_npy_name << " not found\n";
        exit(-1);
    }

    // hardcode feature array size
    int seq_length = height;
    int feature_length = width;
    std::vector<double> feature_array(seq_length*feature_length);

    file.read(reinterpret_cast<char*>(&header), sizeof(t_npy_header));
    file.seekg(header.header_len, std::ios::cur);

    file.read(reinterpret_cast<char*>(feature_array.data()), seq_length*feature_length*sizeof(double));

    //int count = 0;
    //for (auto item : feature_array) {
        //if (item == 0)
            //count ++;
        //else
            ////printf("%lf\n", item);
            //cout << item << endl;
    //}
    //printf("%d\n", count);
    return feature_array;
}
}  // namespace seq_inference
}  // namespace tflite
