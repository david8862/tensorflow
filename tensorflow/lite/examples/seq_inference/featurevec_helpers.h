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

#ifndef TENSORFLOW_LITE_EXAMPLES_SEQ_INFERENCE_FEATUREVEC_HELPERS_H_
#define TENSORFLOW_LITE_EXAMPLES_SEQ_INFERENCE_FEATUREVEC_HELPERS_H_

#include "tensorflow/lite/examples/seq_inference/featurevec_helpers_impl.h"
#include "tensorflow/lite/examples/seq_inference/seq_inference.h"

namespace tflite {
namespace seq_inference {

using namespace std;

std::vector<double> read_featurevec(const std::string& featurevec_name, int* width,
                              int* height, Settings* s);

template <class T>
void feed(T* out, double* in, int height, int width, Settings* s);

// explicit instantiation
template void feed<uint8_t>(uint8_t*, double*, int, int, Settings*);
template void feed<float>(float*, double*, int, int, Settings*);

}  // namespace seq_inference
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXAMPLES_SEQ_INFERENCE_FEATUREVEC_HELPERS_H_
