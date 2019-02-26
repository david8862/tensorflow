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

#ifndef TENSORFLOW_CONTRIB_LITE_EXAMPLES_SEQ_INFERENCE_FEATUREVEC_HELPERS_IMPL_H_
#define TENSORFLOW_CONTRIB_LITE_EXAMPLES_SEQ_INFERENCE_FEATUREVEC_HELPERS_IMPL_H_

#include "tensorflow/contrib/lite/examples/seq_inference/seq_inference.h"
#include "tensorflow/contrib/lite/examples/seq_inference/featurevec_helpers.h"

#include "tensorflow/contrib/lite/builtin_op_data.h"
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/string_util.h"
#include "tensorflow/contrib/lite/version.h"

#define LOG(x) std::cerr

namespace tflite {
namespace seq_inference {

inline double get_mean(double *x, int len)
{
    double sum = 0;
    for (int i = 0; i < len; i++)
        sum += x[i];
    return sum/len;
}


inline double get_stddev(double *x, int len)
{
    double mean = get_mean(x, len);
    double sum = 0;
    for (int i = 0; i < len; i++)
        sum += pow(x[i] - mean, 2);
    return sqrt(sum/(len-1));
}


template <class T>
void feed(T* out, double* in, int height, int width, Settings* s) {
  auto number_of_input = height * width;
  double mean_val = get_mean(in, number_of_input);
  double stddev_val = get_stddev(in, number_of_input);
  double temp = 0;
  LOG(FATAL) << "xiaobizh --- s->input_floating: " << s->input_floating << "\n";

  for (int i = 0; i < number_of_input; i++) {
    if (s->input_floating) {
      out[i] = static_cast<float>(in[i]);
      //if(i<100 && in[i] != 0) {
          //LOG(FATAL) << "xiaobizh --- in[" << i << "]: " << in[i] << "\n";
          //LOG(FATAL) << "xiaobizh --- out[" << i << "]: " << out[i] << "\n";
      //}
    }
    else {
      temp = (in[i] - mean_val) / stddev_val;
      out[i] = static_cast<uint8_t>((temp + 1.0)*127.5);
    }
  }

}

}  // namespace seq_inference
}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_EXAMPLES_SEQ_INFERENCE_FEATUREVEC_HELPERS_IMPL_H_
