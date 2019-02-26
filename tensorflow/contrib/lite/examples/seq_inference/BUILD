# Description:
# TensorFlow Lite Example Label Image.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_cc_binary")
load("//tensorflow/contrib/lite:build_def.bzl", "tflite_linkopts")

exports_files(glob([
    "testdata/*.bmp",
]))

tf_cc_binary(
    name = "seq_inference",
    srcs = [
        "get_top_n.h",
        "get_top_n_impl.h",
        "seq_inference.cc",
    ],
    linkopts = tflite_linkopts() + select({
        "//tensorflow:android": [
            "-pie",  # Android 5.0 and later supports only PIE
            "-lm",  # some builtin ops, e.g., tanh, need -lm
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":featurevec_helpers",
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite:string_util",
        "//tensorflow/contrib/lite/kernels:builtin_ops",
    ],
)

cc_library(
    name = "featurevec_helpers",
    srcs = ["featurevec_helpers.cc"],
    hdrs = [
        "featurevec_helpers.h",
        "featurevec_helpers_impl.h",
        "seq_inference.h",
    ],
    deps = [
        "//tensorflow/contrib/lite:builtin_op_data",
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite:schema_fbs_version",
        "//tensorflow/contrib/lite:string",
        "//tensorflow/contrib/lite:string_util",
        "//tensorflow/contrib/lite/kernels:builtin_ops",
        "//tensorflow/contrib/lite/schema:schema_fbs",
    ],
)

cc_test(
    name = "seq_inference_test",
    srcs = [
        "get_top_n.h",
        "get_top_n_impl.h",
        "seq_inference_test.cc",
    ],
    data = [
        "testdata/grace_hopper.bmp",
    ],
    tags = ["no_oss"],
    deps = [
        ":featurevec_helpers",
        "@com_google_googletest//:gtest",
    ],
)
