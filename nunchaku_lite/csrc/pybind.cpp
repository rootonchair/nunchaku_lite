#include "ops_lite.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def_submodule("ops")
        .def("gemm_w4a4", nunchaku_lite::ops::gemm_w4a4)
        .def("quantize_w4a4_act_fuse_lora", nunchaku_lite::ops::quantize_w4a4_act_fuse_lora)
        .def("gemv_awq", nunchaku_lite::ops::gemv_awq)
        .def("attention_fp16", nunchaku_lite::ops::attention_fp16);
}
