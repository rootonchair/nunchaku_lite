#pragma once

#include "interop/torch.h"
#include "kernels/zgemm/zgemm.h"

namespace nunchaku_lite::ops {

inline Tensor as_tensor(std::optional<torch::Tensor> &tensor) {
    return tensor.has_value() ? from_torch(tensor.value()) : Tensor{};
}

inline void gemm_w4a4(std::optional<torch::Tensor> act,
                      std::optional<torch::Tensor> wgt,
                      std::optional<torch::Tensor> out,
                      std::optional<torch::Tensor> qout,
                      std::optional<torch::Tensor> ascales,
                      std::optional<torch::Tensor> wscales,
                      std::optional<torch::Tensor> oscales,
                      std::optional<torch::Tensor> poolout,
                      std::optional<torch::Tensor> lora_act_in,
                      std::optional<torch::Tensor> lora_up,
                      std::optional<torch::Tensor> lora_down,
                      std::optional<torch::Tensor> lora_act_out,
                      std::optional<torch::Tensor> norm_q,
                      std::optional<torch::Tensor> norm_k,
                      std::optional<torch::Tensor> rotary_emb,
                      std::optional<torch::Tensor> bias,
                      std::optional<torch::Tensor> smooth_factor,
                      std::optional<torch::Tensor> out_vk,
                      std::optional<torch::Tensor> out_linearattn,
                      bool act_unsigned,
                      std::vector<float> lora_scales,
                      bool fuse_silu,
                      bool fp4,
                      float alpha,
                      std::optional<torch::Tensor> wcscales,
                      std::optional<torch::Tensor> out_q,
                      std::optional<torch::Tensor> out_k,
                      std::optional<torch::Tensor> out_v,
                      int attn_tokens) {
    TorchOpContext ctx;
    nunchaku::kernels::gemm_w4a4(as_tensor(act),
                                 as_tensor(wgt),
                                 as_tensor(out),
                                 as_tensor(qout),
                                 as_tensor(ascales),
                                 as_tensor(wscales),
                                 as_tensor(oscales),
                                 as_tensor(poolout),
                                 as_tensor(lora_act_in),
                                 as_tensor(lora_up),
                                 as_tensor(lora_down),
                                 as_tensor(lora_act_out),
                                 as_tensor(norm_q),
                                 as_tensor(norm_k),
                                 as_tensor(rotary_emb),
                                 as_tensor(bias),
                                 as_tensor(smooth_factor),
                                 as_tensor(out_vk),
                                 as_tensor(out_linearattn),
                                 act_unsigned,
                                 lora_scales,
                                 fuse_silu,
                                 fp4,
                                 alpha,
                                 as_tensor(wcscales),
                                 as_tensor(out_q),
                                 as_tensor(out_k),
                                 as_tensor(out_v),
                                 attn_tokens);
}

inline void quantize_w4a4_act_fuse_lora(std::optional<torch::Tensor> input,
                                        std::optional<torch::Tensor> output,
                                        std::optional<torch::Tensor> oscales,
                                        std::optional<torch::Tensor> lora_down,
                                        std::optional<torch::Tensor> lora_act_out,
                                        std::optional<torch::Tensor> smooth,
                                        bool fuse_glu,
                                        bool fp4) {
    TorchOpContext ctx;
    nunchaku::kernels::quantize_w4a4_act_fuse_lora(as_tensor(input),
                                                   as_tensor(output),
                                                   as_tensor(oscales),
                                                   as_tensor(lora_down),
                                                   as_tensor(lora_act_out),
                                                   as_tensor(smooth),
                                                   fuse_glu,
                                                   fp4);
}

} // namespace nunchaku_lite::ops
