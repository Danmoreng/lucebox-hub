#include "safetensors.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

constexpr int HIDDEN = 1024;
constexpr int INTER = 3584;
constexpr int FA_Q_HEADS = 8;
constexpr int FA_KV_HEADS = 2;
constexpr int FA_HEAD_DIM = 256;
constexpr int FA_Q_SIZE = FA_Q_HEADS * FA_HEAD_DIM;
constexpr int FA_QPROJ_SIZE = FA_Q_SIZE * 2;
constexpr int FA_KV_SIZE = FA_KV_HEADS * FA_HEAD_DIM;
constexpr int DN_HEADS = 16;
constexpr int DN_KEY = 128;
constexpr int DN_VAL = 128;
constexpr int DN_CONV_K = 4;
constexpr int DN_QK_SIZE = DN_HEADS * DN_KEY;
constexpr int DN_V_SIZE = DN_HEADS * DN_VAL;
constexpr int DN_CONV_CH = DN_QK_SIZE * 2 + DN_V_SIZE;
constexpr int NUM_LAYERS = 24;
constexpr int LAYER_TYPE[NUM_LAYERS] = {0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1};

struct PFLayerWeights {
  int layer_type;
  int pad[3];
  void * ptrs[14];
};

extern "C" void launch_prefill_bf16(
  const int * token_ids, int seq_len, int * output_token,
  const __nv_bfloat16 * embed_weight, const PFLayerWeights * layers,
  const __nv_bfloat16 * final_norm_w, const __nv_bfloat16 * lm_head_w,
  __nv_bfloat16 * fa_k_cache, __nv_bfloat16 * fa_v_cache,
  float * dn_states, float * conv_bufs,
  __nv_bfloat16 * hidden, __nv_bfloat16 * residual, __nv_bfloat16 * normalized,
  __nv_bfloat16 * proj_buf, __nv_bfloat16 * proj_buf2,
  __nv_bfloat16 * attn_buf, __nv_bfloat16 * mlp_buf,
  __nv_bfloat16 * dn_out_buf,
  float * beta_buf, float * alpha_buf,
#if MEGAKERNEL_PREFILL_V2
  float * dn_pre_qkv,
  float * dn_u_scratch, float * dn_w_scratch, float * dn_cs_scratch,
  const __nv_bfloat16 * fused_fa_qkv_base,
  const __nv_bfloat16 * fused_gate_up_base,
#endif
  __nv_bfloat16 * final_normed, __nv_bfloat16 * hidden_bf16_out,
  float * lm_bmv, int * lm_bmi,
  int max_seq_len,
  cudaStream_t stream);

namespace {

struct Options {
  std::string model_dir = "models/qwen3.5-0.8b";
  std::string tensor_prefix = "model.language_model.";
  int prompt_tokens = 520;
  int max_seq_len = 2048;
  int warmup = 5;
  int runs = 20;
};

struct DeviceTensor {
  void * ptr = nullptr;
  std::size_t bytes = 0;
};

void check_cuda(cudaError_t status, const char * what) {
  if (status != cudaSuccess) {
    std::cerr << what << " failed: " << cudaGetErrorString(status) << "\n";
    std::exit(1);
  }
}

bool parse_args(int argc, char ** argv, Options & options) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto require_value = [&](const char * name) -> const char * {
      if (i + 1 >= argc) {
        std::cerr << name << " requires a value\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (arg == "--hf-model-dir") {
      options.model_dir = require_value("--hf-model-dir");
    } else if (arg == "--tensor-prefix") {
      options.tensor_prefix = require_value("--tensor-prefix");
    } else if (arg == "--prompt-tokens") {
      options.prompt_tokens = std::stoi(require_value("--prompt-tokens"));
    } else if (arg == "--max-seq-len") {
      options.max_seq_len = std::stoi(require_value("--max-seq-len"));
    } else if (arg == "--warmup") {
      options.warmup = std::stoi(require_value("--warmup"));
    } else if (arg == "--runs") {
      options.runs = std::stoi(require_value("--runs"));
    } else if (arg == "--help") {
      std::cout << "usage: bench_prefill_native --hf-model-dir <dir> [--tensor-prefix model.language_model.] [--prompt-tokens 520] [--warmup 5] [--runs 20]\n";
      return false;
    } else {
      std::cerr << "unknown argument: " << arg << "\n";
      std::exit(2);
    }
  }
  if (options.prompt_tokens <= 0 || options.max_seq_len < options.prompt_tokens || options.warmup < 0 || options.runs <= 0) {
    std::cerr << "invalid benchmark options\n";
    std::exit(2);
  }
  return true;
}

std::vector<std::uint16_t> load_bf16(const std::string & model_dir, const std::string & name) {
  std::string file;
  std::string error;
  qwen35x::SafetensorTensorInfo info;
  if (!qwen35x::SafetensorLoader::resolve_tensor_file(model_dir, name, file, error) ||
      !qwen35x::SafetensorLoader::load_tensor_info(file, name, info, error)) {
    std::cerr << "tensor info failed for " << name << ": " << error << "\n";
    std::exit(3);
  }
  std::vector<std::uint16_t> data;
  if (info.dtype == "BF16") {
    if (!qwen35x::SafetensorLoader::read_bf16_tensor(file, info, data, error)) {
      std::cerr << "tensor load failed for " << name << ": " << error << "\n";
      std::exit(3);
    }
  } else {
    qwen35x::SafetensorTensorF32 tensor;
    if (!qwen35x::SafetensorLoader::read_tensor_f32(model_dir, name, tensor, error)) {
      std::cerr << "tensor f32 load failed for " << name << ": " << error << "\n";
      std::exit(3);
    }
    data.resize(tensor.data.size());
    for (std::size_t i = 0; i < tensor.data.size(); ++i) {
      std::uint32_t bits = 0;
      std::memcpy(&bits, &tensor.data[i], sizeof(float));
      const std::uint32_t lsb = (bits >> 16) & 1u;
      bits += 0x7FFFu + lsb;
      data[i] = static_cast<std::uint16_t>(bits >> 16);
    }
  }
  return data;
}

DeviceTensor upload_raw(const void * host, std::size_t bytes) {
  DeviceTensor t;
  t.bytes = bytes;
  check_cuda(cudaMalloc(&t.ptr, bytes), "cudaMalloc");
  check_cuda(cudaMemcpy(t.ptr, host, bytes, cudaMemcpyHostToDevice), "cudaMemcpy(H2D)");
  return t;
}

DeviceTensor upload_bf16(const std::vector<std::uint16_t> & data) {
  return upload_raw(data.data(), data.size() * sizeof(std::uint16_t));
}

DeviceTensor alloc_bytes(std::size_t bytes) {
  DeviceTensor t;
  t.bytes = bytes;
  check_cuda(cudaMalloc(&t.ptr, bytes), "cudaMalloc");
  return t;
}

void free_tensor(DeviceTensor & t) {
  if (t.ptr != nullptr) {
    cudaFree(t.ptr);
    t.ptr = nullptr;
    t.bytes = 0;
  }
}

std::vector<std::uint16_t> concat_rows(
  const std::vector<std::uint16_t> & a,
  const std::vector<std::uint16_t> & b) {
  std::vector<std::uint16_t> out;
  out.reserve(a.size() + b.size());
  out.insert(out.end(), a.begin(), a.end());
  out.insert(out.end(), b.begin(), b.end());
  return out;
}

std::vector<std::uint16_t> concat_rows3(
  const std::vector<std::uint16_t> & a,
  const std::vector<std::uint16_t> & b,
  const std::vector<std::uint16_t> & c) {
  std::vector<std::uint16_t> out;
  out.reserve(a.size() + b.size() + c.size());
  out.insert(out.end(), a.begin(), a.end());
  out.insert(out.end(), b.begin(), b.end());
  out.insert(out.end(), c.begin(), c.end());
  return out;
}

std::string layer_name(const Options & options, int layer, const char * suffix) {
  return options.tensor_prefix + "layers." + std::to_string(layer) + "." + suffix;
}

} // namespace

int main(int argc, char ** argv) {
  Options options;
  if (!parse_args(argc, argv, options)) {
    return 0;
  }

  std::cout << "Loading BF16 weights from " << options.model_dir << "\n";
  const auto load_start = std::chrono::steady_clock::now();

  std::vector<DeviceTensor> owned;
  auto own = [&](DeviceTensor t) -> void * {
    owned.push_back(t);
    return owned.back().ptr;
  };
  auto load_device = [&](const std::string & name) -> void * {
    return own(upload_bf16(load_bf16(options.model_dir, name)));
  };

  void * embed = load_device(options.tensor_prefix + "embed_tokens.weight");
  void * final_norm = load_device(options.tensor_prefix + "norm.weight");
  void * lm_head = embed;

  std::vector<PFLayerWeights> host_layers(NUM_LAYERS);
  std::vector<std::vector<std::uint16_t>> fused_fa_qkv_host;
  std::vector<std::vector<std::uint16_t>> fused_gate_up_host;
  fused_fa_qkv_host.reserve(6);
  fused_gate_up_host.reserve(NUM_LAYERS);

  for (int li = 0; li < NUM_LAYERS; ++li) {
    PFLayerWeights & lw = host_layers[li];
    lw.layer_type = LAYER_TYPE[li];
    lw.ptrs[0] = load_device(layer_name(options, li, "input_layernorm.weight"));
    if (LAYER_TYPE[li] == 0) {
      lw.ptrs[1] = load_device(layer_name(options, li, "linear_attn.in_proj_qkv.weight"));
      lw.ptrs[2] = load_device(layer_name(options, li, "linear_attn.in_proj_z.weight"));
      lw.ptrs[3] = load_device(layer_name(options, li, "linear_attn.in_proj_b.weight"));
      lw.ptrs[4] = load_device(layer_name(options, li, "linear_attn.in_proj_a.weight"));
      lw.ptrs[5] = load_device(layer_name(options, li, "linear_attn.conv1d.weight"));
      lw.ptrs[6] = load_device(layer_name(options, li, "linear_attn.A_log"));
      lw.ptrs[7] = load_device(layer_name(options, li, "linear_attn.dt_bias"));
      lw.ptrs[8] = load_device(layer_name(options, li, "linear_attn.norm.weight"));
      lw.ptrs[9] = load_device(layer_name(options, li, "linear_attn.out_proj.weight"));
      lw.ptrs[10] = load_device(layer_name(options, li, "post_attention_layernorm.weight"));
      auto gate = load_bf16(options.model_dir, layer_name(options, li, "mlp.gate_proj.weight"));
      auto up = load_bf16(options.model_dir, layer_name(options, li, "mlp.up_proj.weight"));
      lw.ptrs[11] = own(upload_bf16(gate));
      lw.ptrs[12] = own(upload_bf16(up));
      lw.ptrs[13] = load_device(layer_name(options, li, "mlp.down_proj.weight"));
      fused_gate_up_host.push_back(concat_rows(gate, up));
    } else {
      auto q = load_bf16(options.model_dir, layer_name(options, li, "self_attn.q_proj.weight"));
      auto k = load_bf16(options.model_dir, layer_name(options, li, "self_attn.k_proj.weight"));
      auto v = load_bf16(options.model_dir, layer_name(options, li, "self_attn.v_proj.weight"));
      lw.ptrs[1] = own(upload_bf16(q));
      lw.ptrs[2] = own(upload_bf16(k));
      lw.ptrs[3] = own(upload_bf16(v));
      lw.ptrs[4] = load_device(layer_name(options, li, "self_attn.q_norm.weight"));
      lw.ptrs[5] = load_device(layer_name(options, li, "self_attn.k_norm.weight"));
      lw.ptrs[6] = load_device(layer_name(options, li, "self_attn.o_proj.weight"));
      lw.ptrs[7] = load_device(layer_name(options, li, "post_attention_layernorm.weight"));
      auto gate = load_bf16(options.model_dir, layer_name(options, li, "mlp.gate_proj.weight"));
      auto up = load_bf16(options.model_dir, layer_name(options, li, "mlp.up_proj.weight"));
      lw.ptrs[8] = own(upload_bf16(gate));
      lw.ptrs[9] = own(upload_bf16(up));
      lw.ptrs[10] = load_device(layer_name(options, li, "mlp.down_proj.weight"));
      fused_fa_qkv_host.push_back(concat_rows3(q, k, v));
      fused_gate_up_host.push_back(concat_rows(gate, up));
    }
  }

  DeviceTensor layers_dev = upload_raw(host_layers.data(), host_layers.size() * sizeof(PFLayerWeights));

  std::vector<std::uint16_t> fused_fa_qkv;
  for (const auto & item : fused_fa_qkv_host) fused_fa_qkv.insert(fused_fa_qkv.end(), item.begin(), item.end());
  std::vector<std::uint16_t> fused_gate_up;
  for (const auto & item : fused_gate_up_host) fused_gate_up.insert(fused_gate_up.end(), item.begin(), item.end());
  DeviceTensor fused_fa_qkv_dev = upload_bf16(fused_fa_qkv);
  DeviceTensor fused_gate_up_dev = upload_bf16(fused_gate_up);

  const auto load_end = std::chrono::steady_clock::now();
  const double load_ms = std::chrono::duration<double, std::milli>(load_end - load_start).count();
  std::cout << "Load/upload ms: " << load_ms << "\n";

  const int S = options.prompt_tokens;
  const int S_pad = ((S + 7) / 8) * 8;
  std::vector<int> token_ids(S);
  for (int i = 0; i < S; ++i) token_ids[i] = 1000 + (i % 2048);
  DeviceTensor token_ids_dev = upload_raw(token_ids.data(), token_ids.size() * sizeof(int));

  const int n_fa = 6;
  const int n_dn = 18;
  const int mx = std::max({DN_CONV_CH, FA_QPROJ_SIZE + 2 * FA_KV_SIZE, INTER * 2});
  DeviceTensor output_token = alloc_bytes(sizeof(int));
  DeviceTensor fa_k = alloc_bytes(static_cast<std::size_t>(n_fa) * FA_KV_HEADS * options.max_seq_len * FA_HEAD_DIM * 2);
  DeviceTensor fa_v = alloc_bytes(static_cast<std::size_t>(n_fa) * FA_KV_HEADS * options.max_seq_len * FA_HEAD_DIM * 2);
  DeviceTensor dn_states = alloc_bytes(static_cast<std::size_t>(n_dn) * DN_HEADS * DN_KEY * DN_VAL * sizeof(float));
  DeviceTensor conv_bufs = alloc_bytes(static_cast<std::size_t>(n_dn) * DN_CONV_CH * DN_CONV_K * sizeof(float));
  DeviceTensor hidden = alloc_bytes(static_cast<std::size_t>(S) * HIDDEN * 2);
  DeviceTensor residual = alloc_bytes(static_cast<std::size_t>(S) * HIDDEN * 2);
  DeviceTensor normalized = alloc_bytes(static_cast<std::size_t>(S) * HIDDEN * 2);
  DeviceTensor proj_buf = alloc_bytes(static_cast<std::size_t>(S) * mx * 2);
  DeviceTensor proj_buf2 = alloc_bytes(static_cast<std::size_t>(S) * mx * 2);
  DeviceTensor attn_buf = alloc_bytes(static_cast<std::size_t>(S) * std::max(FA_Q_SIZE, FA_KV_SIZE) * 2);
  DeviceTensor mlp_buf = alloc_bytes(static_cast<std::size_t>(S) * INTER * 2);
  DeviceTensor dn_out_buf = alloc_bytes(static_cast<std::size_t>(S) * DN_V_SIZE * 2);
  DeviceTensor beta_buf = alloc_bytes(static_cast<std::size_t>(S) * DN_HEADS * sizeof(float));
  DeviceTensor alpha_buf = alloc_bytes(static_cast<std::size_t>(S) * DN_HEADS * sizeof(float));
  DeviceTensor dn_pre_qkv = alloc_bytes(static_cast<std::size_t>(S) * DN_CONV_CH * sizeof(float));
  DeviceTensor dn_u = alloc_bytes(static_cast<std::size_t>(S_pad) * DN_HEADS * 128 * sizeof(float));
  DeviceTensor dn_w = alloc_bytes(static_cast<std::size_t>(S_pad) * DN_HEADS * 128 * sizeof(float));
  DeviceTensor dn_cs = alloc_bytes(static_cast<std::size_t>(S_pad) * DN_HEADS * sizeof(float));
  DeviceTensor final_normed = alloc_bytes(HIDDEN * 2);
  DeviceTensor hidden_bf16_out = alloc_bytes(HIDDEN * 2);
  DeviceTensor lm_bmv = alloc_bytes(1024 * sizeof(float));
  DeviceTensor lm_bmi = alloc_bytes(1024 * sizeof(int));

  cudaStream_t stream = nullptr;
  check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");

  auto reset_state = [&]() {
    check_cuda(cudaMemsetAsync(fa_k.ptr, 0, fa_k.bytes, stream), "cudaMemset(fa_k)");
    check_cuda(cudaMemsetAsync(fa_v.ptr, 0, fa_v.bytes, stream), "cudaMemset(fa_v)");
    check_cuda(cudaMemsetAsync(dn_states.ptr, 0, dn_states.bytes, stream), "cudaMemset(dn_states)");
    check_cuda(cudaMemsetAsync(conv_bufs.ptr, 0, conv_bufs.bytes, stream), "cudaMemset(conv_bufs)");
  };

  auto run_prefill = [&]() {
    reset_state();
    launch_prefill_bf16(
      static_cast<const int *>(token_ids_dev.ptr), S, static_cast<int *>(output_token.ptr),
      static_cast<const __nv_bfloat16 *>(embed), static_cast<const PFLayerWeights *>(layers_dev.ptr),
      static_cast<const __nv_bfloat16 *>(final_norm), static_cast<const __nv_bfloat16 *>(lm_head),
      static_cast<__nv_bfloat16 *>(fa_k.ptr), static_cast<__nv_bfloat16 *>(fa_v.ptr),
      static_cast<float *>(dn_states.ptr), static_cast<float *>(conv_bufs.ptr),
      static_cast<__nv_bfloat16 *>(hidden.ptr), static_cast<__nv_bfloat16 *>(residual.ptr), static_cast<__nv_bfloat16 *>(normalized.ptr),
      static_cast<__nv_bfloat16 *>(proj_buf.ptr), static_cast<__nv_bfloat16 *>(proj_buf2.ptr),
      static_cast<__nv_bfloat16 *>(attn_buf.ptr), static_cast<__nv_bfloat16 *>(mlp_buf.ptr),
      static_cast<__nv_bfloat16 *>(dn_out_buf.ptr),
      static_cast<float *>(beta_buf.ptr), static_cast<float *>(alpha_buf.ptr),
#if MEGAKERNEL_PREFILL_V2
      static_cast<float *>(dn_pre_qkv.ptr),
      static_cast<float *>(dn_u.ptr), static_cast<float *>(dn_w.ptr), static_cast<float *>(dn_cs.ptr),
      static_cast<const __nv_bfloat16 *>(fused_fa_qkv_dev.ptr),
      static_cast<const __nv_bfloat16 *>(fused_gate_up_dev.ptr),
#endif
      static_cast<__nv_bfloat16 *>(final_normed.ptr), static_cast<__nv_bfloat16 *>(hidden_bf16_out.ptr),
      static_cast<float *>(lm_bmv.ptr), static_cast<int *>(lm_bmi.ptr),
      options.max_seq_len,
      stream);
    check_cuda(cudaGetLastError(), "launch_prefill_bf16");
  };

  for (int i = 0; i < options.warmup; ++i) {
    run_prefill();
  }
  check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(warmup)");

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
  check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");
  check_cuda(cudaEventRecord(start, stream), "cudaEventRecord(start)");
  for (int i = 0; i < options.runs; ++i) {
    run_prefill();
  }
  check_cuda(cudaEventRecord(stop, stream), "cudaEventRecord(stop)");
  check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
  float total_ms = 0.0f;
  check_cuda(cudaEventElapsedTime(&total_ms, start, stop), "cudaEventElapsedTime");

  int out_token = -1;
  check_cuda(cudaMemcpy(&out_token, output_token.ptr, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy(output_token)");
  const double avg_ms = static_cast<double>(total_ms) / options.runs;
  const double tok_s = static_cast<double>(S) * 1000.0 / avg_ms;
  std::cout << "prefill native benchmark\n";
  std::cout << "prompt_tokens: " << S << "\n";
  std::cout << "warmup: " << options.warmup << "\n";
  std::cout << "runs: " << options.runs << "\n";
  std::cout << "avg_ms: " << avg_ms << "\n";
  std::cout << "pp" << S << ": " << tok_s << " tok/s (" << avg_ms << "ms)\n";
  std::cout << "output_token: " << out_token << "\n";

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);
  free_tensor(layers_dev);
  free_tensor(fused_fa_qkv_dev);
  free_tensor(fused_gate_up_dev);
  for (auto & t : owned) free_tensor(t);
  return 0;
}
