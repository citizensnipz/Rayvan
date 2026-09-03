#include "rayvan_emc/tensor.hpp"

#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace rayvan::emc {
namespace {

template <typename scalar_t>
__device__ __forceinline__ float load_scalar(const scalar_t* pointer, std::int64_t index) {
    return static_cast<float>(pointer[index]);
}

template <typename scalar_t>
__device__ __forceinline__ void store_scalar(scalar_t* pointer, std::int64_t index, float value) {
    pointer[index] = static_cast<scalar_t>(value);
}

template <typename scalar_t>
__global__ void delta_forward_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    const float* __restrict__ alpha,
    const float* __restrict__ beta,
    const float* __restrict__ initial_state,
    float* __restrict__ output,
    float* __restrict__ final_state,
    float* __restrict__ boundaries,
    std::int64_t sequence,
    std::int64_t heads,
    std::int64_t dimension,
    std::int64_t chunk_size,
    std::int64_t num_chunks) {
    const auto head = static_cast<std::int64_t>(blockIdx.x);
    const auto batch = static_cast<std::int64_t>(blockIdx.y);
    const auto thread = static_cast<std::int64_t>(threadIdx.x);
    const auto matrix_elements = dimension * dimension;
    const auto state_base = (batch * heads + head) * matrix_elements;
    extern __shared__ float shared[];
    float* state = shared;
    float* retrieved = state + matrix_elements;
    float* residual = retrieved + dimension;

    for (auto index = thread; index < matrix_elements; index += blockDim.x) {
        state[index] = initial_state[state_base + index];
        boundaries[((batch * heads + head) * (num_chunks + 1)) * matrix_elements + index] = state[index];
    }
    __syncthreads();

    for (std::int64_t token = 0; token < sequence; ++token) {
        const auto vector_base = ((batch * sequence + token) * heads + head) * dimension;
        const auto gate_index = (batch * sequence + token) * heads + head;
        const float a = alpha[gate_index];
        const float b = beta[gate_index];
        if (thread < dimension) {
            float dot = 0.0f;
            const auto row_base = thread * dimension;
            for (std::int64_t column = 0; column < dimension; ++column) {
                dot += state[row_base + column] * load_scalar(key, vector_base + column);
            }
            retrieved[thread] = dot;
            residual[thread] = load_scalar(value, vector_base + thread) - a * dot;
        }
        __syncthreads();
        for (auto index = thread; index < matrix_elements; index += blockDim.x) {
            const auto row = index / dimension;
            const auto column = index - row * dimension;
            state[index] = a * state[index] + b * residual[row] * load_scalar(key, vector_base + column);
        }
        __syncthreads();
        if (thread < dimension) {
            float dot = 0.0f;
            const auto row_base = thread * dimension;
            for (std::int64_t column = 0; column < dimension; ++column) {
                dot += state[row_base + column] * load_scalar(query, vector_base + column);
            }
            output[vector_base + thread] = dot;
        }
        __syncthreads();
        if ((token + 1) % chunk_size == 0 || token + 1 == sequence) {
            const auto boundary = (token + 1 + chunk_size - 1) / chunk_size;
            const auto boundary_base = ((batch * heads + head) * (num_chunks + 1) + boundary) * matrix_elements;
            for (auto index = thread; index < matrix_elements; index += blockDim.x) {
                boundaries[boundary_base + index] = state[index];
            }
            __syncthreads();
        }
    }
    for (auto index = thread; index < matrix_elements; index += blockDim.x) {
        final_state[state_base + index] = state[index];
    }
}

template <typename scalar_t>
__global__ void delta_backward_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    const float* __restrict__ alpha,
    const float* __restrict__ beta,
    const float* __restrict__ boundaries,
    const float* __restrict__ grad_output,
    const float* __restrict__ grad_final,
    float* __restrict__ scratch,
    scalar_t* __restrict__ grad_query,
    scalar_t* __restrict__ grad_key,
    scalar_t* __restrict__ grad_value,
    float* __restrict__ grad_alpha,
    float* __restrict__ grad_beta,
    float* __restrict__ grad_initial,
    std::int64_t sequence,
    std::int64_t heads,
    std::int64_t dimension,
    std::int64_t chunk_size,
    std::int64_t num_chunks) {
    const auto head = static_cast<std::int64_t>(blockIdx.x);
    const auto batch = static_cast<std::int64_t>(blockIdx.y);
    const auto thread = static_cast<std::int64_t>(threadIdx.x);
    const auto matrix_elements = dimension * dimension;
    const auto state_base = (batch * heads + head) * matrix_elements;
    const auto scratch_base = (batch * heads + head) * (chunk_size + 1) * matrix_elements;
    extern __shared__ float shared[];
    float* state = shared;
    float* grad_state = state + matrix_elements;
    float* retrieved = grad_state + matrix_elements;
    float* residual = retrieved + dimension;
    float* grad_times_key = residual + dimension;
    float* reductions = grad_times_key + dimension;

    for (auto index = thread; index < matrix_elements; index += blockDim.x) {
        grad_state[index] = grad_final[state_base + index];
    }
    __syncthreads();

    for (std::int64_t chunk = num_chunks; chunk-- > 0;) {
        const auto start = chunk * chunk_size;
        const auto end = min(start + chunk_size, sequence);
        const auto length = end - start;
        const auto boundary_base = ((batch * heads + head) * (num_chunks + 1) + chunk) * matrix_elements;
        for (auto index = thread; index < matrix_elements; index += blockDim.x) {
            state[index] = boundaries[boundary_base + index];
            scratch[scratch_base + index] = state[index];
        }
        __syncthreads();

        // Recompute just this chunk. No token-state matrices are retained by autograd.
        for (std::int64_t local = 0; local < length; ++local) {
            const auto token = start + local;
            const auto vector_base = ((batch * sequence + token) * heads + head) * dimension;
            const auto gate_index = (batch * sequence + token) * heads + head;
            const float a = alpha[gate_index];
            const float b = beta[gate_index];
            if (thread < dimension) {
                float dot = 0.0f;
                const auto row_base = thread * dimension;
                for (std::int64_t column = 0; column < dimension; ++column) {
                    dot += state[row_base + column] * load_scalar(key, vector_base + column);
                }
                retrieved[thread] = dot;
                residual[thread] = load_scalar(value, vector_base + thread) - a * dot;
            }
            __syncthreads();
            for (auto index = thread; index < matrix_elements; index += blockDim.x) {
                const auto row = index / dimension;
                const auto column = index - row * dimension;
                state[index] = a * state[index] + b * residual[row] * load_scalar(key, vector_base + column);
            }
            __syncthreads();
            const auto destination = scratch_base + (local + 1) * matrix_elements;
            for (auto index = thread; index < matrix_elements; index += blockDim.x) {
                scratch[destination + index] = state[index];
            }
            __syncthreads();
        }

        for (std::int64_t local = length; local-- > 0;) {
            const auto token = start + local;
            const auto vector_base = ((batch * sequence + token) * heads + head) * dimension;
            const auto gate_index = (batch * sequence + token) * heads + head;
            const float a = alpha[gate_index];
            const float b = beta[gate_index];
            const float* previous = scratch + scratch_base + local * matrix_elements;
            const float* current = scratch + scratch_base + (local + 1) * matrix_elements;

            if (thread < dimension) {
                float dq = 0.0f;
                for (std::int64_t row = 0; row < dimension; ++row) {
                    dq += current[row * dimension + thread] * grad_output[vector_base + row];
                }
                store_scalar(grad_query, vector_base + thread, dq);
            }
            for (auto index = thread; index < matrix_elements; index += blockDim.x) {
                const auto row = index / dimension;
                const auto column = index - row * dimension;
                grad_state[index] += grad_output[vector_base + row] * load_scalar(query, vector_base + column);
            }
            __syncthreads();

            if (thread < dimension) {
                float r = 0.0f;
                float gk = 0.0f;
                const auto row_base = thread * dimension;
                for (std::int64_t column = 0; column < dimension; ++column) {
                    const float kval = load_scalar(key, vector_base + column);
                    r += previous[row_base + column] * kval;
                    gk += grad_state[row_base + column] * kval;
                }
                retrieved[thread] = r;
                residual[thread] = load_scalar(value, vector_base + thread) - a * r;
                grad_times_key[thread] = gk;
                store_scalar(grad_value, vector_base + thread, b * gk);
            }
            __syncthreads();

            if (thread < dimension) {
                float dk = 0.0f;
                for (std::int64_t row = 0; row < dimension; ++row) {
                    dk += b * grad_state[row * dimension + thread] * residual[row]
                        - a * b * previous[row * dimension + thread] * grad_times_key[row];
                }
                store_scalar(grad_key, vector_base + thread, dk);
            }
            if (thread == 0) {
                reductions[0] = 0.0f;
                reductions[1] = 0.0f;
            }
            __syncthreads();
            float local_alpha = 0.0f;
            for (auto index = thread; index < matrix_elements; index += blockDim.x) {
                local_alpha += grad_state[index] * previous[index];
            }
            atomicAdd(&reductions[0], local_alpha);
            if (thread < dimension) {
                atomicAdd(&reductions[0], -b * grad_times_key[thread] * retrieved[thread]);
                atomicAdd(&reductions[1], grad_times_key[thread] * residual[thread]);
            }
            __syncthreads();
            if (thread == 0) {
                grad_alpha[gate_index] = reductions[0];
                grad_beta[gate_index] = reductions[1];
            }
            for (auto index = thread; index < matrix_elements; index += blockDim.x) {
                const auto row = index / dimension;
                const auto column = index - row * dimension;
                grad_state[index] = a * (
                    grad_state[index] - b * grad_times_key[row] * load_scalar(key, vector_base + column));
            }
            __syncthreads();
        }
    }
    for (auto index = thread; index < matrix_elements; index += blockDim.x) {
        grad_initial[state_base + index] = grad_state[index];
    }
}

std::size_t forward_shared_bytes(std::int64_t dimension) {
    return static_cast<std::size_t>(dimension * dimension + 2 * dimension) * sizeof(float);
}

std::size_t backward_shared_bytes(std::int64_t dimension) {
    return static_cast<std::size_t>(2 * dimension * dimension + 3 * dimension + 2) * sizeof(float);
}

}  // namespace

std::vector<Tensor> delta_cuda_forward(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& alpha,
    const Tensor& beta,
    const Tensor& initial_state,
    std::int64_t chunk_size) {
    c10::cuda::CUDAGuard guard(query.device());
    const auto batch = query.size(0);
    const auto sequence = query.size(1);
    const auto heads = query.size(2);
    const auto dimension = query.size(3);
    const auto num_chunks = 1 + (sequence - 1) / chunk_size;
    auto output = torch::empty(query.sizes(), query.options().dtype(torch::kFloat32));
    auto final_state = torch::empty_like(initial_state);
    auto boundaries = torch::empty(
        {batch, heads, num_chunks + 1, dimension, dimension}, initial_state.options());
    const dim3 grid(static_cast<unsigned>(heads), static_cast<unsigned>(batch));
    constexpr int threads = 256;
    const auto stream = at::cuda::getCurrentCUDAStream(query.get_device()).stream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, query.scalar_type(), "rayvan_delta_forward", [&] {
            delta_forward_kernel<scalar_t><<<grid, threads, forward_shared_bytes(dimension), stream>>>(
                query.const_data_ptr<scalar_t>(), key.const_data_ptr<scalar_t>(), value.const_data_ptr<scalar_t>(),
                alpha.const_data_ptr<float>(), beta.const_data_ptr<float>(), initial_state.const_data_ptr<float>(),
                output.mutable_data_ptr<float>(), final_state.mutable_data_ptr<float>(), boundaries.mutable_data_ptr<float>(),
                sequence, heads, dimension, chunk_size, num_chunks);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {output, final_state, boundaries};
}

std::vector<Tensor> delta_cuda_backward(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& alpha,
    const Tensor& beta,
    const Tensor& initial_state,
    const Tensor& boundaries,
    const Tensor& grad_output,
    const Tensor& grad_final,
    std::int64_t chunk_size,
    std::int64_t max_scratch_bytes) {
    c10::cuda::CUDAGuard guard(query.device());
    const auto batch = query.size(0);
    const auto sequence = query.size(1);
    const auto heads = query.size(2);
    const auto dimension = query.size(3);
    const auto effective_chunk = std::min(chunk_size, sequence);
    const auto scratch_elements = batch * heads * (effective_chunk + 1) * dimension * dimension;
    const auto scratch_bytes = scratch_elements * static_cast<std::int64_t>(sizeof(float));
    if (scratch_bytes > max_scratch_bytes) {
        throw std::runtime_error("Delta CUDA backward scratch allocation refused by safety limit");
    }
    auto scratch = torch::empty({scratch_elements}, initial_state.options());
    auto grad_query = torch::empty_like(query);
    auto grad_key = torch::empty_like(key);
    auto grad_value = torch::empty_like(value);
    auto grad_alpha = torch::empty_like(alpha);
    auto grad_beta = torch::empty_like(beta);
    auto grad_initial = torch::empty_like(initial_state);
    const auto num_chunks = 1 + (sequence - 1) / effective_chunk;
    const dim3 grid(static_cast<unsigned>(heads), static_cast<unsigned>(batch));
    constexpr int threads = 256;
    const auto stream = at::cuda::getCurrentCUDAStream(query.get_device()).stream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, query.scalar_type(), "rayvan_delta_backward", [&] {
            delta_backward_kernel<scalar_t><<<grid, threads, backward_shared_bytes(dimension), stream>>>(
                query.const_data_ptr<scalar_t>(), key.const_data_ptr<scalar_t>(), value.const_data_ptr<scalar_t>(),
                alpha.const_data_ptr<float>(), beta.const_data_ptr<float>(), boundaries.const_data_ptr<float>(),
                grad_output.const_data_ptr<float>(), grad_final.const_data_ptr<float>(), scratch.mutable_data_ptr<float>(),
                grad_query.mutable_data_ptr<scalar_t>(), grad_key.mutable_data_ptr<scalar_t>(), grad_value.mutable_data_ptr<scalar_t>(),
                grad_alpha.mutable_data_ptr<float>(), grad_beta.mutable_data_ptr<float>(), grad_initial.mutable_data_ptr<float>(),
                sequence, heads, dimension, effective_chunk, num_chunks);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {grad_query, grad_key, grad_value, grad_alpha, grad_beta, grad_initial};
}

}  // namespace rayvan::emc
