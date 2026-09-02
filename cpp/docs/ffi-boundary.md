# Future Rust FFI boundary

Rust integration is intentionally not implemented in this task. The production boundary should be a small C ABI over opaque native handles; Rust must not include C++ templates, LibTorch headers, exceptions, STL containers, or `at::Tensor` layout.

Recommended boundary:

```c
struct rv_emc_model;
struct rv_emc_trainer;

typedef struct {
    const void* data;
    uint64_t byte_length;
    int64_t sizes[4];
    uint32_t rank;
    uint32_t dtype;
    uint32_t device_type;
    int32_t device_index;
} rv_emc_tensor_view;

typedef struct {
    void* data;
    uint64_t byte_length;
    void (*release)(void* data, void* context);
    void* context;
} rv_emc_owned_bytes;
```

Required functions:

- create/destroy model handle from a config byte buffer;
- load/save checkpoint with UTF-8 path bytes and explicit lengths;
- set train/eval mode and active top-K;
- infer from owned or borrowed contiguous int64 token IDs;
- perform one train step from token/target views;
- retrieve diagnostics as versioned bytes;
- release every C++-allocated output explicitly;
- retrieve thread-local error code and UTF-8 error message.

Ownership rules:

1. Every handle has exactly one destroy function and is never shared implicitly.
2. Borrowed input views remain owned by Rust and are valid only for the call duration unless an API explicitly returns a lease.
3. C++ output uses an explicit release callback or a dedicated `rv_emc_bytes_free`; Rust never calls `delete`.
4. CUDA tensors should initially cross as copied host bytes. A later zero-copy API can add versioned CUDA IPC/DLPack handles without changing the basic ABI.
5. Exceptions never cross the ABI. Entry points catch them and return stable error codes.
6. Structs contain fixed-width integer fields, explicit rank/length, and a leading ABI version where evolution is likely.
7. Training and inference on one model handle are externally serialized unless a later API documents stream/thread ownership.

The existing C++ `EMCModel` owns its module through a `unique_ptr` and has move-only semantics, matching an opaque-handle implementation. `at::Tensor` remains inside the C++ library API for native callers but should terminate at the C shim.
