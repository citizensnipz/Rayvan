cmake_minimum_required(VERSION 3.25)
if(POLICY CMP0219)
    cmake_policy(SET CMP0219 NEW)
endif()
project(rayvan_emc_native VERSION 0.1.0 LANGUAGES CXX)

option(RAYVAN_EMC_BUILD_TESTS "Build native EMC tests" ON)
option(RAYVAN_EMC_BUILD_TOOLS "Build native EMC command-line tools" ON)

find_package(Torch REQUIRED)
if(TARGET torch_cuda)
    find_package(CUDAToolkit REQUIRED)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES native)
        endif()
        set(RAYVAN_EMC_NATIVE_CUDA ON)
    endif()
endif()

add_library(rayvan_emc
    src/config.cpp
    src/n1.cpp
    src/shared_gqa.cpp
    src/routing_free_collective.cpp
    src/delta_core.cpp
    src/delta_n1.cpp
    src/n2.cpp
    src/model.cpp
    src/dataset.cpp
    src/checkpoint.cpp
    src/diagnostics.cpp
    src/foreach_adamw.cpp
    src/trainer.cpp
)
if(RAYVAN_EMC_NATIVE_CUDA)
    target_sources(rayvan_emc PRIVATE src/delta_cuda.cu)
    set_source_files_properties(src/delta_cuda.cu PROPERTIES LANGUAGE CUDA)
    set_target_properties(rayvan_emc PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
endif()
add_library(rayvan::emc ALIAS rayvan_emc)

target_compile_features(rayvan_emc PUBLIC cxx_std_20)
target_include_directories(rayvan_emc PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(rayvan_emc PUBLIC ${TORCH_LIBRARIES})
if(TARGET torch_cuda)
    target_compile_definitions(rayvan_emc PRIVATE RAYVAN_TORCH_CUDA=1)
endif()
if(RAYVAN_EMC_NATIVE_CUDA)
    target_compile_definitions(rayvan_emc PRIVATE RAYVAN_EMC_CUDA_KERNELS=1)
    target_link_libraries(rayvan_emc PRIVATE CUDA::cudart)
endif()
if(MSVC)
    target_compile_options(rayvan_emc PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4;/permissive-;/EHsc;/bigobj>)
    target_link_libraries(rayvan_emc PRIVATE Psapi)
else()
    target_compile_options(rayvan_emc PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>)
endif()

# LibTorch publishes required ABI/compiler flags through TORCH_CXX_FLAGS.
separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(rayvan_emc PUBLIC ${TORCH_CXX_FLAGS_LIST})

if(RAYVAN_EMC_BUILD_TOOLS)
    add_executable(rayvan-emc-smoke tools/emc_smoke.cpp)
    target_link_libraries(rayvan-emc-smoke PRIVATE rayvan::emc)
    if(TARGET torch_cuda)
        target_compile_definitions(rayvan-emc-smoke PRIVATE RAYVAN_TORCH_CUDA=1)
    endif()
    add_executable(rayvan-emc-benchmark tools/emc_benchmark.cpp)
    target_link_libraries(rayvan-emc-benchmark PRIVATE rayvan::emc)
    add_executable(rayvan-emc-checkpoint-probe tools/emc_checkpoint_probe.cpp)
    target_link_libraries(rayvan-emc-checkpoint-probe PRIVATE rayvan::emc)
    if(TARGET torch_cuda)
        add_executable(rayvan-emc-optimizer-profile tools/emc_optimizer_profile.cpp)
        target_link_libraries(rayvan-emc-optimizer-profile PRIVATE rayvan::emc CUDA::cudart CUDA::cupti)
        if(RAYVAN_EMC_NATIVE_CUDA)
            add_executable(rayvan-emc-delta-benchmark tools/emc_delta_benchmark.cpp)
            target_link_libraries(rayvan-emc-delta-benchmark PRIVATE rayvan::emc CUDA::cudart CUDA::cupti)
            add_executable(rayvan-emc-train tools/emc_train.cpp)
            target_link_libraries(rayvan-emc-train PRIVATE rayvan::emc)
            add_executable(rayvan-emc-causal tools/emc_causal.cpp)
            target_link_libraries(rayvan-emc-causal PRIVATE rayvan::emc)
            add_executable(rayvan-emc-capability-eval tools/emc_capability_eval.cpp)
            target_link_libraries(rayvan-emc-capability-eval PRIVATE rayvan::emc)
            add_executable(rayvan-emc-routing-free-benchmark tools/emc_routing_free_benchmark.cpp)
            target_link_libraries(rayvan-emc-routing-free-benchmark PRIVATE rayvan::emc CUDA::cudart CUDA::cupti)
        endif()
    endif()
endif()

include(CTest)
if(BUILD_TESTING AND RAYVAN_EMC_BUILD_TESTS)
    add_executable(rayvan-emc-tests tests/test_main.cpp)
    target_link_libraries(rayvan-emc-tests PRIVATE rayvan::emc)
    target_compile_definitions(rayvan-emc-tests PRIVATE
        RAYVAN_EMC_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME rayvan-emc-native COMMAND rayvan-emc-tests)
endif()
