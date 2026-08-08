#include <torch/torch.h>
#include <lean/lean.h>
#include <vector>
#include <iostream>
#include <cstdint>

extern "C" {

// Converts Lean Float array into a std::vector<double>
static std::vector<double> lean_float_array_to_vector(lean_object* arr) {
    size_t size = lean_array_size(arr);
    std::vector<double> vec(size);
    for (size_t i = 0; i < size; ++i) {
        vec[i] = lean_unbox_float(lean_array_get_core(arr, i));
    }
    return vec;
}

// Converts torch::ScalarType integer code to at::ScalarType
static torch::ScalarType parse_dtype(int32_t dtype_code) {
    switch (dtype_code) {
        case 0: return torch::kFloat32;
        case 1: return torch::kFloat64;
        case 2: return torch::kInt32;
        case 3: return torch::kInt64;
        default: return torch::kFloat64;
    }
}

// Converts torch device integer code to torch::Device
static torch::Device parse_device(int32_t device_code) {
    if (device_code == 0) {
        return torch::kCPU;
    } else if (device_code >= 100) {
        return torch::Device(torch::kCUDA, static_cast<int16_t>(device_code - 100));
    }
    return torch::kCPU;
}

/**
 * Creates a 1D tensor from a Lean Float array.
 */
lean_obj_res leantorch_tensor_create_1d(b_lean_obj_arg data_arr, size_t size, int32_t dtype_code, int32_t device_code) {
    std::vector<double> vec = lean_float_array_to_vector(data_arr);
    torch::TensorOptions options = torch::TensorOptions()
                                      .dtype(parse_dtype(dtype_code))
                                      .device(parse_device(device_code));
    
    at::Tensor* tensor_ptr = new at::Tensor(torch::from_blob(vec.data(), {static_cast<int64_t>(size)}, torch::kFloat64).clone().to(options));
    return reinterpret_cast<lean_obj_res>(tensor_ptr);
}

/**
 * Creates a 2D tensor (matrix) from a Lean Float array.
 */
lean_obj_res leantorch_tensor_create_2d(b_lean_obj_arg data_arr, size_t rows, size_t cols, int32_t dtype_code, int32_t device_code) {
    std::vector<double> vec = lean_float_array_to_vector(data_arr);
    torch::TensorOptions options = torch::TensorOptions()
                                      .dtype(parse_dtype(dtype_code))
                                      .device(parse_device(device_code));
    
    at::Tensor* tensor_ptr = new at::Tensor(torch::from_blob(vec.data(), {static_cast<int64_t>(rows), static_cast<int64_t>(cols)}, torch::kFloat64).clone().to(options));
    return reinterpret_cast<lean_obj_res>(tensor_ptr);
}

/**
 * Frees a previously allocated at::Tensor handle.
 */
lean_obj_res leantorch_tensor_free(lean_obj_arg handle) {
    at::Tensor* tensor_ptr = reinterpret_cast<at::Tensor*>(handle);
    if (tensor_ptr) {
        delete tensor_ptr;
    }
    return lean_io_result_mk_ok(lean_box(0));
}

/**
 * Performs pointwise addition: A + B
 */
lean_obj_res leantorch_tensor_add(lean_obj_arg handle_a, lean_obj_arg handle_b) {
    at::Tensor* a = reinterpret_cast<at::Tensor*>(handle_a);
    at::Tensor* b = reinterpret_cast<at::Tensor*>(handle_b);
    at::Tensor* res = new at::Tensor((*a) + (*b));
    return reinterpret_cast<lean_obj_res>(res);
}

/**
 * Performs pointwise subtraction: A - B
 */
lean_obj_res leantorch_tensor_sub(lean_obj_arg handle_a, lean_obj_arg handle_b) {
    at::Tensor* a = reinterpret_cast<at::Tensor*>(handle_a);
    at::Tensor* b = reinterpret_cast<at::Tensor*>(handle_b);
    at::Tensor* res = new at::Tensor((*a) - (*b));
    return reinterpret_cast<lean_obj_res>(res);
}

/**
 * Performs matrix multiplication: A x B
 */
lean_obj_res leantorch_tensor_matmul(lean_obj_arg handle_a, lean_obj_arg handle_b) {
    at::Tensor* a = reinterpret_cast<at::Tensor*>(handle_a);
    at::Tensor* b = reinterpret_cast<at::Tensor*>(handle_b);
    at::Tensor* res = new at::Tensor(torch::matmul(*a, *b));
    return reinterpret_cast<lean_obj_res>(res);
}

/**
 * Applies ReLU activation pointwise.
 */
lean_obj_res leantorch_tensor_relu(lean_obj_arg handle) {
    at::Tensor* t = reinterpret_cast<at::Tensor*>(handle);
    at::Tensor* res = new at::Tensor(torch::relu(*t));
    return reinterpret_cast<lean_obj_res>(res);
}

/**
 * Applies Tanh activation pointwise.
 */
lean_obj_res leantorch_tensor_tanh(lean_obj_arg handle) {
    at::Tensor* t = reinterpret_cast<at::Tensor*>(handle);
    at::Tensor* res = new at::Tensor(torch::tanh(*t));
    return reinterpret_cast<lean_obj_res>(res);
}

lean_obj_res leantorch_tensor_to_array(lean_obj_arg handle) {
    at::Tensor* t = reinterpret_cast<at::Tensor*>(handle);
    at::Tensor cpu_tensor = t->to(torch::kCPU).to(torch::kFloat64).contiguous();
    
    int64_t numel = cpu_tensor.numel();
    const double* data_ptr = cpu_tensor.data_ptr<double>();

    lean_object* arr = lean_alloc_array(numel, numel);
    for (int64_t i = 0; i < numel; ++i) {
        lean_array_set_core(arr, i, lean_box_float(data_ptr[i]));
    }

    return lean_io_result_mk_ok(arr);
}

lean_obj_res leantorch_tensor_get_rank(lean_obj_arg handle) {
    at::Tensor* t = reinterpret_cast<at::Tensor*>(handle);
    size_t dim = static_cast<size_t>(t->dim());
    return lean_io_result_mk_ok(lean_box_usize(dim));
}

lean_obj_res leantorch_tensor_get_dim(lean_obj_arg handle, size_t dim_idx) {
    at::Tensor* t = reinterpret_cast<at::Tensor*>(handle);
    size_t size = static_cast<size_t>(t->size(static_cast<int64_t>(dim_idx)));
    return lean_io_result_mk_ok(lean_box_usize(size));
}

} 
