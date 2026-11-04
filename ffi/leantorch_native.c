/*
LeanTorch native CPU backend (default). Pure C, no external dependencies.

Tensors are Lean external objects wrapping a heap-allocated, row-major double buffer. 
Lean's reference counting frees them through the finalizer, so there is no manual free and no double-free. 
Every function returns an IO result; shape errors become Lean IO.userError instead of crashing the process.

Supported : rank-1/2 float64 CPU tensors.
 */
#include <lean/lean.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    size_t rank;      /* 1 or 2 */
    size_t dims[2];
    size_t n;         /* number of elements */
    double *data;
} lt_tensor;

static lean_external_class *g_class = NULL;

static void lt_finalize(void *p) {
    lt_tensor *t = (lt_tensor *)p;
    if (t) { free(t->data); free(t); }
}
static void lt_foreach(void *p, b_lean_obj_arg f) { (void)p; (void)f; }

static lean_external_class *lt_class(void) {
    if (!g_class) g_class = lean_register_external_class(lt_finalize, lt_foreach);
    return g_class;
}

static lean_obj_res io_err(const char *msg) {
    return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(msg)));
}
static lt_tensor *unwrap(b_lean_obj_arg o) { return (lt_tensor *)lean_get_external_data(o); }

static lt_tensor *lt_alloc(size_t rank, size_t d0, size_t d1) {
    lt_tensor *t = (lt_tensor *)calloc(1, sizeof(lt_tensor));
    if (!t) return NULL;
    t->rank = rank; t->dims[0] = d0; t->dims[1] = rank == 2 ? d1 : 1;
    t->n = rank == 2 ? d0 * d1 : d0;
    t->data = (double *)calloc(t->n ? t->n : 1, sizeof(double));
    if (!t->data) { free(t); return NULL; }
    return t;
}
static lean_obj_res lt_ok(lt_tensor *t) {
    if (!t) return io_err("[LeanTorch native] out of memory");
    return lean_io_result_mk_ok(lean_alloc_external(lt_class(), t));
}
static int same_shape(const lt_tensor *a, const lt_tensor *b) {
    return a->rank == b->rank && a->dims[0] == b->dims[0] && a->dims[1] == b->dims[1];
}

LEAN_EXPORT lean_obj_res leantorch_backend_name(lean_obj_arg unit) {
    (void)unit;
    return lean_mk_string("native-c (CPU, float64)");
}

LEAN_EXPORT lean_obj_res leantorch_tensor_create_1d(b_lean_obj_arg data, uint32_t device, lean_obj_arg w) {
    (void)w;
    if (device != 0) return io_err("[LeanTorch native] only the CPU device is available in the native-c backend");
    size_t n = lean_array_size(data);
    lt_tensor *t = lt_alloc(1, n, 1);
    if (!t) return io_err("[LeanTorch native] out of memory");
    for (size_t i = 0; i < n; ++i) t->data[i] = lean_unbox_float(lean_array_get_core(data, i));
    return lt_ok(t);
}

LEAN_EXPORT lean_obj_res leantorch_tensor_create_2d(b_lean_obj_arg data, b_lean_obj_arg rows,
                                                    b_lean_obj_arg cols, uint32_t device, lean_obj_arg w) {
    (void)w;
    if (device != 0) return io_err("[LeanTorch native] only the CPU device is available in the native-c backend");
    if (!lean_is_scalar(rows) || !lean_is_scalar(cols)) return io_err("[LeanTorch native] dimension too large");
    size_t r = lean_usize_of_nat(rows), c = lean_usize_of_nat(cols);
    if (lean_array_size(data) != r * c) return io_err("[LeanTorch native] array size does not match rows*cols");
    lt_tensor *t = lt_alloc(2, r, c);
    if (!t) return io_err("[LeanTorch native] out of memory");
    for (size_t i = 0; i < r * c; ++i) t->data[i] = lean_unbox_float(lean_array_get_core(data, i));
    return lt_ok(t);
}

static lean_obj_res binop(b_lean_obj_arg a, b_lean_obj_arg b, int sub) {
    lt_tensor *x = unwrap(a), *y = unwrap(b);
    if (!same_shape(x, y)) return io_err("[LeanTorch native] shape mismatch in elementwise operation");
    lt_tensor *r = lt_alloc(x->rank, x->dims[0], x->dims[1]);
    if (!r) return io_err("[LeanTorch native] out of memory");
    for (size_t i = 0; i < x->n; ++i) r->data[i] = sub ? x->data[i] - y->data[i] : x->data[i] + y->data[i];
    return lt_ok(r);
}
LEAN_EXPORT lean_obj_res leantorch_tensor_add(b_lean_obj_arg a, b_lean_obj_arg b, lean_obj_arg w) { (void)w; return binop(a, b, 0); }
LEAN_EXPORT lean_obj_res leantorch_tensor_sub(b_lean_obj_arg a, b_lean_obj_arg b, lean_obj_arg w) { (void)w; return binop(a, b, 1); }

/* [m,n] x [n,p] -> [m,p]   or   [m,n] x [n] -> [m] */
LEAN_EXPORT lean_obj_res leantorch_tensor_matmul(b_lean_obj_arg a, b_lean_obj_arg b, lean_obj_arg w) {
    (void)w;
    lt_tensor *x = unwrap(a), *y = unwrap(b);
    if (x->rank != 2) return io_err("[LeanTorch native] matmul: left operand must be a matrix");
    size_t m = x->dims[0], n = x->dims[1];
    if (y->rank == 2) {
        if (y->dims[0] != n) return io_err("[LeanTorch native] matmul: inner dimensions differ");
        size_t p = y->dims[1];
        lt_tensor *r = lt_alloc(2, m, p);
        if (!r) return io_err("[LeanTorch native] out of memory");
        for (size_t i = 0; i < m; ++i)
            for (size_t k = 0; k < p; ++k) {
                double s = 0.0;
                for (size_t j = 0; j < n; ++j) s += x->data[i * n + j] * y->data[j * p + k];
                r->data[i * p + k] = s;
            }
        return lt_ok(r);
    }
    if (y->dims[0] != n) return io_err("[LeanTorch native] matmul: inner dimensions differ");
    lt_tensor *r = lt_alloc(1, m, 1);
    if (!r) return io_err("[LeanTorch native] out of memory");
    for (size_t i = 0; i < m; ++i) {
        double s = 0.0;
        for (size_t j = 0; j < n; ++j) s += x->data[i * n + j] * y->data[j];
        r->data[i] = s;
    }
    return lt_ok(r);
}

static lean_obj_res unop(b_lean_obj_arg a, int which) {
    lt_tensor *x = unwrap(a);
    lt_tensor *r = lt_alloc(x->rank, x->dims[0], x->dims[1]);
    if (!r) return io_err("[LeanTorch native] out of memory");
    for (size_t i = 0; i < x->n; ++i) r->data[i] = which == 0 ? (x->data[i] > 0.0 ? x->data[i] : 0.0) : tanh(x->data[i]);
    return lt_ok(r);
}
LEAN_EXPORT lean_obj_res leantorch_tensor_relu(b_lean_obj_arg a, lean_obj_arg w) { (void)w; return unop(a, 0); }
LEAN_EXPORT lean_obj_res leantorch_tensor_tanh(b_lean_obj_arg a, lean_obj_arg w) { (void)w; return unop(a, 1); }

LEAN_EXPORT lean_obj_res leantorch_tensor_to_array(b_lean_obj_arg a, lean_obj_arg w) {
    (void)w;
    lt_tensor *x = unwrap(a);
    lean_object *arr = lean_alloc_array(x->n, x->n);
    for (size_t i = 0; i < x->n; ++i) lean_array_set_core(arr, i, lean_box_float(x->data[i]));
    return lean_io_result_mk_ok(arr);
}
LEAN_EXPORT lean_obj_res leantorch_tensor_rank(b_lean_obj_arg a, lean_obj_arg w) {
    (void)w; return lean_io_result_mk_ok(lean_usize_to_nat(unwrap(a)->rank));
}
LEAN_EXPORT lean_obj_res leantorch_tensor_dim(b_lean_obj_arg a, b_lean_obj_arg idx, lean_obj_arg w) {
    (void)w;
    lt_tensor *x = unwrap(a);
    if (!lean_is_scalar(idx) || lean_usize_of_nat(idx) >= x->rank) return io_err("[LeanTorch native] dimension index out of range");
    return lean_io_result_mk_ok(lean_usize_to_nat(x->dims[lean_usize_of_nat(idx)]));
}
