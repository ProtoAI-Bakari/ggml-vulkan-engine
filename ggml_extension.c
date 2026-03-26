#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>

// Forward declaration of the engine_forward function
// This should be implemented elsewhere in your ggml engine
typedef struct {
    void* ctx;
    int n_threads;
} EngineContext;

extern int engine_forward(EngineContext* ctx, const float* input, float* output, int input_size);

static PyObject* py_engine_forward(PyObject* self, PyObject* args) {
    PyArrayObject* input_arr = NULL;
    int n_threads = 4;
    
    if (!PyArg_ParseTuple(args, "O!|i", &PyArray_Type, &input_arr, &n_threads)) {
        return NULL;
    }
    
    // Validate input array is contiguous and float type
    if (!PyArray_ISCONTIGUOUS(input_arr) || PyArray_TYPE(input_arr) != NPY_FLOAT) {
        PyErr_SetString(PyExc_ValueError, "Input must be contiguous float array");
        return NULL;
    }
    
    int input_size = (int)PyArray_SIZE(input_arr);
    float* input_data = (float*)PyArray_DATA(input_arr);
    
    // Allocate output buffer
    npy_intp dims[] = {input_size};
    PyArrayObject* output_arr = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_FLOAT);
    if (output_arr == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate output array");
        return NULL;
    }
    
    float* output_data = (float*)PyArray_DATA(output_arr);
    
    // Initialize engine context
    EngineContext ctx = {NULL, n_threads};
    
    // Call the forward function
    int result = engine_forward(&ctx, input_data, output_data, input_size);
    
    if (result != 0) {
        Py_DECREF(output_arr);
        PyErr_Format(PyExc_RuntimeError, "engine_forward failed with code %d", result);
        return NULL;
    }
    
    return (PyObject*)output_arr;
}

static PyObject* py_init_engine(PyObject* self, PyObject* args) {
    int n_threads = 4;
    
    if (!PyArg_ParseTuple(args, "|i", &n_threads)) {
        return NULL;
    }
    
    printf("Initializing GGML engine with %d threads\n", n_threads);
    Py_RETURN_NONE;
}

static PyMethodDef ExtensionMethods[] = {
    {"forward", py_engine_forward, METH_VARARGS,
     "Run forward pass through the model. Takes input array, returns output array."},
    {"init_engine", py_init_engine, METH_VARARGS,
     "Initialize the engine with specified thread count."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef ggmlmodule = {
    PyModuleDef_HEAD_INIT,
    "ggml_extension",
    "GGML Python C Extension",
    -1,
    ExtensionMethods
};

PyMODINIT_FUNC PyInit_ggml_extension(void) {
    import_array();
    return PyModule_Create(&ggmlmodule);
}
