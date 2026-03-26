#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <stdint.h>
#include <stddef.h>

/* Forward declaration of the engine function */
extern void engine_forward(void *ctx, const float *input, float *output,
                           int batch_size, int seq_len, int hidden_dim);

static PyObject* py_engine_forward(PyObject *self, PyObject *args) {
    PyArrayObject *input_arr = NULL;
    PyArrayObject *output_arr = NULL;
    int batch_size, seq_len, hidden_dim;
    
    if (!PyArg_ParseTuple(args, "O!O!iii", &PyArray_Type, &input_arr, &PyArray_Type, &output_arr,
                          &batch_size, &seq_len, &hidden_dim)) {
        return NULL;
    }
    
    /* Validate array dimensions */
    if (PyArray_NDIM(input_arr) != 3 || PyArray_NDIM(output_arr) != 3) {
        PyErr_SetString(PyExc_ValueError, "Input/output arrays must be 3D");
        return NULL;
    }
    
    if (PyArray_DIM(input_arr, 0) != batch_size ||
        PyArray_DIM(input_arr, 1) != seq_len ||
        PyArray_DIM(input_arr, 2) != hidden_dim) {
        PyErr_Format(PyExc_ValueError, "Dimension mismatch: expected (%d,%d,%d)",
                     batch_size, seq_len, hidden_dim);
        return NULL;
    }
    
    if (PyArray_DIM(output_arr, 0) != batch_size ||
        PyArray_DIM(output_arr, 1) != seq_len ||
        PyArray_DIM(output_arr, 2) != hidden_dim) {
        PyErr_Format(PyExc_ValueError, "Output dimension mismatch: expected (%d,%d,%d)",
                     batch_size, seq_len, hidden_dim);
        return NULL;
    }
    
    /* Call the C engine function */
    engine_forward(NULL,
                   (const float *)PyArray_DATA(input_arr),
                   (float *)PyArray_DATA(output_arr),
                   batch_size, seq_len, hidden_dim);
    
    Py_RETURN_NONE;
}

static PyMethodDef ExtensionMethods[] = {
    {"engine_forward", py_engine_forward, METH_VARARGS,
     "Call the GGML engine forward pass"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "ggml_extension",
    "GGML Engine Python Extension",
    -1,
    ExtensionMethods
};

PyMODINIT_FUNC PyInit_ggml_extension(void) {
    import_array();
    return PyModule_Create(&moduledef);
}
