#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>

// Mock engine_forward function - replace with actual GGML implementation
static int engine_forward(float* input, float* output, int size) {
    if (!input || !output || size <= 0) return -1;
    
    // Simple forward pass simulation
    for (int i = 0; i < size; i++) {
        output[i] = input[i] * 2.0f + 1.0f;
    }
    return 0;
}

// Python wrapper function
static PyObject* py_engine_forward(PyObject* self, PyObject* args) {
    Py_buffer input_buf;
    int size;
    
    if (!PyArg_ParseTuple(args, "y*i", &input_buf, &size)) {
        return NULL;
    }
    
    if (input_buf.len != size * sizeof(float)) {
        PyErr_SetString(PyExc_ValueError, "Input buffer size mismatch");
        PyBuffer_Release(&input_buf);
        return NULL;
    }
    
    float* input = (float*)input_buf.buf;
    float* output = malloc(size * sizeof(float));
    
    if (!output) {
        PyBuffer_Release(&input_buf);
        return PyErr_NoMemory();
    }
    
    int result = engine_forward(input, output, size);
    
    if (result != 0) {
        free(output);
        PyBuffer_Release(&input_buf);
        PyErr_SetString(PyExc_RuntimeError, "engine_forward failed");
        return NULL;
    }
    
    PyObject* output_obj = PyBytes_FromStringAndSize((char*)output, size * sizeof(float));
    free(output);
    PyBuffer_Release(&input_buf);
    
    return output_obj;
}

// Method table
static PyMethodDef GGMLExtensionMethods[] = {
    {"forward", py_engine_forward, METH_VARARGS, "Run engine forward pass"},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef ggmlextension_module = {
    PyModuleDef_HEAD_INIT,
    "ggml_extension",
    "GGML Engine Forward Pass Extension",
    -1,
    GGMLExtensionMethods
};

PyMODINIT_FUNC PyInit_ggml_extension(void) {
    return PyModule_Create(&ggmlextension_module);
}
