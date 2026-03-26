#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>

// Forward declaration of the engine function
extern void engine_forward(void *ctx, float *input, float *output, int size);

static PyObject* py_engine_forward(PyObject *self, PyObject *args) {
    Py_buffer input_buf;
    Py_buffer output_buf;
    
    if (!PyArg_ParseTuple(args, "s*s*i", &input_buf, &output_buf, NULL)) {
        return NULL;
    }
    
    // Validate buffer sizes
    if (input_buf.len != output_buf.len) {
        PyErr_SetString(PyExc_ValueError, "Input and output buffers must have same size");
        PyBuffer_Release(&input_buf);
        PyBuffer_Release(&output_buf);
        return NULL;
    }
    
    int size = input_buf.len / sizeof(float);
    
    // Call the engine forward function
    engine_forward(NULL, (float*)input_buf.buf, (float*)output_buf.buf, size);
    
    PyBuffer_Release(&input_buf);
    PyBuffer_Release(&output_buf);
    
    Py_RETURN_NONE;
}

static PyObject* py_engine_version(PyObject *self, PyObject *args) {
    return PyUnicode_FromString("ggml-extension-1.0.0");
}

static PyMethodDef ExtensionMethods[] = {
    {"engine_forward", py_engine_forward, METH_VARARGS, "Run engine forward pass"},
    {"version", py_engine_version, METH_NOARGS, "Get extension version"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef ggmlmodule = {
    PyModuleDef_HEAD_INIT,
    "ggml_extension",
    "GGML Engine Extension Module",
    -1,
    ExtensionMethods
};

PyMODINIT_FUNC PyInit_ggml_extension(void) {
    return PyModule_Create(&ggmlmodule);
}
