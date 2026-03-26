#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>

// Forward declaration of the engine function we are wrapping
extern void engine_forward(void *ctx, float *input, float *output, int size);

static PyObject* py_engine_forward(PyObject *self, PyObject *args) {
    int size;
    Py_buffer input_buf, output_buf;
    
    if (!PyArg_ParseTuple(args, "s*s*i", &input_buf, &output_buf, &size)) {
        return NULL;
    }
    
    // Call the underlying engine_forward function
    engine_forward(NULL, (float*)input_buf.buf, (float*)output_buf.buf, size);
    
    PyBuffer_Release(&input_buf);
    PyBuffer_Release(&output_buf);
    
    Py_RETURN_NONE;
}

static PyObject* py_engine_forward_simple(PyObject *self, PyObject *args) {
    int size = 1024;
    
    if (!PyArg_ParseTuple(args, "|i", &size)) {
        return NULL;
    }
    
    printf("engine_forward called with size=%d\n", size);
    
    Py_RETURN_NONE;
}

static PyMethodDef ExtensionMethods[] = {
    {"forward", py_engine_forward, METH_VARARGS, "Call engine_forward with buffers"},
    {"forward_simple", py_engine_forward_simple, METH_VARARGS, "Simple forward call"},
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
