#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>

/* Forward declaration of engine_forward from ggml */
extern int engine_forward(void *ctx, void *tensor);

static PyObject* py_engine_forward(PyObject *self, PyObject *args) {
    PyObject *py_ctx = NULL;
    PyObject *py_tensor = NULL;
    
    if (!PyArg_ParseTuple(args, "OO", &py_ctx, &py_tensor)) {
        return NULL;
    }
    
    /* In real implementation, these would be converted to actual pointers */
    void *ctx = PyLong_AsVoidPtr(py_ctx);
    void *tensor = PyLong_AsVoidPtr(py_tensor);
    
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    int result = engine_forward(ctx, tensor);
    
    return PyLong_FromLong(result);
}

static PyMethodDef ExtensionMethods[] = {
    {"engine_forward", py_engine_forward, METH_VARARGS,
     "Call engine_forward with context and tensor pointers."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef ggmlextension = {
    PyModuleDef_HEAD_INIT,
    "ggml_extension",
    "GGML Engine Forward Extension Module",
    -1,
    ExtensionMethods
};

PyMODINIT_FUNC PyInit_ggml_extension(void) {
    return PyModule_Create(&ggmlextension);
}
