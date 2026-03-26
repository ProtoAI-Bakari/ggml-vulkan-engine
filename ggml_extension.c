#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>

// Forward declaration of the engine_forward function from ggml
extern int engine_forward(void *ctx, void *tensor);

static PyObject* py_engine_forward(PyObject *self, PyObject *args) {
    void *ctx = NULL;
    void *tensor = NULL;
    
    if (!PyArg_ParseTuple(args, "pp", &ctx, &tensor)) {
        return NULL;
    }
    
    int result = engine_forward(ctx, tensor);
    
    return PyLong_FromLong(result);
}

static PyMethodDef ExtensionMethods[] = {
    {"engine_forward", py_engine_forward, METH_VARARGS,
     "Call the engine_forward function from ggml."},
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
