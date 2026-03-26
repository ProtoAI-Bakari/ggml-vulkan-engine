#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>

// Forward declaration of the engine function from ggml library
extern int engine_forward(void *ctx, void *tensor);

static PyObject* ggml_engine_forward(PyObject *self, PyObject *args) {
    void *ctx = NULL;
    void *tensor = NULL;
    
    if (!PyArg_ParseTuple(args, "pp", &ctx, &tensor)) {
        return NULL;
    }
    
    int result = engine_forward(ctx, tensor);
    
    return PyLong_FromLong(result);
}

static PyMethodDef GGMLMethods[] = {
    {"engine_forward", ggml_engine_forward, METH_VARARGS,
     "Call the engine forward pass."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef ggmlmodule = {
    PyModuleDef_HEAD_INIT,
    "ggml_extension",
    "GGML Engine Extension Module",
    -1,
    GGMLMethods
};

PyMODINIT_FUNC PyInit_ggml_extension(void) {
    return PyModule_Create(&ggmlmodule);
}
