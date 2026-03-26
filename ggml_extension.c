#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>

// Forward declaration of the engine_forward function from ggml
extern int engine_forward(void *ctx, void *tensor);

static PyObject* py_engine_forward(PyObject *self, PyObject *args) {
    void *ctx = NULL;
    void *tensor = NULL;
    
    if (!PyArg_ParseTuple(args, "nn", &ctx, &tensor)) {
        return NULL;
    }
    
    int result = engine_forward(ctx, tensor);
    return PyLong_FromLong(result);
}

static PyObject* py_init_ggml(PyObject *self, PyObject *args) {
    // Initialize ggml context
    printf("ggml_extension initialized\n");
    Py_RETURN_NONE;
}

static PyMethodDef GgmlMethods[] = {
    {"engine_forward", py_engine_forward, METH_VARARGS,
     "Forward pass through the engine"},
    {"init_ggml", py_init_ggml, METH_NOARGS,
     "Initialize ggml context"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef ggmlmodule = {
    PyModuleDef_HEAD_INIT,
    "ggml_extension",
    "GGML Engine Extension Module",
    -1,
    GgmlMethods
};

PyMODINIT_FUNC PyInit_ggml_extension(void) {
    return PyModule_Create(&ggmlmodule);
}
