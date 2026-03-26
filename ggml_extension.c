#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>

// Forward declaration of the engine function we are wrapping
extern int engine_forward(void* ctx, void* tensor);

static PyObject* py_engine_forward(PyObject* self, PyObject* args) {
    Py_buffer buffer;
    
    if (!PyArg_ParseTuple(args, "y#", &buffer.buf, &buffer.len)) {
        return NULL;
    }
    
    // Call the underlying engine_forward function
    int result = engine_forward(NULL, buffer.buf);
    
    PyBuffer_Release(&buffer);
    
    return PyLong_FromLong(result);
}

static PyMethodDef ExtensionMethods[] = {
    {"engine_forward", py_engine_forward, METH_VARARGS,
     "Call the GGML engine forward pass on input data."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef ggml_extension_module = {
    PyModuleDef_HEAD_INIT,
    "ggml_extension",
    "GGML Engine C Extension",
    -1,
    ExtensionMethods
};

PyMODINIT_FUNC PyInit_ggml_extension(void) {
    PyObject* m;
    
    m = PyModule_Create(&ggml_extension_module);
    if (m == NULL) {
        return NULL;
    }
    
    return m;
}
