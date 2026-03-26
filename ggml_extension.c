#include <Python.h>

// Stub implementation of engine_forward for testing
void engine_forward(void* input, size_t len) {
    // Placeholder - actual implementation depends on ggml backend
}

static PyObject* py_engine_forward(PyObject* self, PyObject* args) {
    const char* data;
    Py_ssize_t length;
    if (!PyArg_ParseTuple(args, "y#", &data, &length)) {
        return NULL;
    }
    engine_forward((void*)data, length);
    Py_RETURN_NONE;
}

static PyMethodDef ModuleMethods[] = {
    {"forward", py_engine_forward, METH_VARARGS, "Call engine_forward with binary data"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "ggml_extension",
    NULL,
    -1,
    ModuleMethods
};

PyMODINIT_FUNC PyInit_ggml_extension(void) {
    return PyModule_Create(&moduledef);
}