#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <numpy/arrayobject.h>

// Forward declaration of the engine function we are wrapping
typedef struct {
    float* weights;
    int size;
} EngineContext;

static EngineContext g_engine_ctx = {NULL, 0};

// Mock implementation - replace with actual engine_forward from your library
static float engine_forward(float* input, int input_size) {
    printf("engine_forward called with input_size=%d\n", input_size);
    return 0.0f;
}

// Python wrapper function
static PyObject* py_engine_forward(PyObject* self, PyObject* args) {
    PyArrayObject* input_array = NULL;
    
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &input_array)) {
        PyErr_SetString(PyExc_TypeError, "Expected numpy array as argument");
        return NULL;
    }
    
    if (PyArray_NDIM(input_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Input must be 1D array");
        return NULL;
    }
    
    float* input_data = (float*)PyArray_DATA(input_array);
    int input_size = (int)PyArray_SIZE(input_array);
    
    float result = engine_forward(input_data, input_size);
    
    return PyFloat_FromDouble((double)result);
}

// Initialize function for the module
static PyMethodDef ExtensionMethods[] = {
    {"forward", py_engine_forward, METH_VARARGS, "Call engine_forward on input data"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef extensionmodule = {
    PyModuleDef_HEAD_INIT,
    "ggml_extension",
    "GGML Engine Forward Extension",
    -1,
    ExtensionMethods
};

PyMODINIT_FUNC PyInit_ggml_extension(void) {
    import_array();
    return PyModule_Create(&extensionmodule);
}
