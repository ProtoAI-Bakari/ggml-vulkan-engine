#include <Python.h>
#include <stdio.h>

// Forward declaration of engine_forward from the engine library
extern int engine_forward(void* ctx, const void* input, size_t input_size, 
                          void* output, size_t output_size);

static PyObject* py_engine_forward(PyObject* self, PyObject* args) {
    Py_buffer input_buf;
    Py_buffer output_buf;
    
    if (!PyArg_ParseTuple(args, "s*s*", &input_buf, &output_buf)) {
        return NULL;
    }
    
    int result = engine_forward(NULL, 
                                input_buf.buf, 
                                input_buf.len,
                                output_buf.buf, 
                                output_buf.len);
    
    PyBuffer_Release(&input_buf);
    PyBuffer_Release(&output_buf);
    
    return PyLong_FromLong(result);
}

static PyMethodDef ExtensionMethods[] = {
    {"engine_forward", py_engine_forward, METH_VARARGS, 
     "Call engine_forward with input/output buffers"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "ggml_extension",
    "GGML Engine Forward Extension",
    -1,
    ExtensionMethods
};

PyMODINIT_FUNC PyInit_ggml_extension(void) {
    return PyModule_Create(&moduledef);
}
