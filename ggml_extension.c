#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>

/* Forward declaration of the engine function */
extern int engine_forward(void *ctx, float *input, float *output, size_t n);

static PyObject* ggml_forward(PyObject* self, PyObject* args) {
    Py_buffer input_buf;
    Py_buffer output_buf;
    int result;

    if (!PyArg_ParseTuple(args, "s*s*i", &input_buf, &output_buf, &result)) {
        return NULL;
    }

    /* Call the underlying engine_forward function */
    int ret = engine_forward(NULL, (float*)input_buf.buf, (float*)output_buf.buf, input_buf.len / sizeof(float));

    PyBuffer_Release(&input_buf);
    PyBuffer_Release(&output_buf);

    return PyLong_FromLong(ret);
}

static PyObject* ggml_init(PyObject* self, PyObject* args) {
    printf("ggml_extension initialized\n");
    Py_RETURN_NONE;
}

static PyMethodDef GgmlMethods[] = {
    {"forward", ggml_forward, METH_VARARGS, "Run forward pass through the model"},
    {"init", ggml_init, METH_NOARGS, "Initialize ggml extension"},
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
