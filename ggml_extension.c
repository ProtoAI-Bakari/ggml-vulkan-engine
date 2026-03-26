#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>

// Mock engine_forward implementation - replace with actual GGML forward pass
static int engine_forward(float* input, float* weights, float* output,
                          int batch_size, int input_dim, int hidden_dim) {
    // Simple matrix multiplication as placeholder for GGML forward pass
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < hidden_dim; i++) {
            float sum = 0.0f;
            for (int j = 0; j < input_dim; j++) {
                sum += input[b * input_dim + j] * weights[j * hidden_dim + i];
            }
            output[b * hidden_dim + i] = sum;
        }
    }
    return 0;
}

// Python wrapper function
static PyObject* py_engine_forward(PyObject* self, PyObject* args) {
    Py_buffer input_buf, weights_buf, output_buf;
    int batch_size, input_dim, hidden_dim;

    if (!PyArg_ParseTuple(args, "y*y*y*iii", &input_buf, &weights_buf, &output_buf,
                          &batch_size, &input_dim, &hidden_dim)) {
        return NULL;
    }

    if (input_buf.len != batch_size * input_dim * sizeof(float)) {
        PyErr_SetString(PyExc_ValueError, "Invalid input buffer size");
        PyBuffer_Release(&input_buf);
        PyBuffer_Release(&weights_buf);
        PyBuffer_Release(&output_buf);
        return NULL;
    }

    if (weights_buf.len != input_dim * hidden_dim * sizeof(float)) {
        PyErr_SetString(PyExc_ValueError, "Invalid weights buffer size");
        PyBuffer_Release(&input_buf);
        PyBuffer_Release(&weights_buf);
        PyBuffer_Release(&output_buf);
        return NULL;
    }

    if (output_buf.len != batch_size * hidden_dim * sizeof(float)) {
        PyErr_SetString(PyExc_ValueError, "Invalid output buffer size");
        PyBuffer_Release(&input_buf);
        PyBuffer_Release(&weights_buf);
        PyBuffer_Release(&output_buf);
        return NULL;
    }

    int result = engine_forward((float*)input_buf.buf, (float*)weights_buf.buf,
                                (float*)output_buf.buf, batch_size, input_dim, hidden_dim);

    PyBuffer_Release(&input_buf);
    PyBuffer_Release(&weights_buf);
    PyBuffer_Release(&output_buf);

    return PyLong_FromLong(result);
}

static PyMethodDef GGMLMethods[] = {
    {"engine_forward", py_engine_forward, METH_VARARGS,
     "Run forward pass through GGML model"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef ggml_module = {
    PyModuleDef_HEAD_INIT,
    "ggml_extension",
    "GGML Engine Forward Pass Extension",
    -1,
    GGMLMethods
};

PyMODINIT_FUNC PyInit_ggml_extension(void) {
    return PyModule_Create(&ggml_module);
}
