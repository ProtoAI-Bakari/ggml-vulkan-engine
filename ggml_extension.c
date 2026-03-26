#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdint.h>
#include <stdlib.h>

// External function declarations from libggml_llama_full.so
extern void* llama_engine_init(const char* model_path);
extern int llama_engine_forward(void* engine, float* input_tokens, int num_tokens, float* output_logits, int vocab_size);
extern void llama_engine_reset_kv(void* engine);
extern void llama_engine_free(void* engine);

static PyObject* py_engine = NULL;

static PyObject* engine_load_gguf(PyObject* self, PyObject* args) {
    const char* model_path;
    if (!PyArg_ParseTuple(args, "s", &model_path)) {
        return NULL;
    }
    
    void* engine = llama_engine_init(model_path);
    if (engine == NULL) {
        Py_RETURN_NONE;
    }
    
    py_engine = PyLong_FromVoidPtr(engine);
    Py_XINCREF(py_engine);
    return py_engine;
}

static PyObject* engine_forward(PyObject* self, PyObject* args) {
    PyObject* tokens_obj;
    int vocab_size;
    
    if (!PyArg_ParseTuple(args, "Oi", &tokens_obj, &vocab_size)) {
        return NULL;
    }
    
    if (py_engine == NULL || py_engine == Py_None) {
        PyErr_SetString(PyExc_RuntimeError, "Engine not loaded. Call engine_load_gguf first.");
        return NULL;
    }
    
    // Convert numpy array to raw pointer
    PyArrayObject* tokens_array = (PyArrayObject*)PyArray_FROM_OTF(tokens_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    if (tokens_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Input must be a numpy array of floats");
        return NULL;
    }
    
    float* tokens_data = (float*)PyArray_DATA(tokens_array);
    npy_intp num_tokens = PyArray_SIZE(tokens_array);
    
    // Allocate output buffer
    npy_intp out_shape[] = {vocab_size};
    PyArrayObject* logits_array = (PyArrayObject*)PyArray_SimpleNew(1, out_shape, NPY_FLOAT32);
    if (logits_array == NULL) {
        Py_DECREF(tokens_array);
        PyErr_NoMemory();
        return NULL;
    }
    
    float* logits_data = (float*)PyArray_DATA(logits_array);
    
    void* engine = PyLong_AsVoidPtr(py_engine);
    int result = llama_engine_forward(engine, tokens_data, (int)num_tokens, logits_data, vocab_size);
    
    Py_DECREF(tokens_array);
    
    if (result != 0) {
        Py_DECREF(logits_array);
        PyErr_Format(PyExc_RuntimeError, "Forward failed with code %d", result);
        return NULL;
    }
    
    return (PyObject*)logits_array;
}

static PyObject* engine_reset_kv(PyObject* self, PyObject* args) {
    if (py_engine == NULL || py_engine == Py_None) {
        PyErr_SetString(PyExc_RuntimeError, "Engine not loaded. Call engine_load_gguf first.");
        return NULL;
    }
    
    void* engine = PyLong_AsVoidPtr(py_engine);
    llama_engine_reset_kv(engine);
    Py_RETURN_NONE;
}

static PyObject* engine_free(PyObject* self, PyObject* args) {
    if (py_engine != NULL && py_engine != Py_None) {
        void* engine = PyLong_AsVoidPtr(py_engine);
        llama_engine_free(engine);
        Py_XDECREF(py_engine);
        py_engine = NULL;
    }
    Py_RETURN_NONE;
}

static PyMethodDef ExtensionMethods[] = {
    {"engine_load_gguf", engine_load_gguf, METH_VARARGS, "Load GGUF model"},
    {"engine_forward", engine_forward, METH_VARARGS, "Run forward pass"},
    {"engine_reset_kv", engine_reset_kv, METH_VARARGS, "Reset KV cache"},
    {"engine_free", engine_free, METH_VARARGS, "Free engine resources"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "ggml_extension",
    "GGML vLLM Engine Extension",
    -1,
    ExtensionMethods
};

PyMODINIT_FUNC PyInit_ggml_extension(void) {
    import_array();
    return PyModule_Create(&moduledef);
}