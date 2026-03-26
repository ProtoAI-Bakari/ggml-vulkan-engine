#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>

// Function pointer type for engine_forward
typedef int (*engine_forward_fn)(void *ctx, int n_tokens, void *tokens, void *positions, void *logits);

static void *engine_lib = NULL;
static engine_forward_fn forward_func = NULL;

static PyObject* ggml_init(PyObject *self, PyObject *args) {
    // Load the shared library containing engine_forward
    engine_lib = dlopen("./libggml_llama_gguf.so", RTLD_NOW);
    if (!engine_lib) {
        engine_lib = dlopen("/home/z/AGENT/libggml_llama_gguf.so", RTLD_NOW);
    }
    
    if (!engine_lib) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to load libggml_llama_gguf.so");
        return NULL;
    }
    
    forward_func = (engine_forward_fn)dlsym(engine_lib, "llama_engine_forward");
    if (!forward_func) {
        dlclose(engine_lib);
        engine_lib = NULL;
        PyErr_Format(PyExc_RuntimeError, "Failed to find llama_engine_forward: %s", dlerror());
        return NULL;
    }
    
    Py_RETURN_TRUE;
}

static PyObject* ggml_engine_forward(PyObject *self, PyObject *args) {
    if (!forward_func) {
        PyErr_SetString(PyExc_RuntimeError, "Engine not initialized. Call init first.");
        return NULL;
    }
    
    void *ctx;
    int n_tokens;
    void *tokens;
    void *positions;
    void *logits;
    
    if (!PyArg_ParseTuple(args, "piPPP", &ctx, &n_tokens, &tokens, &positions, &logits)) {
        return NULL;
    }
    
    int result = forward_func(ctx, n_tokens, tokens, positions, logits);
    
    return PyLong_FromLong(result);
}

static PyObject* ggml_cleanup(PyObject *self, PyObject *args) {
    if (engine_lib) {
        dlclose(engine_lib);
        engine_lib = NULL;
        forward_func = NULL;
    }
    Py_RETURN_NONE;
}

static PyMethodDef GGMLMethods[] = {
    {"init", ggml_init, METH_NOARGS, "Initialize the GGML engine."},
    {"engine_forward", ggml_engine_forward, METH_VARARGS,
     "Call the engine forward pass."},
    {"cleanup", ggml_cleanup, METH_NOARGS, "Cleanup resources."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef ggmlmodule = {
    PyModuleDef_HEAD_INIT,
    "ggml_extension",
    "GGML Engine Extension Module", -1,
    GGMLMethods
};

PyMODINIT_FUNC PyInit_ggml_extension(void) {
    PyObject *m = PyModule_Create(&ggmlmodule);
    if (m == NULL)
        return NULL;
    return m;
}
