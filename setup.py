from setuptools import setup, Extension

module = Extension("ggml_extension",
                   sources=["ggml_extension.c"],
                   extra_compile_args=["-O2"])

setup(name="ggml_extension",
      version="1.0",
      description="GGML Engine Forward Pass Extension",
      ext_modules=[module])
