# from distutils.core import setup, Extension
from setuptools import setup, Extension

module = Extension('healcorr',
                   define_macros=[('MAJOR_VERSION', '1'),
                                  ('MINOR_VERSION', '0')],
                   sources=['src/healcorr.c'],
                   extra_compile_args=['-fopenmp -O3'],
                   extra_link_args=['-fopenmp -O3'])

setup(name='healcorr',
      version='1.0',
      description='This is a demo package',
      author='Pier Fiedorowicz',
      author_email='pierfied@email.arizona.edu',
      ext_modules=[module],
      packages=['healcorr'],
      zip_safe=False)
