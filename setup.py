# from distutils.core import setup, Extension
from setuptools import setup, Extension

module = Extension('healcorr',
                    define_macros = [('MAJOR_VERSION', '1'),
                                     ('MINOR_VERSION', '0')],
                    # include_dirs = ['/usr/local/include'],
                    # libraries = ['tcl83'],
                    # library_dirs = ['/usr/local/lib'],
                    sources = ['src/healcorr.c'],
                   extra_compile_args=['-fopenmp'],
                   extra_link_args=['-fopenmp'])

setup (name = 'healcorr',
       version = '1.0',
       description = 'This is a demo package',
       author = 'Pier Fiedorowicz',
       author_email = 'pierfied@email.arizona.edu',
       ext_modules = [module],
       packages=['healcorr'],
       zip_safe=False)