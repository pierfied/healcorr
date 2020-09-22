from setuptools import setup
from cmake_setuptools import CMakeExtension, CMakeBuildExt

module = CMakeExtension('healcorr')

setup(name='healcorr',
      version='1.0',
      description='Angular correlation function calculator for HEALPix maps.',
      author='Pier Fiedorowicz',
      author_email='pierfied@email.arizona.edu',
      ext_modules=[module],
      cmdclass={'build_ext': CMakeBuildExt},
      packages=['healcorr'],
      zip_safe=False)
