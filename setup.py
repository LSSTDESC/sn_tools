from setuptools import setup

setup(
    name='sn_tools',
    version=__version__,
    description='Set of tools used to run SN pipeline',
    url='http://github.com/lsstdesc/sn_tools',
    author='Philippe Gris',
    author_email='philippe.gris@clermont.in2p3.fr',
    license='BSD',
    packages=['sn_tools'],
    python_requires='>=3.5',
    zip_safe=False
)
