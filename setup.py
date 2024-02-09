from setuptools import setup

# get the version here
pkg_vars = {}

with open("version.py") as fp:
    exec(fp.read(), pkg_vars)

setup(
    name='sn_tools',
    version= pkg_vars['__version__'],
    description='Set of tools used to run SN pipeline',
    url='http://github.com/lsstdesc/sn_tools',
    author='Philippe Gris',
    author_email='philippe.gris@clermont.in2p3.fr',
    license='BSD',
    packages= ['sn_tools','sn_script_input'],
    # All files from folder sn_script_input
    package_data={'sn_script_input': ['*.txt']},
    python_requires='>=3.5',
    zip_safe=False
)
