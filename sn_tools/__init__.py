from .version import __version__
import os

# requested infos for throughputs
throughputs_dir = os.path.join(os.getenv("PWD"), "throughputs")
if not os.path.isdir(throughputs_dir):
    cmd = 'git clone https://github.com/lsst/throughputs'
    os.system(cmd)

os.environ['LSST_THROUGHPUTS_BASELINE'] = '{}/{}'.format(
    throughputs_dir, 'baseline')
os.environ['THROUGHPUTS_DIR'] = throughputs_dir

print('Reading throughputs from',
      os.environ['LSST_THROUGHPUTS_BASELINE'], os.environ['THROUGHPUTS_DIR'])
