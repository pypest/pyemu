# Remove the temp directory and then create a fresh one
import os
import shutil
import pyemu

nbdir = os.path.join('..', 'examples')

# -- make working directories
ddir = os.path.join(nbdir, 'data')
if os.path.isdir(ddir):
    shutil.rmtree(ddir)
os.mkdir(ddir)

tempdir = os.path.join('.', 'temp')
if os.path.isdir(tempdir):
    shutil.rmtree(tempdir)
os.mkdir(tempdir)

testdir = os.path.join('.', 'temp', 'Notebooks')
if os.path.isdir(testdir):
    shutil.rmtree(testdir)
os.mkdir(testdir)


def get_notebooks():
    return [f for f in os.listdir(nbdir) if f.endswith('.ipynb') and not "notest" in f]

def run_notebook(fn):
    #pth = os.path.join(nbdir, fn)
    pth = fn
    cmd = 'jupyter ' + 'nbconvert ' + \
          '--ServerApp.iopub_data_rate_limit=1e10 ' + \
          '--ExecutePreprocessor.kernel_name=python ' + \
          '--ExecutePreprocessor.timeout=6000 ' + '--to ' + 'notebook ' + \
          '--execute ' + '{} '.format(pth) + \
          '--output-dir ' + '{} '.format(testdir) + \
          '--output ' + '{}'.format(fn)
    
    #ival = os.system(cmd)
    ival = pyemu.os_utils.run(cmd,cwd=nbdir)
    print(ival,cmd)
    assert ival == 0 or ival is None, 'could not run {}'.format(fn)


def test_notebooks_test():
    files = get_notebooks()

    for fn in files:
        yield run_notebook, fn

if __name__ == '__main__':
    shutil.copy2(os.path.join("..","examples","helpers.py"),"helpers.py")
    files = get_notebooks()
    for fn in files:    
        run_notebook(fn)
