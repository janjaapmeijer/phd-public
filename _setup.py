import sys
import os

if sys.platform == 'darwin':
    homedir = os.path.abspath(os.path.join(os.sep, 'Users', 'Home'))
    phddir = os.path.join(homedir, 'Checkouts', 'PhD')
    datadir = os.path.join(phddir, 'data')
    srcdir = os.path.join(phddir, 'src')
    outdir = os.path.join(phddir, 'out')

elif sys.platform == 'linux' or sys.platform == 'linux2':
    homedir = os.path.abspath(os.path.join(os.sep, 'home', 'janjaapmeijer'))
    phddir = os.path.join(homedir, 'Checkouts', 'PhD')
    datadir = os.path.join(phddir, 'data')
    srcdir = os.path.join(phddir, 'src')
    outdir = os.path.join(phddir, 'out')