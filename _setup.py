import sys
import os

if sys.platform == 'darwin':
    homedir = os.path.abspath(os.path.join(os.sep, 'Users', 'Home'))
    phddir = os.path.join(homedir, 'Checkouts', 'PhD')
    datadir = os.path.join(phddir, 'data')
    srcdir = os.path.join(phddir, 'src')

elif sys.platform == 'linux':
    homedir = os.path.abspath(os.path.join(os.sep, 'home', 'janjaapmeijer'))
    phddir = os.path.join(homedir, 'Checkouts', 'PhD')
    datadir = os.path.join(phddir, 'data')
    srcdir = os.path.join(phddir, 'src')