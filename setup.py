import sys
import os

if sys.platform == 'darwin':
    home = os.path.abspath(os.path.join(os.sep, 'Users', 'Home'))
    phd = os.path.join(home, 'Checkouts', 'PhD')
    data = os.path.join(phd, 'data')
    src = os.path.join(phd, 'src')

elif sys.platform == 'linux':
    home = os.path.abspath(os.path.join(os.sep, 'home', 'janjaapmeijer'))
    phd = os.path.join(home, 'Checkouts', 'PhD')
    data = os.path.join(phd, 'data')
    src = os.path.join(phd, 'src')