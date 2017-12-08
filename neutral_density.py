import sys
import os

def write_dict(dictionary, path, filename):
    import pickle
    with open(os.path.join(path, filename + '.pkl'), 'wb') as f:
        pickle.dump(dictionary, f, pickle.HIGHEST_PROTOCOL)
def read_dict(path, filename):
    import pickle
    with open(os.path.join(path, filename), 'rb') as f:
        return pickle.load(f)

# INITIALISE
if sys.platform == 'darwin':
    root = os.path.abspath(os.path.join(os.sep, 'Users', 'Home', 'Documents', 'Jobs', 'IMAS'))


# read ctd data
dict_ctd = read_dict(os.path.join(root, 'Data', 'Voyages', 'SS9802', 'ctd'), 'ss9802_ctd.pkl')

# read transect data
dict_transect = read_dict(os.path.join(root, 'Analysis', 'SS9802', 'data'), 'ss9802_transects.pkl')


# TRANSECT INFORMATION
transects = {1: list(range(2, 10)), 2: list(reversed(range(10, 18))), 3: list(range(18,27)),
             4: list(reversed(range(26, 34))), 5: list(range(36, 46)), 6: list(reversed(range(46, 57))),
             7: list(range(56, 65)), 8: list(range(68, 76)), 9: list(reversed(range(76, 84))),
             10: list(range(84, 91)), 11: list(reversed([93, 92] + list(range(94, 101))))}

