'''
    config.py

contains variables used in modeling function 
to ensure reproducability, and could be 
modified to allow for different outcomes


'''


from pathlib import Path

#
#	Paths
#
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "ai4i2020.csv"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

#
#	Constants
#
random_state = 42
test_size = 0.2
cv_folds = 5
n_splits = 5
STATE = {} #Holds dict of variables used and retrieved amongst multiple
	   #code files, was a different implementation and i wanted to
	   #assess the viability of this, would scale poorly with code
	   #complexity i think, nonetheless, interesting to experiment
	   #with
