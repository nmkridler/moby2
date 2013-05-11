Whale Detection Challenge Code

Uses the following python packages
Numpy/Scipy
Pandas
OpenCV
Sci-kit Learn

Run in the following order
genTrainMetrics.py
genTestMetrics.py
ordering.py
submission.py

User must edit the baseDir variable in each of the above files.
The code assumes a folder structure where the competition data
is in a folder named data and lives in the same directory as
moby.

HACK HACK HACK
padded to 59 bins (I think this was what it was in the last version)
I used driver to create submission
make sure the templateManager points to the old train.csv and the old train data

TO DO:
figure out the right way to deal with variable widths
figure out how to use the ordering
add other metrics back in, probably okay to use the high freq bar 