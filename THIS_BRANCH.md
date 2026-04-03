This is the legacy code originally used in the arXiv paper. Affected by several important issues:

* inference is “teacher-enforced”: basically it cheats
* based on FARGO3D simulations
* training/test data is not reproducible, scripts for converting from FARGO3D format to numpy not available; parameter mismatch between paper plot and actual simulations.
* under-utilizes the training data: uses only the first 4% of the simulations for training: convert_to_npy.py sliding windows were nearly overlapping, using only ~4% of each simulation's timeline. Fixed on this branch with WINDOW_STRIDE=51; dev still has it plus a bug  where all 20 windows share identical output targets.
* Weird forecasting decisions: seq 2 seq; predicts highly spaced future frames 
* Loss was incoherent. L2 was on a different order of magnitude compared with MAE, so MAE was effectively ignored

Do not use this for production. All these issues have been fixed in branch `dev`.
