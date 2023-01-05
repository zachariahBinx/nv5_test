# Nearest Neighbor Coding Problem

## Created on following specs:
* Windows 10
* NVIDIA GTX 2080
* Intel(R) Core(TM) i-9700K CPU @ 3.60GHz
* 16GB RAM

## Setting Up Environment on Windows 10 

Should work similary on Linux but has been untested
1. Download [Anaconda](https://www.anaconda.com/)
2. Launch conda terminal and copy/paste the following:

  `conda create -n nv5_test python=3.8 -y`
  
  `conda activate nv5_test`
  
  `conda install -c anaconda scikit-learn==1.0.2 -y`
  
  `conda install -c anaconda pandas==1.5.2 -y`
  
  `conda install -c conda-forge matplotlib==3.6.2 -y`
  
  `pip install names`

#Using the code
There are 2 required parameters the user must have to run:

1. canidates (buildings.csv)
2. locations/src points (queires.csv)

The remaining 7 parameters can be changed based on users preference.
1. `algorithm` - [scikit learn nearest neighbor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html) [KDTree](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html) or [BallTree](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html)

2. `dist_metric` - [distance metric](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.distance_metrics.html#sklearn.metrics.pairwise.distance_metrics)

3. `nearest` - number of nearest neighbors wanted

4. `rad_dist` - nearest neighbors within a given radius distance (feet)

5. `filter_type` - both with nearest and rad_dist results, filter by ascending or descending of the following:
  `[elevDiff, elevDiff_inverse, slope, slope_inverse, dist, dist_inverse, elevation, elevation_inverse, height, height_inverse, floors, floors_inverse, year, year_inverse]`
  
6. `more_src` - Number of entries to add to locations/src points - used for profiling and discovering limiting factors
 
7. `more_can` - Number of entries to add to canidates - used for profiling and discovering limiting factors
  
In order to run only one location/src point through user must:
1. comment in line 15

2. comment out lines 32-33

3. highlight and shift-tab lines 35-54

To use the graphing features:
1. follow the 3 steps above

2. comment in lines 41, 46, 51, and 56

3. best to change `nearest` to something like 3 and `rad_dist` to 1000-1500
4. may need to adjust `xlim` and `ylim` within the `create_graph` function within the `func.py` file
