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
    
    * If pip errors to install names, you can cut and paste the `names` and `names-0.3.0.dist-info` folders into your env sitepackages directory (C:/users/zlesl/anaconda3/envs/nv5_test/Lib/site-packages/)
  
    `git clone https://github.com/zachariahBinx/nv5_test`
  
3. Close the terminal

### Using your favorite IDE (vs code)

1. Launch a bash terminal instance

2. Activate your env

3. Run main.py

# Using the code

### There are 2 required parameters the user must have to run:

1. canidates (buildings.csv)
2. locations/src points (queires.csv)

### The remaining 7 parameters can be changed based on users preference.

1. `algorithm` - [scikit learn nearest neighbor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html) [KDTree](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html) or [BallTree](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html)

2. `dist_metric` - [distance metric](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.distance_metrics.html#sklearn.metrics.pairwise.distance_metrics)

3. `nearest` - number of nearest neighbors wanted

4. `rad_dist` - nearest neighbors within a given radius distance (feet)

5. `filter_type` - both with nearest and rad_dist results, filter by ascending or descending of the following:
  `[elevDiff, elevDiff_inverse, slope, slope_inverse, dist, dist_inverse, elevation, elevation_inverse, height, height_inverse, floors, floors_inverse, year, year_inverse]`
  
6. `more_src` - Number of entries to add to locations/src points - used for profiling and discovering bottle necks
 
7. `more_can` - Number of entries to add to canidates - used for profiling and discovering bottle necks
  
### In order to run only one location/src point through user must:

1. Comment in line 22

2. Comment out lines 39-40

3. Highlight and shift-tab lines 32-61

### To use the graphing features:

1. Follow the 3 steps above

2. Comment in lines 48, 53, 58, and 63

3. Best to change `nearest` to something like 3 and `rad_dist` to 1000-1500

4. May need to adjust `xlim` and `ylim` within the `create_graph` function within the `func.py` file
