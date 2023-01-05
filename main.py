from func import *

'''
filter_types = ['elevDiff', 'elevDiff_inverse', 
                'slope', 'slope_inverse', 
                'dist', 'dist_inverse', 
                'elevation', 'elevation_inverse', 
                'height', 'height_inverse', 
                'floors', 'floors_inverse', 
                'year', 'year_inverse']

When setting more_src and more_can, run a profiler tool such as (pyinstrument main.py) to locate limiting factors
'''

def nearest_neighbor(canidates, locations, algorithm='KDTree', dist_metric='euclidean', nearest=4, rad_dist=0, filter_type='dist', more_src=0, more_can=0):
    # Use for testing a graph making
    # src_point = 3

    # Load data and reset index
    df_buildings = pd.read_csv(canidates).copy().reset_index(drop=True)
    df_locations = pd.read_csv(locations).copy().reset_index(drop=True)

    # Convert to list and add more data if wanted
    starting_locations = add_more_src(df_locations, more_src)
    building_locations = add_more_can(df_buildings, more_can)
    building_cords = list(zip(building_locations['X'], building_locations['Y']))
    location_cords = list(zip(starting_locations['X'], starting_locations['Y']))

    # Algorithm decision
    kd = BallTree(building_cords, metric=dist_metric) if algorithm=='BallTree' else KDTree(building_cords, metric=dist_metric)
    dist_array, idx_array = kd.query(location_cords, k=len(df_buildings))

    # List of building canidates and output distance
    for i,_ in enumerate(idx_array):
        src_point = i

        # Create defualtdict to identify duplicates
        val_map = identify_dups(dist_array, idx_array, src_point)
        dup_list = list_dups(val_map)
        
        # Elevation, slope, and 2D distances - returns dict where values = [elevation difference, slope distance to top, 2D distance, Elevation, Height, Floors, Year built]
        near, rad = all_info(val_map, src_point, starting_locations, df_buildings, rad_dist, nearest)
        # near_graph, _ = all_info(val_map, src_point, starting_locations, df_buildings, rad_dist, nearest=nearest+1) #for graphs

        # convert to DataFrame
        df_near = to_df(near, df_buildings, rad_dist, nearest, dup_list)
        df_rad = to_df(rad, df_buildings, rad_dist, nearest, dup_list)
        # df_near_graph = to_df(near_graph, df_buildings, rad_dist, nearest, dup_list) #for graphs

        # Filter DataFrame given filter_type
        near_filter = filter_by(df_near, filter_type)
        rad_filter = filter_by(df_rad, filter_type)
        # graph_filter = filter_by(df_near_graph, filter_type) #for graphs

        # Save each src_point as .csv
        save_files(near_filter, rad_filter, src_point, df_locations)  

    # create_graphs(starting_locations, building_locations, near_filter, rad_filter, rad_dist, nearest, graph_filter, src_point) #for graphs

if __name__ == "__main__":
    nearest_neighbor('./buildings.csv', './queries.csv', nearest=3, filter_type='dist', rad_dist=0)