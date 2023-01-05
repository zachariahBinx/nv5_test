import itertools
import pandas as pd
import collections
import math
from sklearn.neighbors import KDTree, BallTree
import numpy as np
import matplotlib.pyplot as plt
import os
import names
import random
from datetime import date
import shutil

def add_more_src(df_src, more):
    df = pd.DataFrame(columns=['Name', 'X', 'Y', 'Elevation'])
    latLonMax = [max(df_src["X"]+(5280*2)), max(df_src["Y"]+(5280*2))]
    latLonMin = [min(df_src["X"]-(5280*2)), min(df_src["Y"]-(5280*2))]
    elevMinMax = [min(df_src["Elevation"]-(250)), max(df_src["Elevation"]+(250))]

    for x in range(more):
        lat = random.uniform(latLonMin[0], latLonMax[0])
        lon = random.uniform(latLonMin[1], latLonMax[1])
        elev = random.uniform(elevMinMax[0], elevMinMax[1])
        df.loc[x] = [names.get_full_name()] + [lat] + [lon] + [elev]

    return pd.concat([df_src, df], ignore_index=True)

def add_more_can(df_can, more):
    # leaving out rank and notes column, not entirely needed
    df = pd.DataFrame(columns=['Name', 'X', 'Y', 'Elevation', 'Height', 'Floors', 'Year'])
    latLonMax = [max(df_can["X"]+(5280*2)), max(df_can["Y"]+(5280*2))]
    latLonMin = [min(df_can["X"]-(5280*2)), min(df_can["Y"]-(5280*2))]
    elevMinMax = [min(df_can["Elevation"]-(250)), max(df_can["Elevation"]+(250))]
    heightMinMax = [min(df_can["Height"]), max(df_can["Height"]+(250))]
    floorsMinMax = [min(df_can["Floors"]), max(df_can["Floors"]+(20))]
    yearMinMax = [min(df_can["Year"]-(50)), date.today().year-1]

    for x in range(more):
        lat = random.uniform(latLonMin[0], latLonMax[0])
        lon = random.uniform(latLonMin[1], latLonMax[1])
        elev = random.uniform(elevMinMax[0], elevMinMax[1])
        height = random.randint(heightMinMax[0], heightMinMax[1])
        floor = random.randint(floorsMinMax[0], floorsMinMax[1])
        year = random.randint(yearMinMax[0], yearMinMax[1])
        df.loc[x] = [names.get_full_name()] + [lat] + [lon] + [elev] + [height] + [floor] + [year]

    return pd.concat([df_can, df], ignore_index=True)


def identify_dups(dist, idx, src_point):   
    # Convert to list and flatten
    idx_list = idx[src_point].tolist()
    dist_list = dist[src_point].tolist()
    idx_flat = list(itertools.chain(idx_list))
    dist_flat = list(itertools.chain(dist_list))

    # Convert to dict
    d = {k:v for k,v in zip(idx_flat, dist_flat)}  

    # Change dict k,v = dist:[val,] - opposite of common practice - allows duplicates to be identified as 1 nearest neighbor
    val_map = collections.defaultdict(list)
    for k,v in d.items():
        val_map[v].append(k)
    
    return val_map


def match_dict(d_near_rad, d_values):
    myDict = {}
    d_inverse = dict_inverse(d_values)
    for k,v in d_near_rad.items():
        for x,y in d_inverse.items():
            if x==k:
                myDict.update({x:y})

    return myDict


def dict_inverse(dictionary):
    return {v: k for k, l in dictionary.items() for v in l} 


def all_info(val_map, src_point, df_loc, df_canidates, rad_dist, nearest):
    d_elevDiff,d_top,d_elev,d_height,d_floor,d_year,d_X,d_Y,super_dict_near,super_dict_rad=collections.defaultdict(list),collections.defaultdict(list),collections.defaultdict(list),collections.defaultdict(list),collections.defaultdict(list),collections.defaultdict(list),collections.defaultdict(list),collections.defaultdict(list),collections.defaultdict(list),collections.defaultdict(list)
    
    for k,v in val_map.items():
        for val in v:
            evel = df_canidates["Elevation"].loc[val]
            d_elev[evel].append(val)

            height = df_canidates["Height"].loc[val]
            d_height[height].append(val)

            floor = df_canidates["Floors"].loc[val]
            d_floor[floor].append(val)

            year = df_canidates["Year"].loc[val]
            d_year[year].append(val)

            elevDiff = df_loc["Elevation"].loc[src_point] - df_canidates["Elevation"].loc[val]
            d_elevDiff[elevDiff].append(val)

            xLat = df_canidates["X"].loc[val]
            d_X[xLat].append(val)

            yLon = df_canidates["Y"].loc[val]
            d_Y[yLon].append(val)

            if df_loc["Elevation"].loc[src_point] < df_canidates["Elevation"].loc[val]:
                H = height + abs(elevDiff)
            elif df_loc["Elevation"].loc[src_point] > evel and df_loc["Elevation"].loc[src_point] < evel + height:
                H = height - abs(elevDiff)
            else:
                H = abs(elevDiff) - height

            hypo = math.sqrt((k**2) + (H**2)) #Distance from bottom of src to top of building
            d_top[hypo].append(val) #IMPORTANT!!! If buildings are in same location and top heights are different, idx will not be together

    '''NEAREAST DISTANCE'''
    d_near = dict_inverse({A:N for (A,N) in [x for x in val_map.items()][:nearest]})

    e_near = match_dict(d_near, d_elevDiff)
    t_near = match_dict(d_near, d_top)
    elev_near = match_dict(d_near, d_elev)
    height_near = match_dict(d_near, d_height)
    floor_near = match_dict(d_near, d_floor)
    year_near = match_dict(d_near, d_year)
    x_near = match_dict(d_near, d_X)
    y_near = match_dict(d_near, d_Y)


    list_dicts_near = [e_near, t_near, d_near, elev_near, height_near, floor_near, year_near, x_near, y_near]

    for d in list_dicts_near:
        for k,v in d.items():
            super_dict_near[k].append(v)

    '''RADIUS DISTANCE'''
    d_rad = dict_inverse({A:N for (A,N) in [x for x in val_map.items() if x[0] <= rad_dist]})

    e_rad = match_dict(d_rad, d_elevDiff)
    t_rad = match_dict(d_rad, d_top)
    elev_rad = match_dict(d_rad, d_elev)
    height_rad = match_dict(d_rad, d_height)
    floor_rad = match_dict(d_rad, d_floor)
    year_rad = match_dict(d_rad, d_year)
    x_rad = match_dict(d_rad, d_X)
    y_rad = match_dict(d_rad, d_Y)

    list_dicts_rad = [e_rad, t_rad, d_rad, elev_rad, height_rad, floor_rad, year_rad, x_rad, y_rad]

    for d in list_dicts_rad:
        for k,v in d.items():
            super_dict_rad[k].append(v)

    return super_dict_near, super_dict_rad


def list_dups(val_map):
    dup_list = []
    for k,v in val_map.items():
        if len(v) > 1:
            dup_list.append(v)

    return dup_list


def filter_by(df, filter):
    filter_split = filter.split("_")[0]
    list_of_filters = ['idx', 'Name', 'elevDiff', 'slope', 'dist', 'elevation', 'height', 'floors', 'year']
    for i,filter_types in enumerate(list_of_filters):
        if filter_split in filter_types:
            inverse = False if 'inverse' in filter else True
            return df.sort_values(by=[list_of_filters[i]], ascending=inverse)

    return df


def to_df(d, df_canidates, rad_dist, nearest, dup_list):
    df = pd.DataFrame(columns=['buildingIDX', 'Name', 'elevDiff', 'slope', 'dist', 'elevation', 'height', 'floors', 'year', 'X', 'Y', 'radiusDist', 'nearest'])
    for i,(k,v) in enumerate(d.items()):
        L = [k] + [df_canidates.loc[k]['Name']] + v + [rad_dist] + [nearest]
        df.loc[i] = L
    for x in dup_list:
        for y in x:
            idx = list(np.where(df["buildingIDX"] == y)[0])
            if idx:
                df.loc[idx[0],'dup'] = 'yes'

    return df


def create_graphs(df_src, df_can, df_filter_near, df_filter_rad, rad_dist, nearest, df_graph, src_point):
    fig1,ax1 = plt.subplots()
    ax1.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    ax1.scatter(x=df_src.loc[src_point]['X'], y=df_src.loc[src_point]['Y'], color='red', label='src', zorder=1)
    ax1.scatter(x=df_can['X'], y=df_can['Y'], color='orange', label='canidates', zorder=2)
    ax1.scatter(x=df_filter_near['X'], y=df_filter_near['Y'], color='blue', label=f'nearest {nearest}', zorder=3)
    src_xy = [df_src.loc[src_point]['X'], df_src.loc[src_point]['Y']]
    nearest_xy = [df_filter_near.loc[0]['X'], df_filter_near.loc[0]['Y']]
    x_values = [src_xy[0],nearest_xy[0]]
    y_values = [src_xy[1],nearest_xy[1]]
    ax1.plot(x_values, y_values, 'black', label='closest', zorder=0)
    ax1.annotate(f'Dist={round(df_filter_near.loc[0]["dist"],2)}ft', xy=(((x_values[0]+x_values[1])/2),((y_values[0]+y_values[1])/2)), textcoords='offset points', xytext=(20,30), size=8, arrowprops=dict(arrowstyle="->"))
    ax1.annotate(f'#{src_point}', (df_src.loc[src_point]['X'], df_src.loc[src_point]['Y']), ha='center', size='5', textcoords='offset points', xytext=(0,4))
    for i, _ in enumerate(df_can.index):
        ax1.annotate(f'#{i}', (df_can.loc[i]['X'], df_can.loc[i]['Y']), ha='center', size='5', textcoords='offset points', xytext=(0,4))
    ax1.legend(loc='lower left', shadow=True, fancybox=True, markerscale=0.5)
    plt.title('Nearest Neighbors')
    plt.xlabel('X (ft)')
    plt.ylabel('Y (ft)')
    plt.xlim(min(df_filter_near['X'])-1000, max(df_filter_near['X'])+1000)
    plt.ylim(min(df_filter_near['Y'])-1000, max(df_filter_near['Y'])+1000)
    plt.locator_params(axis='x', nbins=5)
    # not setting set_aspect forces into a box and can be misleading
    # ax1.set_aspect('equal')
    plt.show()
    # plt.savefig('./Nearest_Neighbor.png')

    fig2,ax2 = plt.subplots()
    ax2.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    ax2.scatter(x=df_src.loc[src_point]['X'], y=df_src.loc[src_point]['Y'], color='red', label='src', zorder=1)
    ax2.scatter(x=df_can['X'], y=df_can['Y'], color='orange', label='canidates', zorder=2)
    ax2.scatter(x=df_filter_rad['X'], y=df_filter_rad['Y'], color='blue', label=f'Within radius', zorder=3)
    ax2.add_patch(plt.Circle((df_src.loc[src_point]['X'], df_src.loc[src_point]['Y']), rad_dist, color="red", fill=False))
    src_xy = [df_src.loc[src_point]['X'], df_src.loc[src_point]['Y']]
    x_values_rad = [src_xy[0],src_xy[0]-rad_dist]
    y_values_rad = [src_xy[1],src_xy[1]]
    ax2.plot(x_values_rad, y_values_rad, 'black', linestyle='dashed', label='radius', zorder=0)
    ax2.annotate(f'#{src_point}', (df_src.loc[src_point]['X'], df_src.loc[src_point]['Y']), ha='center', size='6', textcoords='offset points', xytext=(0,4))
    ax2.annotate(f'{rad_dist}ft', xy=(((x_values_rad[0]+x_values_rad[1])/2),((y_values_rad[0]+y_values_rad[1])/2)), textcoords='offset points', xytext=(0,-20), size=6, arrowprops=dict(arrowstyle="->"))
    for i, _ in enumerate(df_can.index):
        ax2.annotate(f'#{i}', (df_can.loc[i]['X'], df_can.loc[i]['Y']), ha='center', size='6', textcoords='offset points', xytext=(0,4))
    ax2.legend(loc='lower right', shadow=True, fancybox=True, prop={'size': 6}, markerscale=0.5)
    ax2.set_aspect('equal')
    plt.xlabel('X (ft)')
    plt.ylabel('Y (ft)')
    plt.title('Nearest Neigbors Within Radius')
    plt.xlim(min(df_filter_rad['X'])-1000, max(df_filter_rad['X'])+1500)
    plt.ylim(min(df_filter_rad['Y'])-1000, max(df_filter_rad['Y'])+1000)
    plt.locator_params(axis='x', nbins=5)
    # plt.savefig('./Nearest_Neighbor_Radius.png')
    plt.show()


    x_pos_near, x_pos_far = [], []
    width_near, width_far = [], []
    height_near, height_far = [], []
    for x in range(0,nearest):
        x_pos_near.append(df_graph.loc[x]['dist'])
        width_near.append(50)
        height_near.append(df_graph.loc[x]['height'])
    

    for x in range(nearest, nearest+1):
        x_pos_far.append(df_graph.loc[x]['dist'])
        width_far.append(50)
        height_far.append(df_graph.loc[x]['height'])

    x_pos_near_scale = [i for i in x_pos_near]
    x_pos_far_scale = [i for i in x_pos_far]

    xticks = [0]+list(map(int, x_pos_near_scale))+list(map(int, x_pos_far_scale))

    # plt.bar(x_pos, height, width=width)
    fig3, ax3 = plt.subplots()
    plt.bar(0, 20, width=50, color="red", label='src point')
    plt.bar(x_pos_near_scale, height_near, width=width_near, color="blue", label=f'nearest {nearest}')
    plt.bar(x_pos_far_scale, height_far, width=width_far, color="orange", label='Next nearest')
    plt.xlim(-100, max(df_graph["dist"]+200))
    plt.ylim(0,max(df_graph["height"]+20))
    plt.title('Profile View Distance to Nearest Neighbor')
    plt.xlabel('Distance (ft)')
    plt.ylabel('Height (ft)')
    ax3.annotate(f'#{src_point}', (0,20), ha='center', size='6', textcoords='offset points', xytext=(0,3))
    for i, _ in enumerate(df_graph.index):
        ax3.annotate(f'#{df_graph.loc[i]["buildingIDX"]}', (df_graph.loc[i]['dist'], df_graph.loc[i]['height']), ha='center', size='6', textcoords='offset points', xytext=(0,3))
    ax3.set_xticks(xticks)
    ax3.legend(loc='upper left', shadow=True, fancybox=True, markerscale=0.5)
    # plt.savefig('./Profile_view.png')
    plt.show()

def save_files(df_near, df_rad, src_point, df_src):
    if not os.path.exists('./output_radius/'):
        os.mkdir('./output_radius/')

    if not os.path.exists('./output_nearest/'):
        os.mkdir('./output_nearest/')

    src_name = df_src.loc[src_point]["Name"]
    save_name = src_name.replace(" ", "_")

    df_near = df_near.drop('radiusDist', axis=1)
    df_rad = df_rad.drop('nearest', axis=1)

    df_near.to_csv(f'./output_nearest/{save_name}_nearest.csv', index=False)
    df_rad.to_csv(f'./output_radius/{save_name}_radius.csv', index=False)


# def to_4326(df):
#     points= gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.X, df.Y), crs='esri:102726')
#     points_4326 = points.to_crs(4326)
#     extent = points_4326.total_bounds
#     extent_arrange = [extent[0], extent[2], extent[1], extent[3]]
#     centerX,centerY = (np.average(extent_arrange[2:]), np.average(extent_arrange[:2]))
#     df['X_map'] = points_4326.geometry.apply(lambda x: x.x)
#     df['Y_map'] = points_4326.geometry.apply(lambda x: x.y)
#     return df, centerX, centerY


# def create_map(df_canidates, near_in, near_out, rad_in, rad_out, df_src_points, rad_dist, nearest_num, lat, lon):
#     df_build, lat, lon = to_4326(df_canidates)
#     df_src, _, _ = to_4326(df_src_points)

#     Basemap
#     m = folium.Map(location=(lat, lon), zoom_start=12)

#     Add Layers
#     for name in df_src["Name"]:
#         NearMap(list(zip))
#     m.save("index.html")
#     webbrowser.open("index.html")
#     return

