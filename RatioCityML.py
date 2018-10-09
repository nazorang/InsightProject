#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML models to predict possible building heights based on near by buildings, 
proximity to transit, parcel area, property area and population density.
"""



#Importing packages



%matplotlib inline
​
import os
​import psycopg2
import matplotlib as mpl
import matplotlib.pyplot as plt
​
from shapely.geometry import Point
import pandas as pd
import geopandas as gpd
from geopandas import GeoSeries, GeoDataFrame
from geopandas.tools import sjoin
import numpy as np
import folium
from folium import plugins
from folium.plugins import *
​
import shapely
import unicodedata
import pysal as ps
​
​
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon
​
​
from sklearn import cluster
from sklearn.preprocessing import scale

#Connecting to Postgres and loading data

# set postgres username and database name
dbname = 'toronto'
username = 'ratioadmin'
con = None
con = psycopg2.connect(database = dbname, user = username)

Massing = gpd.read_postgis('SELECT * FROM "3DMassing"', con, coerce_float=False)
Address = gpd.read_postgis('SELECT * FROM "AddressPoints"', con, coerce_float=False)
Centreline = gpd.read_postgis('SELECT * FROM "Centreline"', con, coerce_float=False)
Secondary_plan = gpd.read_postgis('SELECT * FROM "CP_SecondaryPlan_Area_2018_WGS84"', con, coerce_float=False)
Site_policy = gpd.read_postgis('SELECT * FROM "CP_SiteAreaSpecificPolicies_2018_WGS84"', con, coerce_float=False)
greenspace = gpd.read_postgis('SELECT * FROM "CityGreen"', con, coerce_float=False)
Official_plan = gpd.read_postgis('SELECT * FROM "OfficialPlan"', con, coerce_float=False)
Property_boundary = gpd.read_postgis('SELECT * FROM "PropertyBoundaries"', con, coerce_float=False)
TTC_Subway = gpd.read_postgis('SELECT * FROM "TTC_Subway_Line_2018_wgs84"', con, coerce_float=False)
TTC_Streetcar = gpd.read_postgis('SELECT * FROM "TTC_Streetcar"', con, coerce_float=False)
Zoning_heights = gpd.read_postgis('SELECT * FROM "Zoning_Height"', con, coerce_float=False)
Zoning_categories = gpd.read_postgis('SELECT * FROM "ZoningCategories"', con, coerce_float=False)
dev_apps = gpd.read_postgis('SELECT * FROM "DevApplications"', con, coerce_float=False)

con.close()
#Cleaning data- spatial joins-creating new variables

#spatially joining massing and property boundaries
massing_boundaries_join = gpd.sjoin(Massing, Property_boundary, how="inner", op='within')

massing_boundary = massing_boundaries_join [['geom','max_height', 'avg_height', 'base', 'area_calc','mass_area', 'shape_area']]

massing_boundary['max_height'] = massing_boundary['max_height'].astype(float)
massing_boundary['avg_height'] = massing_boundary['avg_height'].astype(float)
massing_boundary['base'] = massing_boundary['base'].astype(float)
massing_boundary['area_calc'] = massing_boundary['area_calc'].astype(float)
massing_boundary['mass_area'] = massing_boundary['mass_area'].astype(float)
massing_boundary['shape_area'] = massing_boundary['shape_area'].astype(float)
​
  

massing_boundary = massing_boundary.dropna()

massing_boundary_cat = gpd.sjoin(massing_boundary,Zoning_heights, how="inner", op='within')
​

massing_boundary_cat = massing_boundary_cat.drop(columns = ['index_right'])

massing_boundary_zoning = gpd.sjoin(massing_boundary_cat,Zoning_categories, how="inner", op='within')
​
massing_boundary_zoning = massing_boundary_zoning [['geom','max_height', 'base', 'area_calc', 'mass_area', 'shape_area', 'ht_height','zn_zone']]

massing_boundary_zoning['ht_height'] = massing_boundary_zoning['ht_height'].astype(float)

massing_boundary_zoning ['delta_height'] = massing_boundary_zoning.apply(lambda row: (-row['ht_height'] + row['max_height']),axis=1)
​

massing_boundary_zoning = massing_boundary_zoning[massing_boundary_zoning.zn_zone != 'R']
massing_boundary_zoning = massing_boundary_zoning[massing_boundary_zoning.zn_zone != 'RD']
massing_boundary_zoning = massing_boundary_zoning[massing_boundary_zoning.zn_zone != 'RS']
​
#Plotting where maximum height exceeds by-law height

plot_delta = massing_boundary_zoning.drop (columns = ["zn_zone","max_height","base", "area_calc", "mass_area", "shape_area" , "ht_height"])

plot_delta = plot_delta [plot_delta.delta_height > 0]

map = folium.Map(location=[43.6532, -79.3832],
                            zoom_start=13,
                            tiles="Stamen Terrain")

plot_delta["long"] = plot_delta.centroid.map(lambda p: p.x)
plot_delta["lat"] = plot_delta.centroid.map(lambda p: p.y)

plot_delta = plot_delta.drop(columns = ["geom"])

def delta_plot(file):
    # generate a new map
    folium_map = folium.Map(location=[43.6532, -79.3832],
                            zoom_start=13,
                            tiles="stamentoner")
​
    for index, row in file.iterrows():
        if row['delta_height']>=30:
            color= "#d7191c"
        elif row['delta_height'] < 30 and row['delta_height'] >= 18:
             color= "#fdae61"
        elif row['delta_height'] < 18 and row['delta_height'] >= 6:
             color= "#abdda4"
        else:
            color="#2b83ba"
​
​
        
            
        folium.CircleMarker(location=(row["lat"],row["long"]), color=color, radius=5, fill=True ).add_to(folium_map)
    return folium_map

map=delta_plot(plot_delta)
map.save(outfile='Delta Map.html')
Calculate minimum distance from transit

def min_distance(point, lines):
    return lines.distance(point).min()
​
​

massing_boundary_zoning['min_dist_to_subway'] = massing_boundary_zoning.geometry.apply(lambda x: min_distance(x, TTC_Subway))

massing_boundary_zoning['min_dist_to_streetcar'] = massing_boundary_zoning.geometry.apply(lambda x: min_distance(x,TTC_Streetcar))
Population density


population = gpd.read_file ('geos (1).geojson')

population = population[["geometry","pop"]]

final = gpd.sjoin(massing_boundary_zoning, population, how="inner", op='within')

final["pop"] = final["pop"].astype(float)
 
​
#Polygon centeroids and final datatable

final["long"] = final.centroid.map(lambda p: p.x)
final["lat"] = final.centroid.map(lambda p: p.y)

from sklearn.model_selection import train_test_split
​

X_train, X_test, y_train, y_test = train_test_split(final.drop (columns =['index_right','max_height']),
                                                    final["max_height"], test_size=0.4, random_state=0)
"""
K Nearest Neighbours
Using only latitude and longitude to predict building height based on the height of the nearest buildings
"""

X1_train = pd.concat([X_train['lat'],X_train['long']], axis=1)
X1_test = pd.concat([X_test ['lat'], X_test['long']], axis=1)


from sklearn.neighbors import KNeighborsRegressor
​
neigh = KNeighborsRegressor(n_neighbors = 4 ,weights = 'distance')
neigh.fit(X1_train, y_train) 
y_predict_neigh = neigh.predict(X1_test)
​
# Evaluate the model
mae_neigh = np.mean(abs(y_predict_neigh - y_test))
R_sq = neigh.score(X1_test, y_test)
​
​
print('MAE = %0.4f  ' % mae_neigh )
print( 'R\u00b2 = %0.4f ' %R_sq )
​

#Cross Validation

error = []
R = []
​
# Calculating error for K values between 1 and 40
for i in range(1, 40):  
    knn = KNeighborsRegressor(n_neighbors=i, weights= 'distance')
    knn.fit(X1_train, y_train)
    pred_i = knn.predict(X1_test)
    error.append(np.mean(abs(pred_i - y_test)))
    R.append(knn.score(X1_test, y_test))

plt.figure(figsize=(12, 6))  
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10)
​
plt.xlabel('K Value', fontsize=20)  
plt.ylabel('MAE', fontsize=20)
plt.tick_params(labelsize=20)
​
 


plt.figure(figsize=(12, 6))  
plt.plot(range(1, 40), R, color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10)
​
plt.xlabel('K Value', fontsize = 20)  
plt.ylabel('R Squared', fontsize =20) 
plt.tick_params(labelsize=20)

#Random Forest Regression

final = final.drop(columns= ['index_right','geom'])

X2_train = X_train.drop(columns =['lat','long','geom', 'delta_height', 'zn_zone', 'base','shape_area'])
X2_test = X_test.drop(columns =['lat','long','geom', 'delta_height', 'zn_zone', 'base','shape_area'])

​
from sklearn.ensemble import RandomForestRegressor
​
random_forest = RandomForestRegressor(n_estimators = 30)
random_forest.fit(X2_train, y_train)
y_predict_rand = random_forest.predict(X2_test)
​
# Evaluate the model
mae_rand = np.mean(abs(y_predict_rand - y_test))
R1_sq = random_forest.score(X2_test, y_test)
​
​
print('MAE = %0.4f  ' % mae_rand )
print( 'R\u00b2 = %0.4f ' %R1_sq )
​


importances = random_forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in random_forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
​
​
for f in range(X2_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
​
# Plot the feature importances of the forest
plt.figure(figsize = (10,5))
plt.bar(range(X2_train.shape[1]), importances[indices],
       color="g", yerr=std[indices], align="center")
plt.xticks(range(X2_train.shape[1]),indices )
plt.xlim([-1, X2_train.shape[1]])
plt.tick_params(labelsize=20)
plt.show()

#Mapping final results

result = []
result = X_test
result ['actual_height'] = y_test
result ['rand_forest'] = y_predict_rand
result ['nearest_neigh'] = y_predict_neigh
​

result_plot = pd.concat ([result['lat'], result['long'], result['actual_height'], 
                          result['rand_forest'], result['nearest_neigh']], axis=1)


import vincent
from vincent import AxisProperties, PropertySet, ValueRef

import json 
def map_plot(file):
    # generate a new map
    folium_map = folium.Map(location=[43.6532, -79.3832],
                            zoom_start=13,
                            tiles="Stamen Terrain")

    for index, row in file.iterrows():

        #Defining marker colours.
        if int(row['actual_height']/3) <= int(row['ht_height']/3):
            color="#4daf4a"  

        else:
            color="#fc8d62" 
        # Create vincent chart and popup.
       
        data = [int(row["ht_height"]/3), int(row["actual_height"]/3),int(row["nearest_neigh"]/3),int(row["rand_forest"]/3)]
        ind = ['By-law Limit', 'Actual', 'Nearest Buildings', 'Random Forest']
        df = pd.DataFrame (data, index = ind)
        bar_chart = vincent.Bar(df,
                                width=350,
                                height=300)
        bar_chart.axis_titles (x='', y='Number of Floors')
        bar_chart.colors(brew='Set3')

        bar_chart.scales['x'].padding = 0.2

        bar_json = bar_chart.to_json()
        bar_dict = json.loads(bar_json)
        popup = folium.Popup(max_width=400)
        folium.Vega(bar_dict, height=350, width=400).add_to(popup)            
        folium.CircleMarker(location=(row["lat"],row["long"]),popup=popup, color=color, radius=5, fill=True ).add_to(folium_map)
    return folium_map


m=map_plot(result_plot)
m.save(outfile='Height Results.html')
