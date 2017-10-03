# Distance Matrix module (with Google Maps API)

import pandas as pd
from geopy.distance import vincenty
from geopy.geocoders import Nominatim
#from pyzipcode import ZipCodeDatabase
import urllib2
import json
import numpy as np

def get_latLong(stationData):
    #stationData = stationData.ix[0:, [u'station_id', u'lat', u'long']]
    stationData['lat_long'] = stationData[['lat', 'long']].apply(tuple, axis=1)
    return stationData

def get_GMaps_url(stationData):
    stationData = get_latLong(stationData)
    listeCoord = []
    listeIDs = []
    for index, row in stationData.iterrows():
        listeCoord.append(str(row['lat_long']))
        listeIDs.append(row['station_id'])
    origins = "origins=" + "|".join(listeCoord).replace("(","").replace(")","").replace(" ","")
    destinations = "destinations=" + "|".join(listeCoord).replace("(","").replace(")","").replace(" ","")
    serverKey = "AIzaSyAZPrFLFxz5ydvxAjBPIu4W-AhFrGHhnLA"
    ## Google Maps Distance matrix
    ## ex.: https://maps.googleapis.com/maps/api/distancematrix/json?origins=contrexeville&destinations=vittel&mode=bicycling&units=metric&language=en-US&key=AIzaSyAZPrFLFxz5ydvxAjBPIu4W-AhFrGHhnLA
    url = "https://maps.googleapis.com/maps/api/distancematrix/json?" + origins \
          + "&" + destinations \
          + "&mode=bicycling&units=metric&language=en-US&key=" + serverKey
    return url

def get_GMaps_distanceMarix_as_json(url):
    resp = urllib2.urlopen(url)
    data = json.load(resp)
    return data

def get_GMaps_distanceMarix(stationData):
    url = get_GMaps_url(stationData)
    jsonData = get_GMaps_distanceMarix_as_json(url)
    return jsonData


def get_distance(latLong1, latLong2, unit='kilometer'):
    if unit == 'mile' or unit == 'miles':
        distance = vincenty(latLong1, latLong2).miles
    else:
        distance = vincenty(latLong1, latLong2).kilometers
    return distance

def build_geoPy_distanceMatrix(stationData, unit = 'kilometer', filename='data/distanceMatrix.csv'):
    stationData = get_latLong(stationData)
    stationData1 = stationData.rename(columns = lambda x : 'orig_' + x)
    stationData2 = stationData.rename(columns = lambda x : 'dest_' + x)
    distArr = []
    for i1, row1 in stationData1.iterrows():
        for i2, row2 in stationData2.iterrows():
            if i2 <= i1:
                continue
            res = row1.append(row2)
            if unit == 'mile' or unit == 'miles':
                res['distance'] = get_distance(res['orig_lat_long'], res['dest_lat_long'], unit= 'miles')       
            else:
                res['distance'] = get_distance(res['orig_lat_long'], res['dest_lat_long'], unit= 'kilometers')       
            res['unit'] =  unit
            distArr.append(res)
    distMat = pd.DataFrame(np.array(distArr), columns=distArr[0].keys())
    distMat.to_csv(filename)
    return distMat

"""
 The radius defines the number of regions:
 - a too small radius will give only one station per region.
 - if too big, only one region will include all stations.
 Regions may overlap and include part of the same stations.
"""
def find_neigbouring_stations(stationID, distanceMatrix, radius=1, unit="kilometer"):
    unitMatrix = list(distanceMatrix[0:1]['unit'])[0]
    if unit == "mile" or unit == "miles":
        if unitMatrix == "kilometers":
            radius = miles_to_kilometers(radius)
    belowRadius = distanceMatrix[distanceMatrix['distance'] < radius]
    neighbours = belowRadius[(belowRadius['orig_station_id'] == stationID)  | (belowRadius['dest_station_id'] == stationID)]
    neighboursIDs = np.unique(np.append(np.array(neighbours['orig_station_id']),
                                        np.array(neighbours['dest_station_id'])))
    index = np.where(neighboursIDs==stationID,)[0]
    neighboursIDs = np.delete(neighboursIDs, index)
    return neighboursIDs


def miles_to_kilometers(x):
    return 1.60934 * x

if __name__ == "__main__":
    stationData = pd.read_csv('data/babs_open_data_year_1/201402_babs_open_data/201402_station_data.csv')
    distMatrix = build_geoPy_distanceMatrix(stationData.ix[0:20], unit = 'kilometer', filename='data/distanceMatrix.csv')
    neighboursIDs = find_neigbouring_stations(4, distMatrix, radius=1, unit="kilometer")
    #url = get_GMaps_url(stationData)
    #jsonData = get_distanceMarix_as_json(url)
    #print jsonData
    #print url
