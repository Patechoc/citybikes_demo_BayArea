#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Explore bikers'ride at Bay area.
(http://www.bayareabikeshare.com/open-data)
"""

import os, sys
import pandas as pd
import numpy as np
import glob
from datetime import datetime, date
import pytz
from pytz import timezone
import dill as pickle

def read_dataTrips_from_csv_files(paths, asUTC=True):
    allFiles = []
    for path in paths:
        # Start Date: start date of trip with date and time, in PST (Pacific Standard Time)
        # End Date: end date of trip with date and time, in PST (Pacific Standard Time)
        search = path + "/2*_trip_data.csv"
        allFiles.extend(glob.glob(search))
    #print("Search for trip files here:\n", allFiles)
    ''' Concatenate all data into one DataFrame '''
    dfData = pd.DataFrame()
    list_ = []
    for file_ in allFiles:
        ## Date format CSV = "8/31/2015 23:13"
        #df = pd.read_csv(file_, index_col=None, header=0, parse_dates=[2,5])
        #parse = lambda x: pytz.timezone('UTC').localize(datetime.strptime(x,'%m/%d/%Y %H:%M')).astimezone(timezone('US/Pacific'))
        parseUTC   = lambda x: pytz.timezone('US/Pacific').localize(datetime.strptime(x,'%m/%d/%Y %H:%M')).astimezone(timezone('UTC'))
        parseLocal = lambda x: pytz.timezone('US/Pacific').localize(datetime.strptime(x,'%m/%d/%Y %H:%M')).astimezone(timezone('US/Pacific'))
        if asUTC:
            df = pd.read_csv(file_, index_col=None, header=0, parse_dates=['Start Date', 'End Date'], date_parser=parseUTC)
        else:
            df = pd.read_csv(file_, index_col=None, header=0, parse_dates=['Start Date', 'End Date']) #, date_parser=parseLocal)

        headers = df.columns
        df.rename(columns={'Subscription Type': u'Subscriber Type',}, inplace=True)
        list_.append(df)
    dfData = pd.concat(list_, ignore_index=True).fillna('')
    return dfData

def read_dataWeather_from_csv_files(paths):
    allFiles = []
    for path in paths:
        search = path + "/2*weather_data.csv"
        allFiles.extend(glob.glob(search))
    #print("Search for files here:\n", allFiles)
    ''' Concatenate all data into one DataFrame '''
    dfData = pd.DataFrame()
    df = pd.DataFrame()
    list_ = []
    for file_ in allFiles:
        ### 3Daily weather information per service area, provided from Weather Underground in PST (Pacific Standard Time)
        #df = pd.read_csv(file_, parse_dates=0, index_col=0)
        #parse = lambda x: pytz.timezone('UTC').localize(datetime.strptime(x,'%m/%d/%Y')).astimezone(timezone('US/Pacific'))
        parse = lambda x: pytz.timezone('US/Pacific').localize(datetime.strptime(x,'%m/%d/%Y')).astimezone(timezone('UTC'))
        df = pd.read_csv(file_, index_col=0, date_parser=parse)
        df.rename(columns={u'Date':u'PDT',
                           u'Max_Temperature_F':u'Max TemperatureF',
                           u'Mean_Temperature_F':u'Mean TemperatureF',
                           u'Min_TemperatureF':u'Min TemperatureF',
                           u'Max_Dew_Point_F':u'Max Dew PointF',
                           u'MeanDew_Point_F':u'MeanDew PointF',
                           u'Min_Dewpoint_F':u'Min DewpointF',
                           u'Max_Humidity':u'Max Humidity',
                           u'Mean_Humidity ':u' Mean Humidity',
                           u'Min_Humidity ':u' Min Humidity',
                           u'Max_Sea_Level_Pressure_In ':u' Max Sea Level PressureIn',
                           u'Mean_Sea_Level_Pressure_In ':u' Mean Sea Level PressureIn', 
                           u'Min_Sea_Level_Pressure_In ':u' Min Sea Level PressureIn',
                           u'Max_Visibility_Miles ':u' Max VisibilityMiles',
                           u'Mean_Visibility_Miles ':u' Mean VisibilityMiles',
                           u'Min_Visibility_Miles ':u' Min VisibilityMiles',
                           u'Max_Wind_Speed_MPH ':u' Max Wind SpeedMPH',
                           u'Mean_Wind_Speed_MPH ':u' Mean Wind SpeedMPH',
                           u'Max_Gust_Speed_MPH':u' Max Gust SpeedMPH',
                           u'Precipitation_In ':u'PrecipitationIn',
                           u'Cloud_Cover ':u' CloudCover',
                           u'Events':u' Events',
                           u'Wind_Dir_Degrees':u' WindDirDegrees',
                           u'zip':u'Zip',},
                           inplace=True)
        list_.append(df)
    dfData = pd.concat(list_, ignore_index=False).fillna('')
    return dfData

def read_dataDocks_from_csv_files(paths):
    allFiles = []
    for path in paths:
        # -time: date and time, PST
        search = path + "/2*status_data.csv"
        allFiles.extend(glob.glob(search))
    ''' Concatenate all data into one DataFrame '''
    dfData = pd.DataFrame()
    list_ = []
    for file_ in allFiles:
        ### date format "2014-09-01 00:00:03" 
        #parse = lambda x: pytz.timezone('US/Pacific').localize(datetime.strptime(x,'%Y/%m/%d %H:%M:%S')).astimezone(timezone('UTC'))
        df = pd.read_csv(file_) #, parse_dates=['time',], date_parser=parse)
        df.set_index(['time',], inplace=True)
        list_.append(df)
    dfData = pd.concat(list_, ignore_index=False).fillna('')
    return dfData

def read_dataStation_from_csv_files(paths):
    allFiles = []
    for path in paths:
        search = path + "/2*station_data.csv"
        allFiles.extend(glob.glob(search))
    ''' Concatenate all data into one DataFrame '''
    dfData = pd.DataFrame()
    list_ = []
    for file_ in allFiles:
        df = pd.read_csv(file_)
        list_.append(df)
    dfData = pd.concat(list_, ignore_index=False)
    return dfData

def filter_subscribers(dfData):
    df = dfData[dfData['Subscriber Type'] == 'Subscriber']
    return df

def filter_oneTimeUsers(dfData):
    df = dfData[dfData['Subscriber Type'] == 'Customer']
    return df

def splitByStationStarts(df):
    groupsStarts = df.groupby('Start Station')
    serieStarts = groupsStarts.size()
    ''' returns an array of dataframes filtering the original data:
    each element of the array is containing series with the same "start station" '''
    hotRoutesDicts = []
    listStations = []
    for nameStation, counts in serieStarts.iteritems():
        obj = {}
        listStations.append(nameStation)
        obj['nameStartStation'] = nameStation
        obj['data'] = df[df['Start Station'] == nameStation]
        hotRoutesDicts.append(obj)
    #print(listStations)
    #print("nameStartStation: ", hotRoutesDicts[0]['nameStartStation'])
    #print("Data about trips leaving from this station: ", hotRoutesDicts[0]['data'][0:3])
    return (hotRoutesDicts, listStations)


def count_bikes_arriving(dataTrips, stationName, freq_hour):
    freq = str(freq_hour)+'h'
    endStationTrips = dataTrips[dataTrips['End Station'] == stationName] # 4399 elements
    cumulEndTrips  = endStationTrips.groupby(pd.Grouper(key='End Date',freq=freq)).count()['Zip Code']
    ## remove the very first elements which causes issues when mapping to data in the dockStatus file
    cumulEndTrips = cumulEndTrips[1:]
    cumulEndTrips.head()
    return cumulEndTrips

def count_bikes_leaving(dataTrips, stationName, freq_hour):
    freq = str(freq_hour)+'h'
    startStationTrips = dataTrips[dataTrips['Start Station'] == stationName] # 5922 elements
    cumulStartTrips  = startStationTrips.groupby(pd.Grouper(key='Start Date',freq=freq)).count()['Zip Code']
    ## remove the very first elements which causes issues when mapping to data in the dockStatus file
    cumulStartTrips = cumulStartTrips[1:]
    cumulStartTrips.head()
    return cumulStartTrips

def get_stationID(stationName, stationData):
    return  list({i for i in stationData[stationData[u'name'] == stationName].station_id})[0]

def get_stationName(stationID, stationData):
    return  list({i for i in stationData[stationData[u'station_id'] == stationID][u'name']})[0]

def get_stationNames(stationIDs, stationData):
    return  [get_stationName(stationID, stationData) for stationID in stationIDs]

def get_stationDockcount(stationName, stationData):
    return list({i for i in stationData[stationData[u'name'] == stationName].dockcount})[0]

def get_stationLandmark(stationName, stationData):
    return {i for i in stationData[stationData[u'name'] == stationName][u'landmark']} # {'San Francisco'} for station "Market at Sansome"

def get_station_latitude(stationName, stationData):
    return list({i for i in stationData[stationData[u'name'] == stationName][u'lat']})[0] # set({...})

def get_station_longitude(stationName, stationData):
    return list({i for i in stationData[stationData[u'name'] == stationName][u'long']})[0] # set({...})

def get_station_coordinates(stationName, stationData):
    return (get_station_latitude(stationName, stationData), get_station_longitude(stationName, stationData))

def get_neighbouring_stationIDs(stationData, stationName, radius=0.5, unit = 'kilometer', filename= "distanceMatrix.csv"):
    ## Build a distance matrix from the stations' coordinates. Get the IDs of the station nearby.
    import distanceMatrix
    if not os.path.exists("distanceMatrix.csv"):
        distMatrix = distanceMatrix.build_geoPy_distanceMatrix(stationData, unit, filename)
    else:
        distMatrix = pd.read_csv('distanceMatrix.csv')
    #print(distMatrix.columns)
    stationID = get_stationID(stationName, stationData)
    neighboursIDs = distanceMatrix.find_neigbouring_stations(stationID, distMatrix, radius, unit="kilometer")
    return neighboursIDs

def get_weatherInfos_py27(weatherData, stationData, stationName):
    ## find weather available at the station zipcode, if not available in data, find weather at the closest zipcode(s) nearby
    from geopy.geocoders import Nominatim
    from pyzipcode import ZipCodeDatabase
    geolocator = Nominatim()
    (lat, lon) = get_station_coordinates(stationName, stationData)
    location = geolocator.reverse( (lat,lon) )
    #print("Address station from coordinates: "+location.address)
    zipcode = location.raw['address']['postcode']
    #print("Station post code: ", zipcode)
    zcDB = ZipCodeDatabase()

    stationWeather = pd.DataFrame()
    radius = 0    
    while radius < 10 and stationWeather.shape[0] == 0:
        zipNearby = [int(z.zip) for z in zcDB.get_zipcodes_around_radius(zipcode, radius)]
        #print(zipNearby)
        stationWeather = weatherData[weatherData['Zip'].isin(zipNearby)]
        #print("radius: ", radius)
        radius += 0.05  ## ?? 50m?, 0.05 miles?
    #print("post codes of neighborhood: ", zipNearby)
    def fixPrecip(x):
        try:
            return float(x)
        except:
            return 0.005 # maybe 0.01 or something?    
    precipitation_inch = stationWeather[u'PrecipitationIn'].apply(fixPrecip)
    temperature_fahrenheit = stationWeather[u'Mean TemperatureF']
    temperature_celcius = (temperature_fahrenheit -32.)/ 1.8
    precipitation_mm =  25.4 * precipitation_inch ## in millimeters
    #sfPrecipitation.max() #[sfPrecipitation != 0.0]
    #sfTemp.head
    return (precipitation_mm, temperature_celcius)


def get_weatherInfos(weatherData, stationData, stationName):
    ## find weather available at the station zipcode, if not available in data, find weather at the closest zipcode(s) nearby
    from geopy.geocoders import Nominatim
    from uszipcode import ZipcodeSearchEngine
    geolocator = Nominatim()
    (lat, lon) = get_station_coordinates(stationName, stationData)
    location = geolocator.reverse( (lat,lon) )
    zipcode = location.raw['address']['postcode']
    search = ZipcodeSearchEngine()
    zipcode_infos = search.by_zipcode(zipcode)
    stationWeather = pd.DataFrame()
    radius = 0    
    while radius < 10 and stationWeather.shape[0] == 0:
        zipNearby = [int(z.Zipcode) for z in search.by_coordinate(lat, lon,
                                                                  radius=radius, returns=5)]
        stationWeather = weatherData[weatherData['Zip'].isin(zipNearby)]
        #print("radius: ", radius)
        radius += 0.05  ## ?? 50m?, 0.05 miles?
    print("post codes of neighborhood: ", zipNearby)
    def fixPrecip(x):
        try:
            return float(x)
        except:
            return 0.005 # maybe 0.01 or something?    
    precipitation_inch = stationWeather[u'PrecipitationIn'].apply(fixPrecip)
    temperature_fahrenheit = stationWeather[u'Mean TemperatureF']
    temperature_celcius = (temperature_fahrenheit -32.)/ 1.8
    precipitation_mm =  25.4 * precipitation_inch ## in millimeters
    #sfPrecipitation.max() #[sfPrecipitation != 0.0]
    #sfTemp.head
    return (precipitation_mm, temperature_celcius)

'''
For a given station, this functions returns a list of neighbours with their distance and bikes/docks status (as dataframes)
'''
def get_docksBikes_available_neighbour_Stations(stationID, docksData, radius=0.7, unit='kilometer'):
    import distanceMatrix
    if not os.path.exists("data/distanceMatrix.csv"):
        distMatrix = distanceMatrix.build_geoPy_distanceMatrix(stationData, unit, filename= "data/distanceMatrix.csv")
    else:
        distMatrix = pd.read_csv('data/distanceMatrix.csv')
    neighboursIDs = distanceMatrix.find_neigbouring_stations(stationID, distMatrix, radius, unit)
    #print(neighboursIDs)
    #print(type(distMatrix))
    #print(distMatrix.columns)
    #print(distMatrix.head())
    neighbourStatus = []
    status = {}
    status['stationID'] = stationID
    #print("stationID:", stationID)
    for sID in neighboursIDs:
        statusNeighbour = {}
        ## Docks available at the station ("Market at Sansome"), ready to be filled
        DockStatus = docksData[docksData[u'station_id'] == sID][u'docks_available']
        ## Bikes available at the station ("Market at Sansome"), ready to leave
        BikeStatus = docksData[docksData[u'station_id'] == sID][u'bikes_available']
        statusNeighbour['DockStatus'] = DockStatus 
        statusNeighbour['BikeStatus'] = BikeStatus
        statusNeighbour['stationID']  = sID
        statusNeighbour['distToMainStation']  = distMatrix[(distMatrix[u'orig_station_id'] == stationID) & (distMatrix[u'dest_station_id'] == sID)]['distance'].tolist()[0]
        #print("sID:", sID)
        #print("statusNeighbour['distToMainStation']:", statusNeighbour['distToMainStation'])
        neighbourStatus.append(statusNeighbour)
    return neighbourStatus

'''
This generates X and y to use as input of our models 
## X : TRAINING DATA = numpy array or sparse matrix of shape [n_samples,n_features]
## y : TARGET VALUES = numpy array of shape [n_samples, n_targets]
'''
def build_model_inputs(inputs, weatherInfos={}): #setCumulTrips, freq_hour, withWeather=True, watch_neighbour=True,
                       #supply_demand='supply', checkDockAvailable=True, checkBikeAvailable=True):
    dataIn = inputs['data']
    ## the first date used from input has to be shifted by the biggest offset in order to get the right data
    offsetsTrip = [pd.DateOffset(**k) for k in [
                                                {'hours': 1*inputs['freq_hour']},\
                                                {'hours': 2*inputs['freq_hour']},\
                                                {'hours': 3*inputs['freq_hour']},\
                                                {'hours': 8*inputs['freq_hour']},\
                                                {'days': 1}, {'days': 7}, {'days': 14},\
                                                {'days': 28}
                                                ]
                   ]
    earliestTime = dataIn.index[0] + offsetsTrip[-1] #Timestamp('2013-08-29 12:00:00', offset='H') + 28 days
    dataInCropped = dataIn[earliestTime:]
    dataDelayed  = [dataIn[earliestTime-i:] for i in offsetsTrip]

    if 'withWeather' not in inputs.keys() or inputs['withWeather'] == True:
        precip = [weatherInfos['precipitation_mm'][weatherInfos['precipitation_mm'].index.asof(str(i))] for i in dataInCropped.index]
        dataDelayed.append(precip)
        temper = [weatherInfos['temperature_celcius'][weatherInfos['temperature_celcius'].index.asof(str(i))] for i in dataInCropped.index]
        dataDelayed.append(temper)

#     if inputs['supply_demand'] == 'supply':
#         if 'checkDockAvailable' not in inputs.keys() or inputs['checkDockAvailable'] == True:
#             dockAvail = [stationDockStatus[stationDockStatus.index.asof(str(i.to_datetime()).replace('-', '/'))] for i in dataInCropped.index]
#             dataDelayed.append(dockAvail)
#         if 'watch_neighbour' in inputs.keys() and inputs['watch_neighbour'] == True: # watch if docks of neighbour are full
#             (neighbourDockStatus, neighbourBikeStatus) = get_docksBikes_available_neighbour_Stations(stationID, radius=0.7, unit='kilometer')
#             sID = neighbourDockStatus.keys()
#             for s in sID:
#                 docksAround = [neighbourDockStatus[s][neighbourDockStatus[s].index.asof(str(i.to_datetime()).replace('-', '/'))] for i in setCumulTrips.index]
#                 countsDelayed.append(docksAround)
#     elif inputs['supply_demand'] == 'demand':
#         if 'checkBikeAvailable' not in inputs.keys() or inputs['checkBikeAvailable'] == True:
#             dockAvail = [stationDockStatus[stationDockStatus.index.asof(str(i.to_datetime()).replace('-', '/'))] for i in dataInCropped.index]
#             dataDelayed.append(dockAvail)
#         if 'watch_neighbour' in inputs.keys() and inputs['watch_neighbour'] == True: # watch if docks of neighbour are full
#             (neighbourDockStatus, neighbourBikeStatus) = get_docksBikes_available_neighbour_Stations(stationID, radius=0.7, unit='kilometer')
#             sID = neighbourDockStatus.keys()
#             for s in sID:
#                 bikesAround = [neighbourDockStatus[s][neighbourDockStatus[s].index.asof(str(i.to_datetime()).replace('-', '/'))] for i in dataInCropped.index]
#                 countsDelayed.append(bikesAround)

    countsZipped = zip(*(dataDelayed))
    X = np.array(countsZipped)
    # array of arrays with the 7 time shift for a given time sample
    # X.shape # (2187, 7) 2187 samples, 7 features
    y = np.array(dataInCropped[:len(X)])
    return (X,y)


def read_or_store_object(variableName, outputDir, fun, *args, **kwargs):
    outputFilename = os.path.join(outputDir, variableName + '.pkl')
    if os.path.isfile(outputFilename):
        with open(outputFilename, 'rb') as file:
            return pickle.load(file)
    else:
        if not os.path.exists(outputDir):
            os.makedirs(outputDir)
        with open(outputFilename, 'wb') as file:
            myObj = fun(*args, **kwargs)
            pickle.dump(myObj, file)
    return myObj


def main():
    from timeit import default_timer as timer
    paths = [r'data/babs_open_data_year_1/2014*']
    paths = [r'data/babs_open_data_year_*/201*', r'data/babs_open_data_year_*']
    paths = [
        r'data/babs_open_data_year_1/201402_babs_open_data',
        #r'data/babs_open_data_year_1/201408_babs_open_data',
        #r'data/babs_open_data_year_2'
    ]
    #weatherData = read_dataWeather_from_csv_files(paths=paths)
    #print(weatherData.columns.tolist())
    t0 = timer()
    docksData1a = read_dataDocks_from_csv_files(['data/babs_open_data_year_1/201402_babs_open_data/'])
    t1 = timer()
    #docksData1b = pd.read_csv('data/babs_open_data_year_1/201408_babs_open_data/201408_weather_data.csv', parse_dates=0, index_col=3)
    #t2 = timer()
    #docksData2 = pd.read_csv('data/babs_open_data_year_2/201508_weather_data.csv', parse_dates=0, index_col=3)
    #t3 = timer()
    #docksData = read_dataDocks_from_csv_files(paths=paths) ## load=80s
    return 0

if __name__ == '__main__':
    main()
