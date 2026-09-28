# Function to get elevation
from datetime import date, datetime, timezone, timedelta
import pytz
import time
from suntime import Sun, SunTimeException
from suntimes import SunTimes 
import pandas as pd

import requests

def get_elevation(lat, long):
    url = f'https://api.opentopodata.org/v1/srtm30m?locations={lat},{long}'
    response = requests.get(url)
    if response.status_code == 200:
        elevation = response.json()['results'][0]['elevation']
        return elevation
    else:
        return None
    

def get_sunrise_sunset(dataframe, latitude, longitude, altitude, tz_name):
    
    tz_poland = pytz.timezone(tz_name)
    
    Sun = SunTimes(longitude=longitude, latitude=latitude, altitude=altitude) 

    dataframe["Datetime"] = pd.to_datetime(dataframe["Datetime"] )

    sun_rise_list = []
    sun_set_list  = []
    for index, row in dataframe.iterrows():

        for_date = row["Datetime"] 
        day = datetime(for_date.year, for_date.month, for_date.day)
        sun_rise = Sun.risewhere(day, tz_name).replace(tzinfo=None)
        sun_set  = Sun.setwhere(day, tz_name).replace(tzinfo=None)

        sun_rise_list.append(sun_rise)
        sun_set_list.append(sun_set)

    
    dataframe["sunrise"] = sun_rise_list
    dataframe["sunset"] = sun_set_list

    return dataframe