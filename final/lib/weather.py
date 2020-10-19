
import darksky
from darksky.api import DarkSky, DarkSkyAsync
from darksky.types import languages, units, weather
import pandas as pd
import datetime as dt


# API_KEY = PLACE API KEY HERE
darksky_api = DarkSky(API_KEY)


def query_one(lat, lon, dtnow):

    res = darksky_api.get_time_machine_forecast(
        lat, lon,
        time=dtnow,
        extend='hourly',
        lang=languages.ENGLISH,
 #       units=units.AUTO,
        exclude=[weather.MINUTELY, weather.ALERTS]
    )

    return res

def query(ds_datetimes, lat, lon):

    dates = ds_datetimes.apply(lambda x: x.date()).unique()

    data = []

    for datenow in dates:

        dtnow = pd.to_datetime(datenow)

        resnow = query_one(lat, lon, dtnow)

        data += [datanow.__dict__ for datanow in resnow.hourly.data]

    if not data:
        return None

    df_weather = pd.DataFrame(data)

    df_weather = df_weather.rename({'time': 'datetime'}, axis=1)

    df_weather['datetime'] = df_weather['datetime'].apply(lambda x: x.replace(tzinfo=None))

    df_weather = df_weather.sort_values('datetime')

    dtnow = dt.datetime.now()

    df_weather['datetimeQuery'] = dtnow

    df_weather['isHistory'] = df_weather['datetime'] < dtnow

    return df_weather

    
