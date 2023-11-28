import sqlite3
import requests
import pandas as pd
import time
from nordvpn_switcher import initialize_VPN,rotate_VPN


conn = sqlite3.connect('FPA_FOD_20170508.sqlite')
initialize_VPN(save=1,area_input=['complete rotation'])

def aggregate_data(data):
    # Convert the hourly data to a pandas dataframe
    df_hourly = pd.DataFrame(data['hourly'])

    # Create empty lists to store our aggregated data
    time = []
    
    temperature_2m_mean = []
    temperature_2m_min = []
    temperature_2m_max = []
    temperature_2m_std = []
    
    relativehumidity_2m_mean = []
    relativehumidity_2m_min = []
    relativehumidity_2m_max = []
    relativehumidity_2m_std = []
    
    dewpoint_2m_mean = []
    dewpoint_2m_min = []
    dewpoint_2m_max = []
    dewpoint_2m_std = []
    
    apparent_temperature_mean = []
    apparent_temperature_min = []
    apparent_temperature_max = []
    apparent_temperature_std = []
    
    cloudcover_mean = []
    cloudcover_min = []
    cloudcover_max = []
    cloudcover_std = []
    
    cloudcover_high_mean = []
    cloudcover_high_min = []
    cloudcover_high_max = []
    cloudcover_high_std = []
    
    cloudcover_low_mean = []
    cloudcover_low_min = []
    cloudcover_low_max = []
    cloudcover_low_std = []
    
    cloudcover_mid_mean = []
    cloudcover_mid_min = []
    cloudcover_mid_max = []
    cloudcover_mid_std = []
    
    diffuse_radiation_mean = []
    diffuse_radiation_min = []
    diffuse_radiation_max = []
    diffuse_radiation_std = []
    
    diffuse_radiation_instant_mean = []
    diffuse_radiation_instant_min = []
    diffuse_radiation_instant_max = []
    diffuse_radiation_instant_std = []
    
    direct_normal_irradiance_mean = []
    direct_normal_irradiance_min = []
    direct_normal_irradiance_max = []
    direct_normal_irradiance_std = []
    
    direct_normal_irradiance_instant_mean = []
    direct_normal_irradiance_instant_min = []
    direct_normal_irradiance_instant_max = []
    direct_normal_irradiance_instant_std = []
    
    direct_radiation_mean = []
    direct_radiation_min = []
    direct_radiation_max = []
    direct_radiation_std = []
    
    direct_radiation_instant_mean = []
    direct_radiation_instant_min = []
    direct_radiation_instant_max = []
    direct_radiation_instant_std = []
    
    et0_fao_evapotranspiration_mean = []
    et0_fao_evapotranspiration_min = []
    et0_fao_evapotranspiration_max = []
    et0_fao_evapotranspiration_std = []
    
    precipitation_mean = []
    precipitation_min = []
    precipitation_max = []
    precipitation_std = []
    
    pressure_msl_mean = []
    pressure_msl_min = []
    pressure_msl_max = []
    pressure_msl_std = []
    
    rain_mean = []
    rain_min = []
    rain_max = []
    rain_std = []
    
    shortwave_radiation_mean = []
    shortwave_radiation_min = []
    shortwave_radiation_max = []
    shortwave_radiation_std = []
    
    shortwave_radiation_instant_mean = []
    shortwave_radiation_instant_min = []
    shortwave_radiation_instant_max = []
    shortwave_radiation_instant_std = []
    
    snow_depth_mean = []
    snow_depth_min = []
    snow_depth_max = []
    snow_depth_std = []
    
    snowfall_mean = []
    snowfall_min = []
    snowfall_max = []
    snowfall_std = []
    
    soil_moisture_0_to_7cm_mean = []
    soil_moisture_0_to_7cm_min = []
    soil_moisture_0_to_7cm_max = []
    soil_moisture_0_to_7cm_std = []
    
    soil_moisture_7_to_28cm_mean = []
    soil_moisture_7_to_28cm_min = []
    soil_moisture_7_to_28cm_max = []
    soil_moisture_7_to_28cm_std = []
    
    soil_moisture_28_to_100cm_mean = []
    soil_moisture_28_to_100cm_min = []
    soil_moisture_28_to_100cm_max = []
    soil_moisture_28_to_100cm_std = []
    
    soil_temperature_0_to_7cm_mean = []
    soil_temperature_0_to_7cm_min = []
    soil_temperature_0_to_7cm_max = []
    soil_temperature_0_to_7cm_std = []
    
    soil_temperature_7_to_28cm_mean = []
    soil_temperature_7_to_28cm_min = []
    soil_temperature_7_to_28cm_max = []
    soil_temperature_7_to_28cm_std = []
    
    surface_pressure_mean = []
    surface_pressure_min = []
    surface_pressure_max = []
    surface_pressure_std = []
    
    terrestrial_radiation_mean = []
    terrestrial_radiation_min = []
    terrestrial_radiation_max = []
    terrestrial_radiation_std = []
    
    terrestrial_radiation_instant_mean = []
    terrestrial_radiation_instant_min = []
    terrestrial_radiation_instant_max = []
    terrestrial_radiation_instant_std = []
    
    vapor_pressure_deficit_mean = []
    vapor_pressure_deficit_min = []
    vapor_pressure_deficit_max = []
    vapor_pressure_deficit_std = []
    
    windgusts_10m_mean = []
    windgusts_10m_min = []
    windgusts_10m_max = []
    windgusts_10m_std = []
    
    windspeed_10m_mean = []
    windspeed_10m_min = []
    windspeed_10m_max = []
    windspeed_10m_std = []
    
    windspeed_100m_mean = []
    windspeed_100m_min = []
    windspeed_100m_max = []
    windspeed_100m_std = []

    # Split the dataframe by day and aggregate
    for date in data['daily']['time']:
        mask = df_hourly['time'].str.startswith(date)
        df_day = df_hourly[mask]
        
        time.append(date)
        
        temperature_2m_mean.append(df_day['temperature_2m'].mean())
        temperature_2m_min.append(df_day['temperature_2m'].min())
        temperature_2m_max.append(df_day['temperature_2m'].max())
        temperature_2m_std.append(df_day['temperature_2m'].std())
        
        relativehumidity_2m_mean.append(df_day['relativehumidity_2m'].mean())
        relativehumidity_2m_min.append(df_day['relativehumidity_2m'].min())
        relativehumidity_2m_max.append(df_day['relativehumidity_2m'].max())
        relativehumidity_2m_std.append(df_day['relativehumidity_2m'].std())
        
        dewpoint_2m_mean.append(df_day['dewpoint_2m'].mean())
        dewpoint_2m_min.append(df_day['dewpoint_2m'].min())
        dewpoint_2m_max.append(df_day['dewpoint_2m'].max())
        dewpoint_2m_std.append(df_day['dewpoint_2m'].std())
        
        apparent_temperature_mean.append(df_day['apparent_temperature'].mean())
        apparent_temperature_min.append(df_day['apparent_temperature'].min())
        apparent_temperature_max.append(df_day['apparent_temperature'].max())
        apparent_temperature_std.append(df_day['apparent_temperature'].std())
        
        cloudcover_mean.append(df_day['cloudcover'].mean())
        cloudcover_min.append(df_day['cloudcover'].min())
        cloudcover_max.append(df_day['cloudcover'].max())
        cloudcover_std.append(df_day['cloudcover'].std())
        
        cloudcover_high_mean.append(df_day['cloudcover_high'].mean())
        cloudcover_high_min.append(df_day['cloudcover_high'].min())
        cloudcover_high_max.append(df_day['cloudcover_high'].max())
        cloudcover_high_std.append(df_day['cloudcover_high'].std())
        
        cloudcover_low_mean.append(df_day['cloudcover_low'].mean())
        cloudcover_low_min.append(df_day['cloudcover_low'].min())
        cloudcover_low_max.append(df_day['cloudcover_low'].max())
        cloudcover_low_std.append(df_day['cloudcover_low'].std())
        
        cloudcover_mid_mean.append(df_day['cloudcover_mid'].mean())
        cloudcover_mid_min.append(df_day['cloudcover_mid'].min())
        cloudcover_mid_max.append(df_day['cloudcover_mid'].max())
        cloudcover_mid_std.append(df_day['cloudcover_mid'].std())
        
        diffuse_radiation_mean.append(df_day['diffuse_radiation'].mean())
        diffuse_radiation_min.append(df_day['diffuse_radiation'].min())
        diffuse_radiation_max.append(df_day['diffuse_radiation'].max())
        diffuse_radiation_std.append(df_day['diffuse_radiation'].std())
        
        diffuse_radiation_instant_mean.append(df_day['diffuse_radiation_instant'].mean())
        diffuse_radiation_instant_min.append(df_day['diffuse_radiation_instant'].min())
        diffuse_radiation_instant_max.append(df_day['diffuse_radiation_instant'].max())
        diffuse_radiation_instant_std.append(df_day['diffuse_radiation_instant'].std())
        
        direct_normal_irradiance_mean.append(df_day['direct_normal_irradiance'].mean())
        direct_normal_irradiance_min.append(df_day['direct_normal_irradiance'].min())
        direct_normal_irradiance_max.append(df_day['direct_normal_irradiance'].max())
        direct_normal_irradiance_std.append(df_day['direct_normal_irradiance'].std())
        
        direct_normal_irradiance_instant_mean.append(df_day['direct_normal_irradiance_instant'].mean())
        direct_normal_irradiance_instant_min.append(df_day['direct_normal_irradiance_instant'].min())
        direct_normal_irradiance_instant_max.append(df_day['direct_normal_irradiance_instant'].max())
        direct_normal_irradiance_instant_std.append(df_day['direct_normal_irradiance_instant'].std())
        
        direct_radiation_mean.append(df_day['direct_radiation'].mean())
        direct_radiation_min.append(df_day['direct_radiation'].min())
        direct_radiation_max.append(df_day['direct_radiation'].max())
        direct_radiation_std.append(df_day['direct_radiation'].std())
        
        direct_radiation_instant_mean.append(df_day['direct_radiation_instant'].mean())
        direct_radiation_instant_min.append(df_day['direct_radiation_instant'].min())
        direct_radiation_instant_max.append(df_day['direct_radiation_instant'].max())
        direct_radiation_instant_std.append(df_day['direct_radiation_instant'].std())
        
        et0_fao_evapotranspiration_mean.append(df_day['et0_fao_evapotranspiration'].mean())
        et0_fao_evapotranspiration_min.append(df_day['et0_fao_evapotranspiration'].min())
        et0_fao_evapotranspiration_max.append(df_day['et0_fao_evapotranspiration'].max())
        et0_fao_evapotranspiration_std.append(df_day['et0_fao_evapotranspiration'].std())
        
        precipitation_mean.append(df_day['precipitation'].mean())
        precipitation_min.append(df_day['precipitation'].min())
        precipitation_max.append(df_day['precipitation'].max())
        precipitation_std.append(df_day['precipitation'].std())
        
        pressure_msl_mean.append(df_day['pressure_msl'].mean())
        pressure_msl_min.append(df_day['pressure_msl'].min())
        pressure_msl_max.append(df_day['pressure_msl'].max())
        pressure_msl_std.append(df_day['pressure_msl'].std())
        
        rain_mean.append(df_day['rain'].mean())
        rain_min.append(df_day['rain'].min())
        rain_max.append(df_day['rain'].max())
        rain_std.append(df_day['rain'].std())
        
        shortwave_radiation_mean.append(df_day['shortwave_radiation'].mean())
        shortwave_radiation_min.append(df_day['shortwave_radiation'].min())
        shortwave_radiation_max.append(df_day['shortwave_radiation'].max())
        shortwave_radiation_std.append(df_day['shortwave_radiation'].std())
        
        shortwave_radiation_instant_mean.append(df_day['shortwave_radiation_instant'].mean())
        shortwave_radiation_instant_min.append(df_day['shortwave_radiation_instant'].min())
        shortwave_radiation_instant_max.append(df_day['shortwave_radiation_instant'].max())
        shortwave_radiation_instant_std.append(df_day['shortwave_radiation_instant'].std())
        
        snow_depth_mean.append(df_day['snow_depth'].mean())
        snow_depth_min.append(df_day['snow_depth'].min())
        snow_depth_max.append(df_day['snow_depth'].max())
        snow_depth_std.append(df_day['snow_depth'].std())
        
        snowfall_mean.append(df_day['snowfall'].mean())
        snowfall_min.append(df_day['snowfall'].min())
        snowfall_max.append(df_day['snowfall'].max())
        snowfall_std.append(df_day['snowfall'].std())
        
        
        soil_moisture_0_to_7cm = (1/9)*df_day['soil_moisture_0_to_1cm'] + (2/9)*df_day['soil_moisture_1_to_3cm']  +  (6/9)*df_day['soil_moisture_3_to_9cm'] 
        soil_moisture_0_to_7cm_mean.append(soil_moisture_0_to_7cm.mean())
        soil_moisture_0_to_7cm_min.append(soil_moisture_0_to_7cm.min())
        soil_moisture_0_to_7cm_max.append(soil_moisture_0_to_7cm.max())
        soil_moisture_0_to_7cm_std.append(soil_moisture_0_to_7cm.std())
        
        
        soil_moisture_7_to_28cm_mean.append(df_day['soil_moisture_9_to_27cm'].mean())
        soil_moisture_7_to_28cm_min.append(df_day['soil_moisture_9_to_27cm'].min())
        soil_moisture_7_to_28cm_max.append(df_day['soil_moisture_9_to_27cm'].max())
        soil_moisture_7_to_28cm_std.append(df_day['soil_moisture_9_to_27cm'].std())
        
        soil_moisture_28_to_100cm_mean.append(df_day['soil_moisture_27_to_81cm'].mean())
        soil_moisture_28_to_100cm_min.append(df_day['soil_moisture_27_to_81cm'].min())
        soil_moisture_28_to_100cm_max.append(df_day['soil_moisture_27_to_81cm'].max())
        soil_moisture_28_to_100cm_std.append(df_day['soil_moisture_27_to_81cm'].std())
        
        
        soil_temperature_0_to_7cm = (df_day['soil_temperature_0cm'] + df_day['soil_temperature_6cm'])/2
        soil_temperature_0_to_7cm_mean.append(soil_temperature_0_to_7cm.mean())
        soil_temperature_0_to_7cm_min.append(soil_temperature_0_to_7cm.min())
        soil_temperature_0_to_7cm_max.append(soil_temperature_0_to_7cm.max())
        soil_temperature_0_to_7cm_std.append(soil_temperature_0_to_7cm.std())
        
        soil_temperature_7_to_28cm_mean.append(df_day['soil_temperature_18cm'].mean())
        soil_temperature_7_to_28cm_min.append(df_day['soil_temperature_18cm'].min())
        soil_temperature_7_to_28cm_max.append(df_day['soil_temperature_18cm'].max())
        soil_temperature_7_to_28cm_std.append(df_day['soil_temperature_18cm'].std())
        
        
        surface_pressure_mean.append(df_day['surface_pressure'].mean())
        surface_pressure_min.append(df_day['surface_pressure'].min())
        surface_pressure_max.append(df_day['surface_pressure'].max())
        surface_pressure_std.append(df_day['surface_pressure'].std())
        
        terrestrial_radiation_mean.append(df_day['terrestrial_radiation'].mean())
        terrestrial_radiation_min.append(df_day['terrestrial_radiation'].min())
        terrestrial_radiation_max.append(df_day['terrestrial_radiation'].max())
        terrestrial_radiation_std.append(df_day['terrestrial_radiation'].std())
        
        terrestrial_radiation_instant_mean.append(df_day['terrestrial_radiation_instant'].mean())
        terrestrial_radiation_instant_min.append(df_day['terrestrial_radiation_instant'].min())
        terrestrial_radiation_instant_max.append(df_day['terrestrial_radiation_instant'].max())
        terrestrial_radiation_instant_std.append(df_day['terrestrial_radiation_instant'].std())
        
        vapor_pressure_deficit_mean.append(df_day['vapor_pressure_deficit'].mean())
        vapor_pressure_deficit_min.append(df_day['vapor_pressure_deficit'].min())
        vapor_pressure_deficit_max.append(df_day['vapor_pressure_deficit'].max())
        vapor_pressure_deficit_std.append(df_day['vapor_pressure_deficit'].std())
        
        windgusts_10m_mean.append(df_day['windgusts_10m'].mean())
        windgusts_10m_min.append(df_day['windgusts_10m'].min())
        windgusts_10m_max.append(df_day['windgusts_10m'].max())
        windgusts_10m_std.append(df_day['windgusts_10m'].std())
        
        windspeed_10m_mean.append(df_day['windspeed_10m'].mean())
        windspeed_10m_min.append(df_day['windspeed_10m'].min())
        windspeed_10m_max.append(df_day['windspeed_10m'].max())
        windspeed_10m_std.append(df_day['windspeed_10m'].std())
        
        windspeed_100m_mean.append(df_day['windspeed_100m'].mean())
        windspeed_100m_min.append(df_day['windspeed_100m'].min())
        windspeed_100m_max.append(df_day['windspeed_100m'].max())
        windspeed_100m_std.append(df_day['windspeed_100m'].std())

    # Create the resulting dataframe
    df_result = pd.DataFrame({
        'time': time,
        'latitude': data['latitude'],
        'longitude': data['longitude'],
        'timezone': data['timezone'],
        'elevation': data['elevation'],
        'weathercode': data['daily']['weathercode'],
        'apparent_temperature_max_daily': data['daily']['apparent_temperature_max'],
        'apparent_temperature_mean_daily': data['daily']['apparent_temperature_mean'],
        'apparent_temperature_min_daily': data['daily']['apparent_temperature_min'],
        'et0_fao_evapotranspiration_daily': data['daily']['et0_fao_evapotranspiration'],
        'precipitation_hours_daily': data['daily']['precipitation_hours'],
        'precipitation_sum_daily': data['daily']['precipitation_sum'],
        'rain_sum_daily': data['daily']['rain_sum'],
        'shortwave_radiation_sum_daily': data['daily']['shortwave_radiation_sum'],
        'snowfall_sum_daily': data['daily']['snowfall_sum'],
        'sunrise_daily': data['daily']['sunrise'],
        'sunset_daily': data['daily']['sunset'],
        'temperature_2m_max_daily': data['daily']['temperature_2m_max'],
        'temperature_2m_mean_daily': data['daily']['temperature_2m_mean'],
        'temperature_2m_min_daily': data['daily']['temperature_2m_min'],
        'weathercode_daily': data['daily']['weathercode'],
        'winddirection_10m_dominant_daily': data['daily']['winddirection_10m_dominant'],
        'windgusts_10m_max_daily': data['daily']['windgusts_10m_max'],
        'windspeed_10m_max_daily': data['daily']['windspeed_10m_max'],
            
        'temperature_2m_mean': temperature_2m_mean,
        'temperature_2m_min': temperature_2m_min,
        'temperature_2m_max': temperature_2m_max,
        'temperature_2m_std': temperature_2m_std,
        'relativehumidity_2m_mean': relativehumidity_2m_mean,
        'relativehumidity_2m_min': relativehumidity_2m_min,
        'relativehumidity_2m_max': relativehumidity_2m_max,
        'relativehumidity_2m_std': relativehumidity_2m_std,
        'dewpoint_2m_mean': dewpoint_2m_mean,
        'dewpoint_2m_min': dewpoint_2m_min,
        'dewpoint_2m_max': dewpoint_2m_max,
        'dewpoint_2m_std': dewpoint_2m_std,
        'apparent_temperature_mean': apparent_temperature_mean,
        'apparent_temperature_min': apparent_temperature_min,
        'apparent_temperature_max': apparent_temperature_max,
        'apparent_temperature_std': apparent_temperature_std,
        'cloudcover_mean': cloudcover_mean,
        'cloudcover_min': cloudcover_min,
        'cloudcover_max': cloudcover_max,
        'cloudcover_std': cloudcover_std,
        'cloudcover_high_mean': cloudcover_high_mean,
        'cloudcover_high_min': cloudcover_high_min,
        'cloudcover_high_max': cloudcover_high_max,
        'cloudcover_high_std': cloudcover_high_std,
        'cloudcover_low_mean': cloudcover_low_mean,
        'cloudcover_low_min': cloudcover_low_min,
        'cloudcover_low_max': cloudcover_low_max,
        'cloudcover_low_std': cloudcover_low_std,
        'cloudcover_mid_mean': cloudcover_mid_mean,
        'cloudcover_mid_min': cloudcover_mid_min,
        'cloudcover_mid_max': cloudcover_mid_max,
        'cloudcover_mid_std': cloudcover_mid_std,
        'diffuse_radiation_mean': diffuse_radiation_mean,
        'diffuse_radiation_min': diffuse_radiation_min,
        'diffuse_radiation_max': diffuse_radiation_max,
        'diffuse_radiation_std': diffuse_radiation_std,
        'diffuse_radiation_instant_mean': diffuse_radiation_instant_mean,
        'diffuse_radiation_instant_min': diffuse_radiation_instant_min,
        'diffuse_radiation_instant_max': diffuse_radiation_instant_max,
        'diffuse_radiation_instant_std': diffuse_radiation_instant_std,
        'direct_normal_irradiance_mean': direct_normal_irradiance_mean,
        'direct_normal_irradiance_min': direct_normal_irradiance_min,
        'direct_normal_irradiance_max': direct_normal_irradiance_max,
        'direct_normal_irradiance_std': direct_normal_irradiance_std,
        'direct_normal_irradiance_instant_mean': direct_normal_irradiance_instant_mean,
        'direct_normal_irradiance_instant_min': direct_normal_irradiance_instant_min,
        'direct_normal_irradiance_instant_max': direct_normal_irradiance_instant_max,
        'direct_normal_irradiance_instant_std': direct_normal_irradiance_instant_std,
        'direct_radiation_mean': direct_radiation_mean,
        'direct_radiation_min': direct_radiation_min,
        'direct_radiation_max': direct_radiation_max,
        'direct_radiation_std': direct_radiation_std,
        'direct_radiation_instant_mean': direct_radiation_instant_mean,
        'direct_radiation_instant_min': direct_radiation_instant_min,
        'direct_radiation_instant_max': direct_radiation_instant_max,
        'direct_radiation_instant_std': direct_radiation_instant_std,
        'et0_fao_evapotranspiration_mean': et0_fao_evapotranspiration_mean,
        'et0_fao_evapotranspiration_min': et0_fao_evapotranspiration_min,
        'et0_fao_evapotranspiration_max': et0_fao_evapotranspiration_max,
        'et0_fao_evapotranspiration_std': et0_fao_evapotranspiration_std,
        'precipitation_mean': precipitation_mean,
        'precipitation_min': precipitation_min,
        'precipitation_max': precipitation_max,
        'precipitation_std': precipitation_std,
        'pressure_msl_mean': pressure_msl_mean,
        'pressure_msl_min': pressure_msl_min,
        'pressure_msl_max': pressure_msl_max,
        'pressure_msl_std': pressure_msl_std,
        'rain_mean': rain_mean,
        'rain_min': rain_min,
        'rain_max': rain_max,
        'rain_std': rain_std,
        'shortwave_radiation_mean': shortwave_radiation_mean,
        'shortwave_radiation_min': shortwave_radiation_min,
        'shortwave_radiation_max': shortwave_radiation_max,
        'shortwave_radiation_std': shortwave_radiation_std,
        'shortwave_radiation_instant_mean': shortwave_radiation_instant_mean,
        'shortwave_radiation_instant_min': shortwave_radiation_instant_min,
        'shortwave_radiation_instant_max': shortwave_radiation_instant_max,
        'shortwave_radiation_instant_std': shortwave_radiation_instant_std,
        'snow_depth_mean': snow_depth_mean,
        'snow_depth_min': snow_depth_min,
        'snow_depth_max': snow_depth_max,
        'snow_depth_std': snow_depth_std,
        'snowfall_mean': snowfall_mean,
        'snowfall_min': snowfall_min,
        'snowfall_max': snowfall_max,
        'snowfall_std': snowfall_std,
        'soil_moisture_0_to_7cm_mean': soil_moisture_0_to_7cm_mean,
        'soil_moisture_0_to_7cm_min': soil_moisture_0_to_7cm_min,
        'soil_moisture_0_to_7cm_max': soil_moisture_0_to_7cm_max,
        'soil_moisture_0_to_7cm_std': soil_moisture_0_to_7cm_std,
        'soil_moisture_7_to_28cm_mean': soil_moisture_7_to_28cm_mean,
        'soil_moisture_7_to_28cm_min': soil_moisture_7_to_28cm_min,
        'soil_moisture_7_to_28cm_max': soil_moisture_7_to_28cm_max,
        'soil_moisture_7_to_28cm_std': soil_moisture_7_to_28cm_std,
        'soil_moisture_28_to_100cm_mean': soil_moisture_28_to_100cm_mean,
        'soil_moisture_28_to_100cm_min': soil_moisture_28_to_100cm_min,
        'soil_moisture_28_to_100cm_max': soil_moisture_28_to_100cm_max,
        'soil_moisture_28_to_100cm_std': soil_moisture_28_to_100cm_std,
        'soil_temperature_0_to_7cm_mean': soil_temperature_0_to_7cm_mean,
        'soil_temperature_0_to_7cm_min': soil_temperature_0_to_7cm_min,
        'soil_temperature_0_to_7cm_max': soil_temperature_0_to_7cm_max,
        'soil_temperature_0_to_7cm_std': soil_temperature_0_to_7cm_std,
        'soil_temperature_7_to_28cm_mean': soil_temperature_7_to_28cm_mean,
        'soil_temperature_7_to_28cm_min': soil_temperature_7_to_28cm_min,
        'soil_temperature_7_to_28cm_max': soil_temperature_7_to_28cm_max,
        'soil_temperature_7_to_28cm_std': soil_temperature_7_to_28cm_std,
        'surface_pressure_mean': surface_pressure_mean,
        'surface_pressure_min': surface_pressure_min,
        'surface_pressure_max': surface_pressure_max,
        'surface_pressure_std': surface_pressure_std,
        'terrestrial_radiation_mean': terrestrial_radiation_mean,
        'terrestrial_radiation_min': terrestrial_radiation_min,
        'terrestrial_radiation_max': terrestrial_radiation_max,
        'terrestrial_radiation_std': terrestrial_radiation_std,
        'terrestrial_radiation_instant_mean': terrestrial_radiation_instant_mean,
        'terrestrial_radiation_instant_min': terrestrial_radiation_instant_min,
        'terrestrial_radiation_instant_max': terrestrial_radiation_instant_max,
        'terrestrial_radiation_instant_std': terrestrial_radiation_instant_std,
        'vapor_pressure_deficit_mean': vapor_pressure_deficit_mean,
        'vapor_pressure_deficit_min': vapor_pressure_deficit_min,
        'vapor_pressure_deficit_max': vapor_pressure_deficit_max,
        'vapor_pressure_deficit_std': vapor_pressure_deficit_std,
        'windgusts_10m_mean': windgusts_10m_mean,
        'windgusts_10m_min': windgusts_10m_min,
        'windgusts_10m_max': windgusts_10m_max,
        'windgusts_10m_std': windgusts_10m_std,
        'windspeed_10m_mean': windspeed_10m_mean,
        'windspeed_10m_min': windspeed_10m_min,
        'windspeed_10m_max': windspeed_10m_max,
        'windspeed_10m_std': windspeed_10m_std,
        'windspeed_100m_mean': windspeed_100m_mean,
        'windspeed_100m_min': windspeed_100m_min,
        'windspeed_100m_max': windspeed_100m_max,
        'windspeed_100m_std': windspeed_100m_std
    })

    return df_result


    
    
def fetch_weather_data_for_cities(conn, start_date, end_date, state='Georgia'):
    # Extract unique cities in Georgia from LatLon_City_Mapping
    query = """ 
    SELECT nearest_city, county_name, state_name, AVG(LATITUDE) as avg_lat, AVG(LONGITUDE) as avg_lon 
    FROM LatLon_City_Mapping 
    WHERE state_name = ? 
    GROUP BY state_name, county_name, nearest_city
    """
    cities = pd.read_sql(query, conn, params=(state,))
    
    try:
        existing_cities = pd.read_sql("SELECT DISTINCT city FROM AggregatedWeatherForecasts", conn)
        cities = cities[~cities['nearest_city'].isin(existing_cities['city'])]
    except:
        print(f"No cities currently in database. Fetching all cities for {state}")
        
    



    # Define the base URL for fetching weather data
    base_url = (
        "https://api.open-meteo.com/v1/forecast?"
        "latitude={latitude}&"
        "longitude={longitude}&"
        "hourly=temperature_2m,relativehumidity_2m,dewpoint_2m,apparent_temperature,precipitation,rain,snowfall,snow_depth,weathercode,pressure_msl,surface_pressure,cloudcover,cloudcover_low,cloudcover_mid,cloudcover_high,et0_fao_evapotranspiration,vapor_pressure_deficit,windspeed_10m,windspeed_100m,winddirection_10m,winddirection_100m,windgusts_10m,soil_temperature_0cm,soil_temperature_6cm,soil_temperature_18cm,soil_moisture_0_to_1cm,soil_moisture_1_to_3cm,soil_moisture_3_to_9cm,soil_moisture_9_to_27cm,soil_moisture_27_to_81cm,is_day,shortwave_radiation,direct_radiation,diffuse_radiation,direct_normal_irradiance,terrestrial_radiation,shortwave_radiation_instant,direct_radiation_instant,diffuse_radiation_instant,direct_normal_irradiance_instant,terrestrial_radiation_instant&"
        "daily=weathercode,temperature_2m_max,temperature_2m_min,temperature_2m_mean,apparent_temperature_max,apparent_temperature_min,apparent_temperature_mean,sunrise,sunset,precipitation_sum,rain_sum,snowfall_sum,precipitation_hours,windspeed_10m_max,windgusts_10m_max,winddirection_10m_dominant,shortwave_radiation_sum,et0_fao_evapotranspiration&"
        "timezone=America%2FNew_York"
    )
    

    # Loop over each city
    for _, row in cities.iterrows():
        url = base_url.format(latitude=row['avg_lat'], longitude=row['avg_lon'], start_date=start_date, end_date=end_date)

        # Fetch the data
        response = requests.get(url)
        data = response.json()
        max_retries = 5  # Define a maximum number of retries
        retry_count = 0
        
        if data.get('error'):
            while data.get('error'):
                if retry_count >= max_retries:
                    print("Max retries reached. Exiting loop.")
                    break
                print("Hourly API request limit exceeded. Switching server.")
                rotate_VPN()
                print("Waiting 30 seconds")
                time.sleep(30)
                response = requests.get(url)
                data = response.json()
                retry_count += 1


        # Check if data is not empty
        if not data:
            print(f"No data found for {row['nearest_city']} for the period {start_date} to {end_date}.")
            continue

        # Aggregate the data
        aggregated_data = aggregate_data(data)
        aggregated_data['city'] = row['nearest_city']
        aggregated_data['county'] = row['county_name']
        aggregated_data['state'] = row['state_name']

        # Append the data to SQLite database
        aggregated_data.to_sql('AggregatedWeatherForecasts', conn, if_exists='append', index=False)
        print(f"Completed data fetching for {row['nearest_city']} for the period {start_date} to {end_date}.")
        # Delete data to free memory
        del aggregated_data

    print(f"Completed data fetching for the period {start_date} to {end_date}.")

    print("All data fetching completed.")



fetch_weather_data_for_cities(conn, start_date='2023-11-13', end_date='2023-11-19', state='Georgia')
