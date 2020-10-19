
Soil moisture daily prediction experiment system for Grow Time-Series dataset.

Implementation is done in Python. Numerous packages are used. Most used ones:

- numpy
- pandas
- matplotlib
- plotly
- jupyter
- pytorch

Main scripts are jupyter notebook that are inside the main folder. 

FOLDERS
--------------------------------------------------------------------------------

main: 

Contains main jupyter notebooks. Each notebook is numbered such as v01, v02, etc.

Notebooks should be run in order. Explanations are given inside the notebooks.
Please contact us if you come accross any problems.

----------------------------------------------------------------------------------

data:

All input, temporary and output data are in this directory. We exclude the data
for submission because of the very large volume.

Input data (GrowTimeSeries.csv, GrowLocations.csv) should be put here before 
running the system.

Also, since time-series data is too large, we split the data into different sensors
first. For used the grep function of Linux. Below is an example:

grep SoilMoisture GrowTimeSeries.csv > GrowTimeSeries_SoilMoisture.csv

and then, we added the header to the first line manually. Sensor types are:

- BatteryLevel
- SoilMoisture
- AirTemperature
- Light

----------------------------------------------------------------------------------

lib:

Contains the main classes and functions. Detailed explanations are given in comments.

- batching.py: Contains class BatcherTrain for preparing batch data to be used
in training of the LSTM.

- lstm.py: Contains LSTM class that is written in general time-series forecasting.

- normalizing.py: Contains the Normalizer class, which apply feature normalization.

- testing.py: Contains TesterMain and Tester classes which are used to mimic the
real-life daily usage. Obtains MAPE scores and contains visualisation functions.

- weather.py: Contains Darksky API query functions to get hourly historical data for 
a given location and given dates.


