
![ML_WSM_Github](https://user-images.githubusercontent.com/33735397/159956101-71874623-1189-48dd-8774-102a9be4964c.png)

# Machine-Learning-Water-Systems-Model
![GitHub](https://img.shields.io/github/license/whitelightning450/Machine-Learning-Water-Systems-Model?logo=GitHub&style=flat-square)
![GitHub top language](https://img.shields.io/github/languages/top/whitelightning450/Machine-Learning-Water-Systems-Model?logo=Jupyter&style=flat-square)
![GitHub repo size](https://img.shields.io/github/repo-size/whitelightning450/Machine-Learning-Water-Systems-Model?logo=Github&style=flat-square)
![GitHub language count](https://img.shields.io/github/languages/count/whitelightning450/Machine-Learning-Water-Systems-Model)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/whitelightning450/Machine-Learning-Water-Systems-Model)
![GitHub Pipenv locked Python version](https://img.shields.io/github/pipenv/locked/python-version/whitelightning450/Machine-Learning-Water-Systems-Model?style=plastic)
![GitHub branch checks state](https://img.shields.io/github/checks-status/whitelightning450/Machine-Learning-Water-Systems-Model/main)
![GitHub issues](https://img.shields.io/github/issues/whitelightning450/Machine-Learning-Water-Systems-Model)
![GitHub milestones](https://img.shields.io/github/milestones/closed/whitelightning450/Machine-Learning-Water-Systems-Model?style=plastic)
![GitHub milestones](https://img.shields.io/github/milestones/open/whitelightning450/Machine-Learning-Water-Systems-Model?style=plastic)
![GitHub closed issues](https://img.shields.io/github/issues-closed/whitelightning450/Machine-Learning-Water-Systems-Model?style=plastic)

 
This machine learning workflow demonstrates a framework to function a digital twin of a systems dynamics model for urban water system seasonal water system reliability, resilience, and vulnerability analysis.

### Abstract

Reliability, Resilience, and vulnerability (RRV) of water systems inform operational and management decisions regarding the system's supply and demand interactions.
Systems dynamics modeling can replicate the physical processes to evaluate system performance, but their high parameterization, high computational requirements, immense development period, difficulty calibrating, and software licensing present significant barriers to their widespread integration in practice and inhibit translation of research to practice.
To address these challenges, this study utilizes the Xtreme Gradient Boost (XGBoost) algorithm, automated feature selection, and automated hyper-parameter optimization as a novel machine learning framework to predict daily reservoir levels, groundwater extraction rates, and out-of-district water requests for determining each their RRV.
We examine the XGBoost water systems model (XGB-WSM) forecasting accuracy during dry, average, and wet climate scenarios compared with a water systems model developed to assess Salt Lake City's water system vulnerability to climate and population growth.
The XGB-WSM accurately models seasonal reservoir level dynamics, groundwater extraction rates, and out-of-district requests during the dry and average climate scenarios with a low RMSE and high R<sup>2</sup> (>0.91).
The model accurately predicted all water system components for all hydroclimate scnearios, correctly demonstrating the seasonal trends and relationships to water system component thresholds.
We find that machine learning demonstrates high potential for further development and integration in water resources planning and management, such as identifying and optimizing system operations, increasing community engagement, and strengthening the understanding of the water system for utilities without an existing systems model. 
 

### About the Salt Lake City Machine Learning Water System Model

This ML-WSM leverages decades of collaborate water systems modeling development between the University of Utah, the University of Alabama, and the Salt Lake Department of Public Utilities (SLCDPU), leveraging an existing documented and calibrated systems dynamics model (Salt Lake City Water Systems Model, SLC-WSM) built in the Goldsim software package for machine learning model training and testing.
This model serves as a template for other water systems with or without an existing systems dynamic water systems model to adapt and create a machine learning water systems model tailored to their unique system interactions and feedbacks. 

### Brief Salt Lake City water system background
![studyArea](https://user-images.githubusercontent.com/33735397/159961402-7a06a9fd-d275-4cb0-bb13-6f6b722a3860.PNG)

**Study Area**: Salt Lake City, Utah depends on winter snowpack in the adjacent Wasatch mountains to support surface water supplies, fill the Dell reservoir storage system, and replenish valley aquifers.
Its mountainous topography, geographical location, and arid climate result in highly skewed April to October water use to counteract seasonally high evapotranspiration for outdoor landscaping and irrigation.

The SLCDPU shares many similarities with western and intermountain water utilities in growing metropolitan areas.
The region's interannual climate variability and seasonality strongly influence winter snowpack accumulation, extent, and duration, functioning as the primary mechanism controlling surface water supplies.
The region's cold semi-arid (BSk) to cold desert climate (BWk) significantly influences seasonal water use in response to high evapotranspiration during summer months.
From April to October, outdoor water use approaches 1000 mm to irrigate commercial and residential landscaping.
While a supply limited region, high seasonal water use places Utah as the 2<sup>nd</sup> or 3<sup>rd</sup> highest per-capita water use state depending on the year.

The utility satisfies its demands by sourcing adjacent Wasatch mountain surface water, underlying valley groundwater, and nearby Deer Creek reservoir and Central Utah Project (CUP) out-of-district supplies.
From the Wasatch mountains, City creek, Parley's creek, Big Cottonwood creek, and Little Cottonwood creek contribute over 60% of the municipality's annual supply.
The Parley's watershed contains Mountain Dell reservoir and Little Dell reservoir that hold up to 3.2x10<sup>6</sup> m<sup>3</sup> and 25x10<sup>6</sup>  m<sup>3</sup>, respectively, and are the only utility owned long-term storage sources.
When surface water supplies cannot meet demand during the summer months, SLCDPU has access up to 22x10<sup>6</sup> m^3</sup> per year of groundwater.
If surface and groundwater supplies cannot satisfy demands, out-of-district Deer Creek reservoir and CUP sources support up to 61x10<sup>6</sup> m<sup>3</sup> per year of use.
This water comes at a greater cost, resulting in it being the least prioritized supply source, and for purposes of RRV metrics in this study,  considered a reliability failure for the water system.


![SLC_WS_schematic](https://user-images.githubusercontent.com/33735397/159962157-b7ef6a33-e758-4d9b-924c-a266f17c0b0e.PNG)

**Water System:** Salt Lake City water system leverages adjacent Wasatch Mountain surface water supplies, small reservoirs (Dell system), groundwater withdrawal, minimal in-system storage, and access to larger U.S. Bureau of Reclamation reservoir systems. 

### Machine Learning Model Inputs and Training
Many of the SLC-WSM inputs drive the XGB-WSM, including daily surface water supplies (e.g., City Creek), total system demand, service area population, reservoir levels, and the previous time step's reservoir levels, groundwater extraction rate, and out-of-district requests.
For example, the reservoir level on July 1<sup>st</sup> functions as an input to predict the reservoir level on July 2<sup>nd</sup>.
This research developed three additional metrics to further enhance model performance: total daily surface water supplies, day of the year, and month.
For model training we source streamflow and demand data from the utility's long-term records, the Kem C. Gardner Policy Institute provides population data, SLC-WSM simulations provide groundwater extraction rates, out-of-district Deer Creek reservoir use, and Mountain Dell and Little Dell reservoir levels.
Streamflow values entering water treatment facilities at each canyon's mouth form model inputs, as well as demand data conisting of the total volume of water entering the distribution system (all connected demands, leaks, and unaccounted-for losses).

For model demand predictions, the ML-based Climate-Supply-Development water demand model (CSD-WDM, https://github.com/whitelightning450/Water-Demand-Forecasting) predicts mean monthly demands in response to climate, supply, and socioeconomic factors. 
The CSD-WDM demonstrates high seasonal prediction accuracy with a mean absolute error of 62.8 lpcd and a mean absolute percent error  8.4%.
Since XGB-WSM operates at a daily time step, cubic spline interpolation downscales the demand predictions to a daily temporal resolution of total system water demand.

The XGB-WSM uses these features to predict daily Mountain and Little Dell reservoir level, system groundwater extraction rate, and out-of-district Deer Creek reservoir requests during peak SLCDPU water use between April and October.
Model training is on seventeen years of daily simulation data spanning from 2000 to 2020, omitting the three testing scenarios described below.

### Evaluation Scenarios
XGB-WSM predictive performance is on three different scenarios, based on annual snowpack, to represent the intermountain regions' hydroclimate variability.
The Alta Guard MesoWest weather station in the headwaters of Little Cottonwood Creek provides a long-term (1945-present) snowfall record to identify the most recent dry (2015), average (2017), and wet (2008) conditions for XGB-WSM simulation evaluation.
A Log-Pearson Type III analysis indicates the dry year demonstrates a non-exceedance return interval greater than 200 years and the wet year with an exceedance probability of 50 years.
These three water years establish the foundation of the testing scenarios by defining supply availability with the streamflow quantities at the canyon mouths and the corresponding observed per-capita water use.
Other system factors such as population, conservation, policy, and initial reservoir levels remain constant between scenarios.

## Using the XGB-WSM
The XGB-WSM requires the following dependencies and packages

## Dependencies (versions, environments)
Python: Version 3.8 or later

### Required packages

|  collections |   numpy     |  scikit-learn  |
|:-----------: | :--------:  | :-------: | 
| collinearity |     os      |  scipy    |
|     copy     |   pandas    |    seaborn   |
|   jenkspy    |  pathlib    | time    |
|     joblib   |    pickle   |  warning   |
|   matplotlib | progressbar |     xgboost      |


### Running XGB-WSM

The current XGB-WSM platform runs on preloaded and processed hydroclimate input data, including demand projections from the CSD-WDM.
This interface provides the user an opportunity to set reservoir reliability, resilience, and vulnerability thresholds (Mountain and Little Dell reservoirs), desired water use units (Acre-Feet, Million Gallons (MG), cubic meters(x10<sup>4</sup>  m<sup>3</sup>)) for out-of-district water requests (Deer Creek Reservoir) and groundwater withdrawall, and selecting the desired hydroclimate scenario (Wet, Average, Dry). 

![thresholds_scneario_units](https://user-images.githubusercontent.com/33735397/159972495-f21c839c-169f-4ddf-bd8c-da3fc3c1a4a5.PNG)

**XGB-WSM:** The existing user interface supports reservoir thresholds, desired units, and hydroclimate scenario.

Setting up the XGB-WSM uses the XGB_Model_v3_uncertainty.py module and imports as XGB_Model. 
This module takes in the reservoir threholds, desired units, prediction inputs, hydroclimate scenario, the evaluation timeframe, directory link, and asks if any figures should be saved. 

Before XGB-WSM makes predictions, the input data must be processed into the required format via the ProcessData class. 
During this step, change observations to True or False depending on the desired output. 
Choose False if making a forecast or choose True if validating the model and want to evaluate model performance. 

![DataProcessing](https://user-images.githubusercontent.com/33735397/159974043-529144fc-314f-4b43-88d4-ec65d5f6cd7b.PNG)

**Initiating the model:** Running the model provides the user with updates on data processing, predictions, and comparison with historical observations for a comprehensive reliability, resilience, and vulnerability (RRV) analysis. 

![Predictions](https://user-images.githubusercontent.com/33735397/159974131-f23caf7d-628a-4d03-a3a3-6ffc39768f06.PNG)

**XGB-WSM Results** The model communicate the reservoir thresholds and key information used to determine system satisfactory and unsatisfactory condtions with respect to historical obervations. 
For this system, the total volume of groundwater and out-of-district (Deer Creek Reservoir) water requests are key metrics to determine system RRV, which the model prints out for easy identification along with the range in anticipated values. 

![Mod_Ave_obs_True_Analysis](https://user-images.githubusercontent.com/33735397/159975932-8e7f343b-b7dc-491c-b31c-52b4f53001af.png)

**XGB-WSM Prediction w/Observations** Using average hydroclimate conditions as an example, the model illustrates water system performance for each key component with an April to October hydrograph of use or level.
The prediction is the blue line with the uncertainty (to a 95% confidence level) surounding this line in the lighter shade. 
The plots to the right communicate RRV, using the metrics of reliabiity, vulnerability, and max severity.
The model color codes satisfactory (green) and unsatisfactory (red) conditions in relation to the indicated threshold (reservoirs) or historical water use (groundwater and out-of-district) for easy communication concerning the timing of component conditions. 
This figure has observations set to True, demonstrated by the parity plot illustrating model performance. 


![Mod_Ave_obs_False_Analysis](https://user-images.githubusercontent.com/33735397/159975865-9e5e9f51-bde7-4928-8ac1-8eaed1fb875f.png)

**XGB-WSM Prediction w/o Observations** This figure is for the same average hydroclimate conditions as above, but run in forecasting mode without observations. 
Future development of the XGB-WSM will build on this platform to generate predictions for an array of hydroclimate influenced water system conditions.







