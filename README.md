
![ML_WSM_Github](https://user-images.githubusercontent.com/33735397/159956101-71874623-1189-48dd-8774-102a9be4964c.png)

# Machine-Learning-Water-Systems-Model
This machine learning workflow demonstrates a framework to function a digital twin of a systems dynamics model for urban water system seasonal water system reliability, resilience, and vulnerability analysis.

### Abstract

Reliability, Resilience, and vulnerability (RRV) of water systems inform operational and management decisions regarding the system's supply and demand interactions.
Systems dynamics modeling can replicate the physical processes to evaluate system performance, but their high parameterization, high computational requirements, immense development period, difficulty calibrating, and software licensing present significant barriers to their widespread integration in practice and inhibit translation of research to practice.
To address these challenges, this study utilizes the Xtreme Gradient Boost (XGBoost) algorithm, automated feature selection, and automated hyper-parameter optimization as a novel machine learning framework to predict daily reservoir levels, groundwater extraction rates, and out-of-district water requests for determining each their RRV.
We examine the XGBoost water systems model (XGB-WSM) forecasting accuracy during dry, average, and wet climate scenarios compared with a water systems model developed to assess Salt Lake City's water system vulnerability to climate and population growth.
The XGB-WSM accurately models seasonal reservoir level dynamics, groundwater extraction rates, and out-of-district requests during the dry and average climate scenarios with a low RMSE and high $R^2$ ($>0.91$).
Wet climate conditions challenged the model; however, the seasonal trends and relationships to water system component thresholds mirrored the observed.
We find that machine learning demonstrates high potential for further development and integration in water resources planning and management, such as identifying and optimizing system operations, increasing community engagement, and strengthening the understanding of the water system for utilities without an existing systems model. 


### About the Salt Lake City Machine Learning Water System Model

This ML-WSM leverages decades of collaborate water systems modeling development between the University of Utah, the University of Alabama, and the Salt Lake Department of Public Utilities (SLCDPU), leveraging an existing documented and calibrated systems dynamics model built in the Goldsim software package for machine learning model training and testing.
This model serves as a template for other water systems with or without an existing systems dynamic water systems model to adapt and create a machine learning water systems model tailored to their unique system interactions and feedbacks. 

### Brief Salt Lake City water system background
![studyArea](https://user-images.githubusercontent.com/33735397/159961402-7a06a9fd-d275-4cb0-bb13-6f6b722a3860.PNG)

The SLCDPU shares many similarities with western and intermountain water utilities in growing metropolitan areas.
The region's interannual climate variability and seasonality strongly influence winter snowpack accumulation, extent, and duration, functioning as the primary mechanism controlling surface water supplies.
The region's cold semi-arid (BSk) to cold desert climate (BWk) significantly influences seasonal water use in response to high evapotranspiration during summer months.
From April to October, outdoor water use approaches 1000 mm to irrigate commercial and residential landscaping.
While a supply limited region, high seasonal water use places Utah as the 2nd or 3rd highest per-capita water use state depending on the year.

The utility satisfies its demands by sourcing adjacent Wasatch mountain surface water, underlying valley groundwater, and nearby Deer Creek reservoir and Central Utah Project (CUP) out-of-district supplies.
From the Wasatch mountains, City creek, Parley's creek, Big Cottonwood creek, and Little Cottonwood creek contribute over 60% of the municipality's annual supply.
The Parley's watershed contains Mountain Dell reservoir and Little Dell reservoir that hold up to 3.2x10^6 m^3 and 25x10^6  m^3, respectively, and are the only utility owned long-term storage sources.
When surface water supplies cannot meet demand during the summer months, SLCDPU has access up to 22x10^6 m^3 per year of groundwater.
If surface and groundwater supplies cannot satisfy demands, out-of-district Deer Creek reservoir and CUP sources support up to 61x10^6 m^3 per year of use.
This water comes at a greater cost, resulting in it being the least prioritized supply source and for purposes of RRV metrics in this study, these out-of-district requests are considered a reliability failure for the SLCDPU water system.


![SLC_WS_schematic](https://user-images.githubusercontent.com/33735397/159962157-b7ef6a33-e758-4d9b-924c-a266f17c0b0e.PNG)


