
![ML_WSM_Github](https://user-images.githubusercontent.com/33735397/159956101-71874623-1189-48dd-8774-102a9be4964c.png)

# Machine-Learning-Water-Systems-Model
This machine learning workflow demonstrates a framework to function a digital twin of a systems dynamics model for urban water system seasonal water system reliability, resilience, and vulnerability analysis.

##Abstract
Reliability, Resilience, and vulnerability (RRV) of water systems inform operational and management decisions regarding the system's supply and demand interactions.
Systems dynamics modeling can replicate the physical processes to evaluate system performance, but their high parameterization, high computational requirements, immense development period, difficulty calibrating, and software licensing present significant barriers to their widespread integration in practice and inhibit translation of research to practice.
To address these challenges, this study utilizes the Xtreme Gradient Boost (XGBoost) algorithm, automated feature selection, and automated hyper-parameter optimization as a novel machine learning framework to predict daily reservoir levels, groundwater extraction rates, and out-of-district water requests for determining each their RRV.
We examine the XGBoost water systems model (XGB-WSM) forecasting accuracy during dry, average, and wet climate scenarios compared with a water systems model developed to assess Salt Lake City's water system vulnerability to climate and population growth.
The XGB-WSM accurately models seasonal reservoir level dynamics, groundwater extraction rates, and out-of-district requests during the dry and average climate scenarios with a low RMSE and high $R^2$ ($>0.91$).
Wet climate conditions challenged the model; however, the seasonal trends and relationships to water system component thresholds mirrored the observed.
We find that machine learning demonstrates high potential for further development and integration in water resources planning and management, such as identifying and optimizing system operations, increasing community engagement, and strengthening the understanding of the water system for utilities without an existing systems model. 
