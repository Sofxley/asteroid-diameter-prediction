# Asteroid Diameter Prediction: EDA

_I also recommend reading the notebook on Kaggle._

<a href="https://www.kaggle.com/code/blulypsee/asteroid-diameter-prediction-eda/notebook"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open in Kaggle" /></a> 

## SUMMARY

* Examined the distributions, correlations of the features and relationships between them, which allowed for a better understanding of the dataset and helped to inform decisions on feature selection and preprocessing for the predictive model.
* Identified outliers and missing values, which were then handled through various methods such as imputation and removal.
* Conducted exploratory data analysis on a dataset of asteroid observations, which consisted of various features such as absolute magnitude, semi-major axis, albedo, and diameter of the asteroids.
* The analysis can contribute to the development of an accurate and robust model for predicting asteroid diameter, which is important for identifying potentially hazardous asteroids, and for the models that actually classify hazardous and non-hazardous asteroids.

## Table of Contents
1. [Introduction](#intro)
2. [Problem Statement and Motivation](#prob)
3. [EDA](#EDA)
4. Feature Engineering and Data Preprocessing (yet to come)
5. Building Neural Networks and Making Predictions  (yet to come)

<a id="intro"></a>
## Introduction

Asteroids are small, rocky objects that revolve around the Sun in elliptical orbits. They are sometimes called minor planets or planetoids and are remnants left over from early formation of our solar system about 4.6 billion years ago. 

Asteroids range in size from tiny specks to hundreds of kilometers across, and they are located primarily in the asteroid belt between Mars and Jupiter, but they can also be found in other regions of the solar system. Most asteroids are irregularly shaped, only some of them are close to spherical, and they are often pitted or cratered. But wait, what do we mean by diameter of the asteroid if it is of irregular shape?

Basically, the diameter of an asteroid of irregular shape is usually represented by its __equivalent diameter__, which is the diameter of a sphere that has the same volume as the asteroid. This equivalent diameter is a useful way to compare the size of different asteroids, but it should be noted that it does not necessarily represent the actual physical size or shape of the asteroid. In some cases, the equivalent diameter may be an overestimate or an underestimate of the true size of the asteroid, depending on its shape and density.

There are __two classes of asteroids__ that are of particular interest to astronomers and planetary scientists: __NEAs and PHAs__. NEAs stands for __"Near-Earth Asteroids"__, and, as the name suggests, these are asteroids that come relatively close to Earth's orbit, meaning they have the potential to collide with Earth. On the other hand, PHA stands for __"Potentially Hazardous Asteroids"__ and these are a subset of NEAs that are particularly worrisome because they have the potential to come even closer to Earth's orbit, and their size is big enough that they could cause significant damage if they were to collide with Earth.

Apart from other reasons, diameter prediction is extremely important for identifying NEAs and PHAs, since the size of an asteroid can have a big impact on how it interacts with Earth. Smaller asteroids are more likely to burn up in Earth's atmosphere before reaching the surface, while larger asteroids can cause more damage upon impact. By predicting the diameter of NEAs and PHAs, scientists can better assess the potential threat these asteroids pose to Earth and develop strategies to diminish any potential harm. In addition, the diameter prediction can also provide valuable information about the composition, structure and evolution of asteroids, which can lead to deeper understanding of origins of the Solar System we live in.

<a id="prob"></a>
## Problem Statement and Motivation

This project aims to estimate the size of asteroids in our solar system based on various observable characteristics such as their absolute magnitude, albedo, distance from the Sun, their orbital parameters. Accurately predicting the size of asteroids is crucial for understanding their potential impact on Earth and for planning future space missions to study these objects. 

The project involves analyzing the set of observational data published by JPL’s Solar System Dynamics (SSD) group, to create models that can predict the diameter of asteroids based on their observable characteristics. Ultimately, the goal of the project is to improve our understanding of the asteroid population and their potential impact on our planet.

As someone who is truly fascinated by astronomy and data mining,, I am looking forward to working on the project that will help me learn captivating facts about the field I take interest in, not only from the textbook, but from the real astronomical data, and also enhance my skills both through careful attention to the details of data and machine learning, when it comes down to the actual use of the knowledge gained during the analysis.

__The current project is still in progress, but the outline of the final project will resemble this.__

To begin with, I will be exploring the dataset in search of interesting relationships between the features and explaining the relationships if possible. That is what I have done so far in the _Exploring the Data_ section. Then I will be preparing the data for training the diameter prediction models further in the project. This step comprises of feature engineering, encoding the categorical features, scaling the data, possibly, removing the outliers etc. After having done that, I am planning to focus more on building and fine-tuning the neural networks using Tensorflow's Sequential/Functional/Subclassing APIs. There is also a chance that I would give Light GBM or CatBoost a try, so that later I will be able to compare these different models and determine which one is the best for the task at hand.

<a id="EDA"></a>
## EDA
_Please note that if one is willing to see more of the takeaways from the EDA, then I suggest reading the whole notebook, which can be found either on [Github](https://github.com/Sofxley/asteroid-diameter-prediction/blob/main/asteroid_diameter_prediction.ipynb) or [Kaggle](https://www.kaggle.com/code/blulypsee/asteroid-diameter-prediction-eda/notebook)._

Here I present some of the highlights from the EDA. 	

<img src="https://raw.githubusercontent.com/Sofxley/asteroid-diameter-prediction/main/images/diameter_vs_class.png" width="600" align="right"/>
<img src="https://github.com/Sofxley/asteroid-diameter-prediction/blob/main/images/condition_code_distrib.png" width="600" align="right"/>

__Asteroid Classes, NEAs, PHAs__
* In the first figure on the right we see that most of the objects which are considered potentially hazardous are only NEAs of classes AMO, APO and ATE, these are the asteroids that either cross the Earth's orbit or move close to it in their orbits. We see that most of these asteroids have very low diameters from 0.1 km to 1 km, with the median of 0.56 km for AMO, 0.59 km for APO and 0.32 km for ATE.
* From other plots not shown here we also know that there are much more non-dangerous asteroids than those that may be threatening, and even though there are 850 NEAs, only a small fraction of them is considered potentially hazardous.

__Condition Code__
* The condition code ranges from 0 to 9. The condition code of 0 or 1 indicates a well-determined orbit, while codes 2-4 indicate a less certain orbit, and codes 5-9 indicate a poorly determined orbit.
* In the lower plot on the right we see that, fortunately, we have a lot of well-determined orbits from almost every class, but we also have a couple of thousands asteroids with poorly determined orbits and these are mosly MBAs, OMBs and IMBs. Since TNOs and CENs orbits are usually large, making it difficult to determine them correctly and also decreasing the opportunities to observe these asteroids, it makes sense that some TNOs and CENs have less certain orbits.
* Some classes of asteroids, such as NEAs, may be of particular interest to researchers and therefore may have more observational data and a lower condition code due to the efforts of astronomers to accurately determine their orbits, as we see for ATEs, AMOs and APOs. However, there are still some of NEAs such as APOs and AMOs with uncertain orbits. The possible reason for that is that they have really small diameters and may be really difficult to observe, not to mention their short orbital periods.
* Overall, there is no specific relationship between the class and condition code, condition code is rather dependent on the number of observations, and one find the exploration of this feature in the notebook.

<img src="https://raw.githubusercontent.com/Sofxley/asteroid-diameter-prediction/main/images/inclination_distrib.png" width="600" align="right"/>
<img src="https://raw.githubusercontent.com/Sofxley/asteroid-diameter-prediction/main/images/inclination_vs_eccentricity.png" width="600" align="right"/>

__Inclination & Eccentricity__
* The upper plot on the right shows that most of the eccentricities range from 0 deg to 25 deg. We notice that the majority of asteroids have either eccentricities from 0.05 to 0.2 and inclinations from around 2 deg to 7 deg, or eccentricitties lower than 0.15 and inclinations somewhere around 10 deg. This may be a hint telling us that most of the asteroids in our dataset are MBAs (low eccentricity due to stable orbit). Not to mention that for these values of inclination we have the range for semi-major axis from 0.5 au to 3.5 au. Such semi-major axes commonly have MBAs, OMBs, IMBs, APOs, MCAs, AMOs, ATEs.
* Therefore, the relationship between an asteroid's inclination and eccentricity can provide important information about the class asteroid belongs to, and thus, its overall characteristics.

From the lower plots on the right this can be concluded:
* MBAs typically have low eccentricities and inclinations, which means their orbits are relatively stable.
* MCAs usually have low-inclination but high-eccentricity orbits, because they may have been perturbed by the gravity of Mars.
* NEAs have orbits that bring them close to or cross the orbit of Earth. As a result, they tend to have higher eccentricities and inclinations than MBAs. Certain types of NEAs such as ATEs tend to have higher inclinations than other.
* TJNs are located in stable regions around the Lagrange points of Jupiter's orbit, and they typically have low eccentricities and inclinations. 
* CENs are objects that orbit the Sun between Jupiter and Neptune, and they typically have high inclinations and eccentricities, since many CENs are believed to have originated in the Kuiper Belt. It suggests that they may have been gravitationally scattered by Neptune or other large planets.
* TNOs have a wide range of eccentricities and inclinations. Some TNOs have highly elliptical and inclined orbits for the same reasons as CENs have too.

<img src="https://raw.githubusercontent.com/Sofxley/asteroid-diameter-prediction/main/images/perihelion_vs_aphelion.png" width="600" align="left"/>
<img src="https://github.com/Sofxley/asteroid-diameter-prediction/blob/main/images/perihelion_vs_aphelion_zoom.png" width="600" align="left"/>

__Perihelion & Aphelion distances__
* The relationship between aphelion distance and perihelion distance for asteroids of different classes can vary. However, in general, the greater the eccentricity of an asteroid's orbit, the larger the difference between its aphelion and perihelion distances.
* NEAs: all have highly eccentric orbits, which means that their aphelion distance and perihelion distance can differ significantly. ATEs have orbits entirely inside the Earth's orbit but still close to it, typically have aphelion distances that are not significantly larger than their perihelion distances. For APOs, which have orbits that are mostly inside the Earth's orbit and cross it, the aphelion distance is typically much larger than the perihelion distance. For AMOs, which have orbits that cross Earth's orbit from the outside, the aphelion distance can be significantly larger than the perihelion distance as well.
* IMBs and OMBs have orbits that are less eccentric than those of NEAs and TNOs. Therefore, their aphelion distance and perihelion distance are typically more similar in value.
* MBAs: Their orbits are mostly circular, with eccentricities typically less than 0.3. This means that their aphelion and perihelion distances are relatively close, therefore their orbits should be relatively stable and predictable.
* MCAs have orbits that cross the orbit of Mars, which means that their perihelion distances are closer to the Sun than those of IMBs and OMBs. Their aphelion distances can be significantly larger than their perihelion distances depending on the specific orbit of the asteroid.
* TJNs have orbits that are stabilized by the gravity of Jupiter, which means that their orbits are relatively stable and not highly eccentric. Therefore, their aphelion and perihelion distances are typically not very different from each other.
* CENs and TNOs: They have highly elliptical orbits, with eccentricities greater than 0.3. This leads to them having large differences between aphelion and perihelion distances. Their orbits are rather chaotic and hard to predict due to that.

<img src="https://raw.githubusercontent.com/Sofxley/asteroid-diameter-prediction/main/images/aphelion_vs_semimajor.png" width="600" align="right"/>
<img src="https://raw.githubusercontent.com/Sofxley/asteroid-diameter-prediction/main/images/albedo_vs_diameter.png" width="600" align="right"/>

__Aphelion distance & Semi-major Axis__
* Since the semi-major axis of an asteroid is the average distance between the asteroid and the sun, it determines the overall shape and size of the asteroid's elliptical orbit. The aphelion distance is the point at which the asteroid is farthest from the sun in its orbit, and this distance is directly related to the semi-major axis of the ellipse. Specifically, the aphelion distance is equal to $a(1+e)$.
* In summary, the aphelion distance of an asteroid is related to its semi-major axis through Kepler's laws of planetary motion, specifically through the shape and size of the asteroid's elliptical orbit. The upper plots on the right actually confirm this.

__Absolute Magnitude & Albedo__
* In general, smaller asteroids have higher absolute magnitudes (i.e., are darker), while larger asteroids have lower absolute magnitudes (i.e., are brighter). Albedo also plays a role, as objects with higher albedo (i.e., they reflect more light) have lower absolute magnitudes.
* If we were to look at two lower plots, we would conclude that MBAs have a relatively uniform distribution of absolute magnitudes, with a mean value of around $H=13$. There also can be noticed a slight trend for smaller asteroids to have higher absolute magnitudes, and for larger asteroids to have lower absolute magnitudes. This class spans all albedo values. Most of the time asteroids with albedo around 14 may have high albedos (greater than 0.6), but MBAs with $H > 14$ and $H < 14$ usually have $albedo < 0.6$. And the brighter the MBA is, the larger it is, then its albedo gets lower.
* IMBs are known for having high albedos compared to other asteroid classes. This is because they are composed of silicate rocks, which are more reflective than other materials such as carbonaceous chondrites that are found in outer main belt asteroids. Additionally, many IMBs have undergone space weathering, a process that can make their surfaces brighter by removing the darker outer layer of material and revealing the brighter interior.
* OMBs and MCAs usually have low albedos and follow the general rule of lower H leading to greater size.
* NEAs seem to have higher albedos when they are brighter (absolute magnitude is lower) and larger. Still, despite their proximity to the Sun, their albedos are much lower than these of MBAs sometimes, and never are higher than 0.6.
* For TJNs, there is a correlation between absolute magnitude and diameter, with brighter asteroids generally being larger, and also having higher albedos.
* The general rule applies to CENs too, since brighter CENs are often larger too, but they do not neccessarily have an increase in albedo. Most of them have really low albedos. They have a relatively uniform distribution of absolute magnitudes, with a mean value of around $H=10$.
* TNOs have a wide range of absolute magnitudes. Brighter TNOs generally are larger too, but not less absorptive.

<img src="https://raw.githubusercontent.com/Sofxley/asteroid-diameter-prediction/main/images/moid.png" width="600" align="right"/>

__Minimum Orbit Intersection Distance (MOID)__
* NEAs are more likely to have smaller MOIDs due to their proximity to Earth and their orbits that intersect Earth's orbit. 
* MBAs generally have larger MOIDs because they have more circular orbits and are located farther from Earth.
* Only NEAs whose $moid < 0.05$ and whose $H <= 22.0$ may usually be considered PHAs. And our plot on the right tells that asteroids in our dataset are entirely in accord with the statement.

__Bivariate Distributions__
* The last figure shows the probability of two given variables occurring. For example, we see that most of the asteroids have inclinations below 15. deg, and these occurrences spread over all values of LAN and argument of perihelion.

<img src="https://raw.githubusercontent.com/Sofxley/asteroid-diameter-prediction/main/images/bivariate_distributions.png" align="right"/>
