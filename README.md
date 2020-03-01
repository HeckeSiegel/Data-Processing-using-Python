# Data Processing using Python
>Computational tools for processing measurement data of atmospheric trace gas concentrations using Pyhton.

## Installation
### Clone
- Clone this repo using 'https://github.com/HeckeSiegel/Data-Processing-using-Python.git'
### Setup
- Save the .txt files in the same folder as the .py files
- Make sure you have Python 2.7 installed on your local machine
- Run the .py files using for example IDLE (built in Python IDE)
## Linear Regression
### Data
>NO2 concentrations measured in Hongkong in 3 different locations (ascending height) using DOAS (differential optical absorption spectroscopy): 
- Street
- Roof of School
- Roof of Langham Hotel
>The measurements were done during the course of a day, from 8 AM to 7 PM.
### Processing Methods
>Linear fits using different least square approximations

![regression](https://github.com/HeckeSiegel/Data-Processing-using-Python/blob/master/linear_regression.png)

## Higher Order Linear Regression
### Data
>NO2 concentrations measured in Munich with measurement error
>2-year average NO2 concentrations
### Processing methods
>Comparison of 1st and 2nd order linear regression
>Error terms for both linear fits

![higher_order_regression](https://github.com/HeckeSiegel/Data-Processing-using-Python/blob/master/higher_order_linear_regression.png)
