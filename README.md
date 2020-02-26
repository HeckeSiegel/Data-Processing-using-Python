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
>Linear fit using least square approximations
- Ordinary least square
![ordinary](https://github.com/HeckeSiegel/Data-Processing-using-Python/blob/master/ordinary.png)
- Weighted Least Square
![weighted](https://github.com/HeckeSiegel/Data-Processing-using-Python/blob/master/weighted.png)
- Total Least Square
![total](https://github.com/HeckeSiegel/Data-Processing-using-Python/blob/master/total.png)
- Error Variances for all 3 methods compared using the data of Langham location
![error](https://github.com/HeckeSiegel/Data-Processing-using-Python/blob/master/error.png)
