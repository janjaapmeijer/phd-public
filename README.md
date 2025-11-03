# Open Repository for Reproducible PhD Research and Publications

This repository contains the result from the PhD research:

PhD Thesis title: [Meander Dynamics in the Antarctic Circumpolar Current](https://figshare.utas.edu.au/articles/thesis/Meander_dynamics_in_the_Antarctic_Circumpolar_Current/25143134?file=45294115)\
Author: Jan Jaap Meijer
Date: April, 2023

The thesis contains an Introductionary chapter, three Research chapters and a Concluding chapter.

The three Research chapters have led to three individual manuscripts that are now published.

 - `01_manuscript`: [Dynamics of a Standing Meander of the Subantarctic Front Diagnosed from Satellite Altimetry and Along-Stream Anomalies of Temperature and Salinity](http://doi.org/10.1175/JPO-D-21-0049.1)
 - `02_manuscript`: [Deep Cyclogenesis and Poleward Heat Advection beneath a Standing Meander in the Subantarctic Front](https://doi.org/10.1175/JPO-D-25-0016.1)
 - `03_manuscript`: []()

## `01_manuscript` - file structure

    01_manuscript/
    |-- data/
      |-- au9706/
      |-- ss9802/
    |-- src/
      |-- 00_data/
      |-- 01_models/
      |-- 02_analysis/

The `data` folder is a collection of:
 - shipboard insitu data during the standing meander survey of the Subantarctic Front from both the RV Southern Surveyor (`ss9802`) and the RSV Aurora Australis (`au9706`)
 - external data (bathymetry etc.)

The `src` folder is subdived into three subfolders:
 - in `00_data` are all the scripts necessary to store the insitu data in netcdf format and calculate some TEOS-19 Oceanographic variables with the use of the [GSW-Python](https://github.com/TEOS-10/GSW-Python) package.
 - `01_models`
 - `02_analysis`


## 02_manuscript - file structure

    02_manuscript/
    |-- data/
    |-- notebooks/
    |-- src/


Please download the datasets below and put them in a folder named data. In this way you can skip retrieving the data from the ACCESS-OM2 simulations and you start using the analysis scripts in the notebooks folder.

Additional oceanographic variables calculated from ACCESS-OM2 output (offline)

However, if you are interested in calculating the additional oceanographic variables offline, for which you need the scripts int the src folder than you have to get access to the ACCESS-OM2 model data. How to get access to this data can be found here:

ACCESS-OM2 data


https://doi.org/10.5281/zenodo.14575540

    PhD/
    |-- data/ contains all the raw data files
    |-- docs/ contains all documents related to the project
    |-- figs/ contains images/ figures from papers and created by scripts
    |-- out/  contains any results from data analysis
    |-- src/ contains all scripts and functions for the analysis

