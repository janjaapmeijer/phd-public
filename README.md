# Open Repository for Reproducible PhD Research and Publications

This repository contains the result from the PhD research:

PhD Thesis title: [Meander Dynamics in the Antarctic Circumpolar Current](https://figshare.utas.edu.au/articles/thesis/Meander_dynamics_in_the_Antarctic_Circumpolar_Current/25143134?file=45294115)\
Author: Jan Jaap Meijer
Date: April, 2023

The thesis contains an Introductionary chapter, three Research chapters and a Concluding chapter.

The three Research chapters have led to three individual manuscripts that are now published. The analysis scipts and datasets or reference to datasets for reproducing the results in these manuscripts are stored in this repository . 

 - `01_manuscript`: [Dynamics of a Standing Meander of the Subantarctic Front Diagnosed from Satellite Altimetry and Along-Stream Anomalies of Temperature and Salinity](http://doi.org/10.1175/JPO-D-21-0049.1)
 - `02_manuscript`: [Deep Cyclogenesis and Poleward Heat Advection beneath a Standing Meander in the Subantarctic Front](https://doi.org/10.1175/JPO-D-25-0016.1)
 - `03_manuscript`: []()

## `01_manuscript` - file structure

In order to reproduce the results of this manuscript, data is required from primarily insitu and satellite altimetry data. This data is structured in the `data` folder and seperated from the source `src` scripts to perform the analysis, as show in this file structure:

    01_manuscript/
    |-- data/
      |-- au9706/
      |-- ss9802/
      |-- external
    |-- src/
      |-- 00_data/
      |-- 01_models/
      |-- 02_analysis/

The `data` folder is a collection of:
 - shipboard insitu data during the standing meander survey of the Subantarctic Front from both the RV Southern Surveyor (`ss9802`) and the RSV Aurora Australis (`au9706`)
 - external data (satellite altimetry, bathymetry etc.)

The `src` folder is subdived into three subfolders:
 - in `00_data` are all the scripts necessary to store the insitu data in netcdf format and calculate some TEOS-19 Oceanographic variables with the use of the [GSW-Python](https://github.com/TEOS-10/GSW-Python) package.
 - the scripts for calculating the gradient wind velocities and quasi-geostrophic vorticity terms are stored in `01_models`  
 - and in `02_analysis` are the scripts that produced the rest of the analysis and figures for the manuscript.


## 02_manuscript - file structure

This manuscript is based on the output of the ACCESS-OM2 1/10 degree global simulation, but only a small subset of the data located where the Subantarctic Front flows over the Southeast Indian Ridge is required to reproduce the results.

    02_manuscript/
    |-- data/
    |-- notebooks/
    |-- src/

The scripts in the `src` folder are needed to calculate additional oceanographic variables (e.g. pressure, geostrophic and gradient wind velocities and the quasi-geostrophic vorticity terms) apart from the stardard model output. Here you can find access to the standard model output variables:\
[ACCESS-OM2 data](https://geonetwork.nci.org.au/geonetwork/srv/eng/catalog.search#/metadata/f1296_4979_4319_7298)

However, if you are just interested in reproducing the results and figures for the manuscript than it is easier to download the netcdf datasets where these additional oceanographic variables are stored in: 

[Additional oceanographic variables calculated from ACCESS-OM2 output (offline)](https://doi.org/10.5281/zenodo.14575540)

Best practise is to save these netcdf files in the `data` folder. In this way, the analysis scripts in the notebooks folder can find the netcdf files easily.
