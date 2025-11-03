# Open Repository for Reproducible PhD Research and Publications

This repository contains the data, scripts, and references necessary to reproduce the results from the PhD research:

**PhD Thesis:** [*Meander Dynamics in the Antarctic Circumpolar Current*](https://figshare.utas.edu.au/articles/thesis/Meander_dynamics_in_the_Antarctic_Circumpolar_Current/25143134?file=45294115)  
**Author:** Jan Jaap Meijer  
**Date:** April 2023  

The thesis consists of an introductory chapter, three research chapters, and a concluding chapter.  
Each of the three research chapters has resulted in a peer-reviewed publication.  
The corresponding analysis scripts, datasets, or dataset references required to reproduce these results are provided in this repository.

---

### 🚀 Publications and Associated Directories

| Directory | Publication | DOI / Link |
|------------|--------------|-------------|
| `01_manuscript` | *Dynamics of a Standing Meander of the Subantarctic Front Diagnosed from Satellite Altimetry and Along-Stream Anomalies of Temperature and Salinity* | [10.1175/JPO-D-21-0049.1](http://doi.org/10.1175/JPO-D-21-0049.1) |
| `02_manuscript` | *Deep Cyclogenesis and Poleward Heat Advection beneath a Standing Meander in the Subantarctic Front* | [10.1175/JPO-D-25-0016.1](https://doi.org/10.1175/JPO-D-25-0016.1) |
| `03_manuscript` | (In preparation / forthcoming) | — |

---

### 🧭 Getting Started

The analyses use standard open-source Python tools for oceanographic data processing. To reproduce the analyses and figures in this repository, you can create the Conda environment using the provided configuration file `ocean3.yml`:

```bash
# Create the environment
conda env create --file ocean3.yml

# Activate the environment
conda activate ocean3
```

## 1. `01_manuscript` — File Structure and Data

This manuscript is based on *in situ* and satellite altimetry data.  
The data and analysis are organized as follows:

    01_manuscript/
    ├── data/
    │ ├── au9706/
    │ ├── ss9802/
    │ └── external/
    └── src/
    | ├── 00_data/
    | ├── 01_models/
    | └── 02_analysis/


**Data folder (`data/`)**  
- Contains *in situ* hydrographic observations collected during surveys of the Subantarctic Front aboard the RSV *Aurora Australis* (`au9706`) and RV *Southern Surveyor* (`ss9802`).  
- Includes additional external datasets (e.g., satellite altimetry, bathymetry, etc.).

**Source folder (`src/`)**  
- `00_data/`: Scripts for converting *in situ* data to NetCDF format and computing TEOS-10 oceanographic variables using [GSW-Python](https://github.com/TEOS-10/GSW-Python).  
- `01_models/`: Scripts for calculating gradient wind velocities and quasi-geostrophic vorticity terms.  
- `02_analysis/`: Scripts for the final analyses and figures presented in the manuscript.

---

## 2. `02_manuscript` — File Structure and Data

This manuscript uses output from the **ACCESS-OM2** global ocean–sea ice simulation at 1/10° resolution.  
Only a subset of the domain, covering the region where the Subantarctic Front crosses the Southeast Indian Ridge, is required for reproducing the results.

    02_manuscript/
    ├── data/
    ├── notebooks/
    └── src/


**ACCESS-OM2 model output:**  
The standard model data can be accessed via the [ACCESS-OM2 data portal](https://geonetwork.nci.org.au/geonetwork/srv/eng/catalog.search#/metadata/f1296_4979_4319_7298).

**Pre-processed data:**  
For convenience, pre-computed NetCDF files containing additional oceanographic variables (pressure, geostrophic and gradient wind velocities, and quasi-geostrophic vorticity terms) are available here:  
👉 [Zenodo dataset – Additional variables from ACCESS-OM2 output](https://doi.org/10.5281/zenodo.14575540)

To reproduce the results and figures:  
1. Create the NetCDF datasets yourself with the scripts in the `src` folder and the standard ACCESS-OM2 model output.
2. Or download the datasets from Zenodo in the links above.  
3. Place them in the `data/` directory.  
4. Run the analysis scripts located in the `notebooks/` folder.

---

### 📚 Citation and Acknowledgement

If you use the scripts, data, or methods provided in this repository in your own work,
please cite the corresponding publication using its DOI, or include a reference in your bibliography.

For example:

#### First manuscript
```bibtex
@article {meijer2022,
      author = "Jan Jaap Meijer and Helen E. Phillips and Nathaniel L. Bindoff and Stephen R. Rintoul and Annie Foppert",
      title = "Dynamics of a Standing Meander of the Subantarctic Front Diagnosed from Satellite Altimetry and Along-Stream Anomalies of Temperature and Salinity",
      journal = "Journal of Physical Oceanography",
      year = "2022",
      publisher = "American Meteorological Society",
      address = "Boston MA, USA",
      volume = "52",
      number = "6",
      doi = "10.1175/JPO-D-21-0049.1",
      pages = "1073 - 1089",
      url = "https://journals.ametsoc.org/view/journals/phoc/52/6/JPO-D-21-0049.1.xml"
}
```

#### Second manuscript
```bibtex
@article {meijer2025,
      author = "Jan Jaap Meijer and Helen E. Phillips and Stephen R. Rintoul and Nathaniel L. Bindoff and Annie Foppert",
      title = "Deep cyclogenesis and poleward heat advection beneath a standing meander in the Subantarctic Front",
      journal = "Journal of Physical Oceanography",
      year = "2025",
      publisher = "American Meteorological Society",
      address = "Boston MA, USA",
      doi = "10.1175/JPO-D-25-0016.1",
      url = "https://journals.ametsoc.org/view/journals/phoc/aop/JPO-D-25-0016.1/JPO-D-25-0016.1.xml"
}
```

Thank you for acknowledging this work — it helps support open and reproducible oceanographic research.

### ✅ Notes

- This repository adheres to open-science and reproducibility principles.  
- All scripts are written to be fully transparent and reproducible with publicly available data.  
- Please cite the corresponding publications when reusing the data or methods.

---
