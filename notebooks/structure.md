In this folder a number of jupyter notebooks guide you through the analysis of the SS9802 dataset. It starts of with the 
notebook density_surfaces. Here, we will analyse the isopycnals that are measured from the CTD profiles. We look at how 
isopycnals change from upstream to downstream and across the standing meander. 

This will then feed into the notebook layer_depth, where we analyse the mixed layer depth and layer thickness. The mixed 
layer depth helps us to create a velocity field in the surface layer and acts as a boundary to the velocity field in the lower layers. The layer 
thickness helps us in determining the vorticity fields in the lower layers based on the potential vorticity equation. 

In the optimal_interpolation notebook, we start by simple examples of optimal interpolated fields, we extend the analysis 
by implementing multivariate analysis and constraints in the form of geostrophy and potential vorticity, by adjusting 
the background error covariance matrix.

1. density surfaces
2. layer_depth
3. optimal_interpolation
4. velocity
    - in the surface layer (multivariate optimal interpolation)
        - background field: u_SSH, v_SSH
        - observations field averaged to MLD: u_ADCP, v_ADCP
        - analysis field: absolute velocity field in surface/ mixed layer (u_abs, v_abs)
    -  extend velocities from surface to full column profiles
        - combine absolute velocities from ADCP with baroclinic velocities from CTD **(flow-direction problem!!, uncertainty in magnitude)** 
        - use vertical covariance in optimal interpolation to account for turning of the velocity vector and use optimal interpolated surface layer as background field
    - use velocity in density layer between 26.8 and 27.2 kgm3 to calculate relative vorticity
5. vorticity
    - use optimal interpolated ADCP velocity field to calculate relative vorticity in the surface layer
    - use mixed layer depth (from model?) as background field to optimal interpolate layer thickness (mixed layer)
    - calculate planetary vorticty/ coriolis parameter
    
