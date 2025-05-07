### Produce Stagnated Water Input

#### Note: The scripts under this directory is running in a new environment other than mofes-env. 

Here we will walk you through the process of turning ADCIRC output (fort.63 and fort.64) for the Neches River case study produced by Wichitrnithed, Chayanon, et al.[1] into the input of MOFES (in netCDF). We need netCDF output (fort.63.nc and fort.64.nc) from Loveland, Mark, et al.[2] for this process. 

ADCIRC output can be replaced by the output of any other flooding models for your region, as long as the post-processed stagnated field has a netCDF format.


#### Instructions:
**optional** Go to [Kalpana](https://github.com/ccht-ncsu/Kalpana.git) for their latest updates.

##### 1. Obtain Kalpana

Go to my repository (an older verion of Kalpana I forked earlier) [old kalpana](https://github.com/largeseabass/Kalpana.git) and download the repository.

##### 2. Set up Conda Environment

Using the envionment.yml in this directory to set up your Kalpana environment.

##### 3. Obtain ADCIRC output

Contact us if you would like a copy.

##### 4. Set up your working directory

Inside your Kalpana folder, create a subfolder and this is where you will download the two files (stagnated.py and generate_stagnated_DG_output.ipynb) in this directory to. In this way, you could call all the Kalpana functions without installing the package, and also enjoying the functions I wrote in stagnated.py.

##### 5. Change all the paths

Go through all the path and directories inside generate_stagnated_DG_output.ipynb, change them to yours.


##### 6. Enjoy

Then you can set the enviornment to env_kalpana_v1 and click run-all.

[1] Wichitrnithed, Chayanon, et al. "A discontinuous Galerkin finite element model for compound flood simulations." Computer Methods in Applied Mechanics and Engineering 420 (2024): 116707.
[2] Loveland, Mark, et al. "Developing a modeling framework to simulate compound flooding: when storm surge interacts with riverine flow." Frontiers in Climate 2 (2021): 609610.