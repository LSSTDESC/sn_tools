# sn_tools 

 * \_\_init\_\_.py
 * sn_analyze_simu.py
 ## sn_cadence_tools.py ##

|name | type | task|
|----|----|----|
|ReferenceData | class | handling SN LC templates |
|GenerateFakeObservations | class | generating fake observations|
|TemplateData | class | loading template LC|
|AnaOS | class | observing strategy analysis|
|Match_DD | function | matching data to DD fields| 


 ## sn_clusters.py ##
 
|name | type | task|
|----|----|----|
|ClusterObs | class | identifying clusters (RA,Dec) of data points|
|getVisitsBand | function | estimating the number of visits per band|
|getName | function | getting the field name corresponding to RA|

 ## sn_io.py ##

|name | type | task|
|----|----|----|
|Read_Sqlite | class| reading a sqlite file (from scheduler) and convert to numpy array|
| append|function |numpy array concatenation|
 |getMetricValues|function |reading and analyzing files from metrics|
 |geth5Data|function |loading the content of hdf5 files|
 |getLC|function |accessing a table in hdf5 file from key|
 |getFile|function |pointing to hdf5 file|
 |selectIndiv|function |applying a selection on an array of data|
 |select|function |applying few selections on an array of data|
 |loadFile|function |loading file according to the type of data it contains|
 |loopStack|function |load files and stack results  according to the type of data it contains|
 |convert_to_npy|function |stacking pandas df and converting to numpy array|
 |convert_save|function |convert and save output to hdf5 file|
 |remove_galactic_area|function |excluding galactic plane from data|
 |getObservations|function |extracting observations (db scheduler->numpy array)|

 * sn_lcana.py
 ## sn_obs.py ##

|name | type | task|
|----|----|----|
|PavingSky| class | paving the sky with rectangles|
|DataInside | class | select data points (RA,Dec) inside a area|
|DataToPixels | class | match observations to pixels|
|ProcessPixels | class | processing (metrics) on a set of pixelized data|
|ProcessArea | class | processing (metrics) on a given part of the sky|
|DDFields | function | defining DD fields|
|patchObs | function |getting informations and patches on the sky|
|getPix| function |grabbing pixels information|
|area| function |making a dict of coordinates|
|areap| function |making polygon out of coordinates|
|areap_diamond| function |defining diamonds|
|proj_gnomonic_plane| function |performing a gnomonic projection on a plane|
|proj_gnomonic_sphere| function |performing a gnomonic projection on a sphere|
|renameFields| function |renaming fields of a numpy array|
|fillCorresp| function |filling a dict used in renameFields|
|pixelate| function |pixelating the sky with data|
|season| function |estimating seasons |
|LSSTPointing| function |LSST focal plane|
|LSSTPointing_circular| function |LSST focal plane made circular|
|getFields_fromId| function |getting fields using fieldIds column|
|getFields| function |get a type of fields (DD or WFD) from a set of observations|


 ## sn_rate.py ##

|name | type | task|
|----|----|----|
|SN_Rate | class |Estimating production rate of typeIa SNe|


 ## sn_telescope.py ##

|name | type | task|
|----|----|----|
|Telescope | class |generating a telescope |
|get_val_decor|function|decorator to access class params|

 ## sn_throughputs.py ##

|name | type | task|
|----|----|----|
|Throughputs | class | handling instrument throughputs|

 ## sn_utils.py ##

|name | type | task|
|----|----|----|
|MultiProc|class|performing multiprocessing|
|GenerateSample|class|generating a sample of parameters for simulation|
|Make_Files_for_Cadence_Metric|class|producing files requested as input for the Cadence Metric|
|X0_norm|class|estimating X0s|
|DiffFlux|class|estimating flux derivatives wrt SN parameters|
|MbCov|class|estimating covariance matrix with mb|
|GetReference|class|loading reference data|
|Gamma|class|estimating gamma parameters vs mag and exposure time|

 ## sn_visu.py ##

|name | type | task|
|----|----|----|
|SnapNight | class | getting a snapshot of the (RA, Dec) pointings observed per night'
|CadenceMovie| class |displaying obs footprint vs time|
|fieldType | function|estimating the type of field (DD or WFD) according to the number of exposures|
|area | function|estimating area of a set of polygons (without overlap)|

 ## sn_calcFast.py ##

|name | type | task|
|----|----|----|
|LCfast | class | simulating supernovae light curves in a fast way|
|CalcSN|class|estimating SN parameters from light curve|
|CalcSN_df|class|estimate SN parameters from light curve|
|CovColor | class | estimating CovColor from lc using Fisher matrix element|
|sigmaSNparams|function| estimating SN parameter errors from Fisher elements|
|faster_inverse | function | inverting a matrix in a fast way|
