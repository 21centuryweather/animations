# Animation demo

*ECR day, 21st Century Weather annual workshop 2025, Launceston*

This repository demonstrates some simple animation functions, applied to gridded weather data in a python/jupyterlab environment. 

## `Animation_notebook.ipynb`

Uses a combination of matplotlib/ImageMagick, hvplot, and xmovie to plot ACCESS rAM3 output and Himawari data.

## `Animation_era5.ipynb`

Uses a matplotlib/ImageMagick to plot some ERA5 data

## Set up and environment

Sample data to use these notebooks is on `gb02`, so you will need access to that and to add gb02 in the `Storage` option of your ARE session. 

If you'd like to plot more Himawari data, it is on `ra22`, and ERA5 data is on `rt52`. 

These notebooks are intended to run in the `xp65` conda environment (any `analysis3`, although hvplot works as intended in older versions like `analysis3-24.12`)

You should use the following settings in your ARE session

* `Compute`: Medium
* `Project`: Your usual compute allocation
* `Storage`: gdata/xp65+gdata/gb02
* `Module directories`: /g/data/xp65/public/modules/
* `Modules`: conda/analysis3-24.12 (older version so hvplot method works as expected)

