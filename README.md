# FireTowerPlacement

# Fire Tower Placement and Visibility Analysis

This project provides a Python script to analyze and visualize the best positions for fire towers on a given elevation map. The script uses the GDAL library to handle GIS data and the SciPy library to perform visibility analysis.

## Features

- **Load Elevation Map**: Load a Digital Elevation Model (DEM) file.
- **Visibility Analysis**: Determine visibility points from a given elevation map based on a specified tower height.
- **Fire Tower Placement**: Select the best positions for a specified number of fire towers based on the elevation map.
- **GeoTIFF Output**: Save the selected fire tower positions to a new GeoTIFF file.
- **Visualization**: Visualize the elevation map and fire tower positions using Matplotlib.

## Requirements

- Python 3.x
- NumPy
- SciPy
- Matplotlib
- GDAL

## Installation

1. Install the required Python libraries:

   ```bash
   pip install numpy scipy matplotlib gdal
