import os
import rasterio
import numpy as np

def single_band_to_multiband(input_folder, output_file):
    tiff_files = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.tif')])
    
    if not tiff_files:
        raise FileNotFoundError("No TIFF files found in the specified directory.")

    # Read the first file to get the metadata
    with rasterio.open(tiff_files[0]) as src:
        meta = src.meta
        meta.update(count=len(tiff_files))  

    # Create the output multi-band TIFF
    with rasterio.open(output_file, 'w', **meta) as dst:
        for idx, file in enumerate(tiff_files, start=1):
            with rasterio.open(file) as src:
                band_data = src.read(1)  
                dst.write(band_data, idx)  # Write it as the idx-th band in the multi-band TIFF

    print(f"Multi-band TIFF created successfully: {output_file}")


input_folder = '/home/bhoj/Desktop/Practice/raster_analysis/20240824'
output_file = '/home/bhoj/Desktop/Practice/raster_analysis/multiband_output.tif'
single_band_to_multiband(input_folder, output_file)
