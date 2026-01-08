
import rasterio

from abc import ABC
from pathlib import Path
import numpy as np

CASTING_DICT = {
    np.uint32 : rasterio.float32,
    np.uint16 : rasterio.float32,
    np.uint64 : rasterio.float64,
    np.int16 : rasterio.float32,
    np.int32 : rasterio.float32,
    np.int64 : rasterio.float64
}


class ImageLoader(ABC):
    # class to read image and image header
    def __init__(self, data_filename, header_file_name=None):
        self.data_filename = data_filename
        self.header_file_name = header_file_name
    
    def read_image(self):
        "Implement this using Inheritance"
        pass
    
    def read_header(self):
        "Implement this using Inheritance"


class GeotifImageLoader:
    def __init__(self, image_filename, image_subset=None, header_filename=None):
        self.image_filename = Path(image_filename)
        self.header_filename = header_filename
        self.image_subset = image_subset
        self.data = None
        self.header = None

        self.profile = None

    def get_image(self):
        if self.data is None:
            self.data = self.__read_image_data()
            if self.data.dtype in list(CASTING_DICT.keys()):
                self.data = self.data.astype(CASTING_DICT.get(self.data.dtype, rasterio.float32))
        print("Image Loading Dtype : ", self.data.dtype)
        return self.data.copy()
    
    def get_header(self):
        if self.header == None:
            self.header = self.__read_image_header_data()
        return self.header


    def __read_image_data(self):
        # TODO: implement load only subset feature
        if not self.image_filename.exists():
            raise Exception("File not Found")
        
        _temp_data = None
        with rasterio.open(self.image_filename) as hyspec_image:
            # _temp_data = hyspec_image.read() #(bands, row, cols)
            _temp_data =self.__get_transposed_image(hyspec_image)
            self.profile = hyspec_image.profile
        return _temp_data
    
    def __get_transposed_image(self, hyspec_image):
        '''returns image in format of (height, bands, width)'''
        return np.transpose(hyspec_image.read(), (1, 0, 2)) #(height, bands, width)

    def __read_image_header_data(self):
        pass