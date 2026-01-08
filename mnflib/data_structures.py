import pickle
import numpy as np

from typing import List
from enum import Enum
from dataclasses import dataclass
from scipy.linalg.blas import sgemv, ssyrk, ssyr, sger

from pprint import pprint

class TransformDirection(Enum):
    RUN_BOTH = 'RUN_BOTH'
    RUN_FORWARD = 'RUN_FORWARD'
    RUN_INVERSE = 'RUN_INVERSE'

@dataclass
class MNFConfig:
    direction: TransformDirection
    basefilename: str
    bands: int
    samples: int
    lines: int
    percentageOfBandsInInverse: float
    noiseMatrixCalculation: str

@dataclass
class MNFResult:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    image_mean: np.ndarray
    noise_mean: np.ndarray
    image_covariance: np.ndarray
    noise_covariance: np.ndarray

class HyspexHeader:
    def __init__(self):
        self.samples:int = 0
        self.bands:int = 0
        self.lines:int = 0
        self.offset:int = 0
        self.wlens:List[float] = 0
        self.datatype:int = 0


class ImageSubset:
    def __init__(self):
        self.startsSamp:int = 0
        self.endSamp:int = 0
        self.startLine:int = 0
        self.endLine:int = 0
        self.startBand:int = 0
        self.endBand:int = 0


class ImageStatistics:
    def __init__(self, bands):
        self.n:int = 0  # number of pixels so far summed over
        self.C = None  # covariances Matrix
        self.means = None # means
        self.bands:int = bands
        self._init(bands)
    
    def _init(self, bands):
        self.C = np.zeros((self.bands, self.bands))
        self.means = np.zeros(self.bands)
        
    def get_means(self):
        return self.means

    def get_cov(self):
        return self.C / self.n 

    def write_mean_to_file(self, location):
        with open(location, 'wb') as fp:
            pickle.dump(self.get_means(), fp)
            np.savetxt(location + ".txt", self.means)

    def write_covariance_to_file(self, location):
        with open(location, 'wb') as fp:
            pickle.dump(self.get_cov(), fp)
            np.savetxt(location + ".txt", self.C)


    def read_mean_from_file(self, location):
        with open(location, 'rb') as fp:
            self.means = pickle.load(fp)
    
    def read_covariance_from_file(self, location):
        with open(location, 'rb') as fp:
            self.C = pickle.load(fp)   

    def write_to_file(self, location):
        self.write_covariance_to_file(location + "_cov.pkl")
        self.write_mean_to_file(location + "_mean.pkl")

    def read_from_file(self, location):
        self.read_covariance_from_file(location + "_cov.pkl")
        self.read_mean_from_file(location + "_mean.pkl")
        self.n = 1.0

    def update_with_line(self, line, samples):
        # new_n = self.n + samples
        # line_mean = np.mean(line, axis=1)
        # line_subtracted_by_mean = line - line_mean[:, None]
        # self.C += line_subtracted_by_mean @ line_subtracted_by_mean.T
        # mean_delta = line_mean - self.means
        
        # self.means += (samples/new_n)*mean_delta
        # self.C += samples*(self.n / new_n) * np.outer(mean_delta, mean_delta)
        # self.n = new_n

        self.n += samples
        # y := alpha * A * x + beta * y  (general matrix-vector multiplication)
        line_mean =  1 / samples * sgemv(alpha=1, a=line, x=np.ones(samples))
        # sger : A := alpha * x * transpose(y) + A (general rank-1 update)
        line_subtracted_by_mean = line - np.outer(line_mean, np.ones(samples, dtype=line.dtype)) # sger(alpha=-1.0, x=line_mean, y=np.ones(samples, dtype=line.dtype), a=line)
        # C := alpha * A * transpose(A) + beta * C (general rank-k update)
        self.C = self.C + line_subtracted_by_mean @ line_subtracted_by_mean.T # ssyrk(alpha=1.0, a=line_subtracted_by_mean, lower=0, beta=1.0,  c=self.C)
        mean_delta = line_mean - self.means # both are numpy arrays of dimention 1
        self.means = self.means + samples * (line_mean - self.means) / self.n
        # C := alpha * x * transpose(x) + A (symmetric rank-1 update)
        # self.C = ssyr(alpha=samples*(self.n - samples)/self.n, x=mean_delta, a=self.C)
        self.C += samples*(self.n - samples)/self.n * np.outer(mean_delta, mean_delta)





class ImageStatisticsFull:
    def __init__(self, bands):
        self.C = None  # covariances Matrix
        self.means = None # means
        self._init(bands)
    
    def _init(self, bands):
        self.C = np.zeros((bands, bands))
        self.means = np.zeros(bands)
        
    def get_means(self):
        return self.means

    def get_cov(self):
        return self.C

    def write_mean_to_file(self, location):
        with open(location, 'wb') as fp:
            pickle.dump(self.get_means(), fp)
            np.savetxt(location + ".txt", self.means)

    def write_covariance_to_file(self, location):
        with open(location, 'wb') as fp:
            pickle.dump(self.get_cov(), fp)
            np.savetxt(location + ".txt", self.C)


    def read_mean_from_file(self, location):
        with open(location, 'rb') as fp:
            self.means = pickle.load(fp)
    
    def read_covariance_from_file(self, location):
        with open(location, 'rb') as fp:
            self.C = pickle.load(fp)   

    def write_to_file(self, location):
        self.write_covariance_to_file(location + "_cov.pkl")
        self.write_mean_to_file(location + "_mean.pkl")

    def read_from_file(self, location):
        self.read_covariance_from_file(location + "_cov.pkl")
        self.read_mean_from_file(location + "_mean.pkl")

    def update(self, image):
        self.means = np.average(image, axis=1)
        self.C = np.cov(image)
