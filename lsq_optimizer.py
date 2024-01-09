import numpy as np
import matplotlib.pyplot as plt
import magpylib as magpy
from scipy.spatial.transform import Rotation as R
from scipy.optimize import lsq_linear
from scipy.ndimage import gaussian_filter

class lsq_optimizer:
    
    def __init__(self):
        self.map = None
        
    def setMap(self, filename):
        from pydicom import dcmread

        ds = dcmread(filename)
        arr = ds.pixel_array
        self.map = arr

        print("\nHere is the inputted field map:\n\n" + 
              str(arr) +  
              "\n\n" +
              f"Standard Deviation of inputted field_map: {np.std(arr)}" +
              "\n")
        plt.imshow(arr, cmap="gray")
        plt.show()

    def generateMap(self, target_area=(256, 256)):
        self.map = np.random.uniform(-100, 100, target_area)
    
    def generateZeroesMap(self, target_area=(256, 256)):
        self.map = np.zeros(target_area)

    def generateOnesMap(self, target_area=(256, 256)):
        self.map = np.ones(target_area)

    def simple_optimizer(self, target_area_x=128):
        X, Y, Z = np.mgrid[-5:5:256j, -5:5:256j, -5:5:256j]
        grid = np.stack([X, Y, Z], axis=3)
        
        coll = magpy.Collection(
            magpy.current.Loop(
                current=1, 
                diameter=10,
                position=[5, 0, 0],
                orientation=R.from_euler('y', 90, degrees=True),
            ),
        )

        magpy.show(coll)

        Bz1 = (coll[0].getB(grid) * (42.58) * (10 ** 3))[target_area_x, :, :, 2]

        X = lsq_linear(Bz1.reshape(-1, 1), self.map.reshape(-1)).x

        print("\nHere are the currents that optimize the correction of the inputted field map:\n\n" + 
              str(X) +  
              "\n")
        
        self.lsq_currents = X

        coll = magpy.Collection(
            magpy.current.Loop(
                current=self.lsq_currents[0], 
                diameter=10,
                position=[5, 0, 0],
                orientation=R.from_euler('y', 90, degrees=True),
            ),
        )

        Bz = (coll.getB(grid) * (42.58) * (10 ** 3))[target_area_x, :, :, 2]
        print("\nHere are the B magnitudes produced by the optimal currents:\n\n" + 
              str(Bz) +  
              "\n")
        
        corrected_B = self.map - Bz
        gaussfilt = gaussian_filter(corrected_B, sigma=30)
        print("\nAnd here is the corrected field_map using the loop collection:\n\n" + 
              str(corrected_B) +  
              "\n\n" +
              f"Standard Deviation of corrected field_map: {np.std(corrected_B)}" +
              "\n")
        
        plt.imshow(np.abs(gaussfilt), cmap='coolwarm', origin='lower')
        plt.show()

    def optimizer(self, target_area_x=128):
        X, Y, Z = np.mgrid[-5:5:256j, -5:5:256j, -5:5:256j]
        grid = np.stack([X, Y, Z], axis=3)

        coll = magpy.Collection(
            magpy.current.Loop(
                current=1, 
                diameter=10,
                position=[0, 0, 5],
            ),
            magpy.current.Loop(
                current=1, 
                diameter=10,
                position=[0, 0, -5],
            ),
            magpy.current.Loop(
                current=1, 
                diameter=10,
                position=[0, 5, 0],
                orientation=R.from_euler('x', 90, degrees=True),
            ),
            magpy.current.Loop(
                current=1, 
                diameter=10,
                position=[0, -5, 0],
                orientation=R.from_euler('x', 270, degrees=True),
            ),
            magpy.current.Loop(
                current=1, 
                diameter=10,
                position=[5, 0, 0],
                orientation=R.from_euler('y', 90, degrees=True),
            ),
            magpy.current.Loop(
                current=1, 
                diameter=10,
                position=[-5, 0, 0],
                orientation=R.from_euler('y', 270, degrees=True),
            )
        )

        magpy.show(coll)

        Bz1 = (coll[0].getB(grid) * (42.58) * (10 ** 3))[target_area_x, :, :, 2]
        Bz2 = (coll[1].getB(grid) * (42.58) * (10 ** 3))[target_area_x, :, :, 2]
        Bz3 = (coll[2].getB(grid) * (42.58) * (10 ** 3))[target_area_x, :, :, 2]
        Bz4 = (coll[3].getB(grid) * (42.58) * (10 ** 3))[target_area_x, :, :, 2]
        Bz5 = (coll[4].getB(grid) * (42.58) * (10 ** 3))[target_area_x, :, :, 2]
        Bz6 = (coll[5].getB(grid) * (42.58) * (10 ** 3))[target_area_x, :, :, 2]

        
        A = np.stack([Bz1, Bz2, Bz3, Bz4, Bz5, Bz6], axis=2).reshape(-1, 6)
        X = lsq_linear(A, self.map.reshape(-1)).x
        
        print("\nHere are the currents that optimize the correction of the inputted field map:\n\n" + 
              str(X) +  
              "\n")
        
        self.lsq_currents = X

        # End of optimization

        coll = magpy.Collection(
            magpy.current.Loop(
                current=self.lsq_currents[0], 
                diameter=2.5,
                position=[0, 0, 5],
            ),
            magpy.current.Loop(
                current=self.lsq_currents[1], 
                diameter=2.5,
                position=[0, 0, -5],
            ),
            magpy.current.Loop(
                current=self.lsq_currents[2], 
                diameter=2.5,
                position=[0, 5, 0],
                orientation=R.from_euler('x', 90, degrees=True),
            ),
            magpy.current.Loop(
                current=self.lsq_currents[3], 
                diameter=2.5,
                position=[0, -5, 0],
                orientation=R.from_euler('x', 270, degrees=True),
            ),
            magpy.current.Loop(
                current=self.lsq_currents[4], 
                diameter=2.5,
                position=[5, 0, 0],
                orientation=R.from_euler('y', 90, degrees=True),
            ),
            magpy.current.Loop(
                current=self.lsq_currents[5], 
                diameter=2.5,
                position=[-5, 0, 0],
                orientation=R.from_euler('y', 270, degrees=True),
            )
        )

        magpy.show(coll)

        Bz = (coll.getB(grid) * (42.58) * (10 ** 3))[target_area_x, :, :, 2]
        print("\nHere are the B magnitudes produced by the optimal currents:\n\n" + 
              str(Bz) +  
              "\n")
        
        corrected_B = self.map - Bz
        gaussfilt = gaussian_filter(corrected_B, sigma=30)
        print("\nAnd here is the corrected field_map using the loop collection:\n\n" + 
              str(corrected_B) +  
              "\n\n" +
              f"Standard Deviation of corrected field_map: {np.std(corrected_B)}" +
              "\n")
        
        plt.imshow(np.abs(gaussfilt), cmap='coolwarm', origin='lower')
        plt.show()

# Test
        
opt = lsq_optimizer()
opt.setMap("s17941/i2828748.MRDC.53")
opt.simple_optimizer()