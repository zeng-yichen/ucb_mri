import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize

import magpylib as magpy

class optimizer:
    def __init__(self, filename):
        from pydicom import dcmread

        ds = dcmread(filename)
        arr = ds.pixel_array
        self.map = arr

        print("\nHere is the inputted field map:\n\n" + 
              str(arr) +  
              "\n\n" +
              f"Average deviation of inputted field_map from np.zeroes: {np.mean(np.abs(arr))}" +
              "\n")
        plt.imshow(arr, cmap="gray")
        plt.show()
    
    def optimize(self):
        # Example function to calculate magnetic field from current loops
        def calculate_magnetic_field(loop_parameters):
            # Calculate the magnetic field for given loop parameters
            # This is a complex calculation
            Y, Z = np.mgrid[-5:5:256j, -5:5:256j].transpose((0, 2, 1))
            grid = np.stack([np.zeros((256, 256)), Y, Z], axis=2)

            coll = magpy.Collection()
            for i in range(6):
                coll.add(
                    magpy.current.Loop(
                        current=loop_parameters[i*8],
                        diameter=loop_parameters[i*8+1],
                        position=loop_parameters[i*8+2:i*8+5],
                        orientation= R.from_euler("xyz", [loop_parameters[i*8+5], loop_parameters[i*8+6], loop_parameters[i*8+7]])
                    )
                )
            
            Bz = coll.getB(grid)[:, :, 2] * (42.58) * (10 ** 3)
            return Bz

        def objective_function(loop_parameters, desired_field):
            generated_field = self.map - calculate_magnetic_field(loop_parameters)
            return np.sum((desired_field - generated_field)**2)

        def std_objective_function(loop_parameters, desired_field):
            generated_field = calculate_magnetic_field(loop_parameters)
            return np.std(generated_field)

        # Define your desired magnetic field (e.g., an array of zeros)
        desired_field = np.zeros((256, 256))

        # Initial guess for loop parameters
        initial_guess = [0.000001, 15, 0, 0, 5, 0, 0, 0,
                         0.000001, 15, 0, 0, -5, 0, 0, 0,
                         0.000001, 15, 5, 0, 0, 0, np.pi/2, 0,
                         0.000001, 15, -5, 0, 0, 0, np.pi/2, 0,
                         0.000001, 15, 0, 5, 0, np.pi/2, 0, 0,
                         0.000001, 15, 0, -5, 0, np.pi/2, 0, 0]
                
        bounds = [(-2, 2), (5, 15), (-10, 10), (-10, 10), (-10, 10), (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi),
                  (-2, 2), (5, 15), (-10, 10), (-10, 10), (-10, 10), (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi),
                  (-2, 2), (5, 15), (-10, 10), (-10, 10), (-10, 10), (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi),
                  (-2, 2), (5, 15), (-10, 10), (-10, 10), (-10, 10), (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi),
                  (-2, 2), (5, 15), (-10, 10), (-10, 10), (-10, 10), (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi),
                  (-2, 2), (5, 15), (-10, 10), (-10, 10), (-10, 10), (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)]

        # Run optimization
        result = minimize(objective_function, initial_guess, args=(desired_field,), bounds=bounds)

        # Extract the optimal parameters
        optimal_parameters = result.x
        
        # Validate or use the result

        print(optimal_parameters)

        print("\nHere is the corrected field map:\n\n" + 
              str(self.map - calculate_magnetic_field(optimal_parameters)) +  
              "\n\n" +
              f"Average deviation of corrected field_map from np.zeroes: {np.mean(np.abs(self.map - calculate_magnetic_field(optimal_parameters)))}" +
              "\n")
        
    def optimizer(self):
        # Example function to calculate magnetic field from current loops
        def calculate_magnetic_field(loop_parameters):
            # Calculate the magnetic field for given loop parameters
            # This is a complex calculation
            Y, Z = np.mgrid[-5:5:256j, -5:5:256j].transpose((0, 2, 1))
            grid = np.stack([np.zeros((256, 256)), Y, Z], axis=2)

            coll = magpy.Collection()
            for i in range(6):
                coll.add(
                    magpy.current.Loop(
                        current=loop_parameters[i*8],
                        diameter=loop_parameters[i*8+1],
                        position=loop_parameters[i*8+2:i*8+5],
                        orientation= R.from_euler("xyz", [loop_parameters[i*8+5], loop_parameters[i*8+6], loop_parameters[i*8+7]])
                    )
                )
            
            Bz = coll.getB(grid)[:, :, 2] * (42.58) * (10 ** 3)
            return Bz

        def objective_function(loop_parameters, desired_field):
            generated_field = self.map - calculate_magnetic_field(loop_parameters)
            return np.sum((desired_field - generated_field)**2)

        def std_objective_function(loop_parameters, desired_field):
            generated_field = calculate_magnetic_field(loop_parameters)
            return np.std(generated_field)

        # Define your desired magnetic field (e.g., an array of zeros)
        desired_field = np.zeros((256, 256))

        bounds = [(-2, 2), (5, 15), (-10, 10), (-10, 10), (-10, 10), (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi),
                  (-2, 2), (5, 15), (-10, 10), (-10, 10), (-10, 10), (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi),
                  (-2, 2), (5, 15), (-10, 10), (-10, 10), (-10, 10), (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi),
                  (-2, 2), (5, 15), (-10, 10), (-10, 10), (-10, 10), (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi),
                  (-2, 2), (5, 15), (-10, 10), (-10, 10), (-10, 10), (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi),
                  (-2, 2), (5, 15), (-10, 10), (-10, 10), (-10, 10), (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)]

        initial_guess = [0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0]
    
        # Run optimization
        for _ in range(100):
            initial_guess = [np.random.uniform(-2, 2), np.random.uniform(0, 15), np.random.uniform(-10, 10), np.random.uniform(-10, 10), np.random.uniform(-10, 10), 0, 0, 0,
                            np.random.uniform(-2, 2), np.random.uniform(0, 15), np.random.uniform(-10, 10), np.random.uniform(-10, 10), np.random.uniform(-10, 10), 0, 0, 0,
                            np.random.uniform(-2, 2), np.random.uniform(0, 15), np.random.uniform(-10, 10), np.random.uniform(-10, 10), np.random.uniform(-10, 10), 0, np.pi/2, 0,
                            np.random.uniform(-2, 2), np.random.uniform(0, 15), np.random.uniform(-10, 10), np.random.uniform(-10, 10), np.random.uniform(-10, 10), 0, np.pi/2, 0,
                            np.random.uniform(-2, 2), np.random.uniform(0, 15), np.random.uniform(-10, 10), np.random.uniform(-10, 10), np.random.uniform(-10, 10), np.pi/2, 0, 0,
                            np.random.uniform(-2, 2), np.random.uniform(0, 15), np.random.uniform(-10, 10), np.random.uniform(-10, 10), np.random.uniform(-10, 10), np.pi/2, 0, 0]
            result = minimize(objective_function, initial_guess, args=(desired_field,), bounds=bounds)
            optimal_parameters = result.x

            if np.mean(np.abs(self.map - calculate_magnetic_field(optimal_parameters))) < 50:
                print("Found one!")
                print(optimal_parameters)

    def load_coils(self, loop_parameters):
        Y, Z = np.mgrid[-5:5:256j, -5:5:256j].transpose((0, 2, 1))
        grid = np.stack([np.zeros((256, 256)), Y, Z], axis=2)

        coll = magpy.Collection()
        for i in range(6):
            coll.add(
                magpy.current.Loop(
                    current=loop_parameters[i*8],
                    diameter=loop_parameters[i*8+1],
                    position=loop_parameters[i*8+2:i*8+5],
                    orientation= R.from_euler("xyz", [loop_parameters[i*8+5], loop_parameters[i*8+6], loop_parameters[i*8+7]])
                )
            )
        
        magpy.show(coll)

        Bz = coll.getB(grid)[:, :, 2] * (42.58) * (10 ** 3)
        print("\nHere is the corrected field map:\n\n" + 
            str(self.map - Bz) +  
            "\n\n" + 
            f"Average deviation of corrected field_map from np.zeroes: {np.mean(np.abs(self.map - Bz))}" + 
            "\n")


op = optimizer("s17941/i2828748.MRDC.53")
#op.optimizer()

op.load_coils([  2,           7.37363851,  -8.29136235,   8.00457431,  -9.97946235,
  -0.18242018,   2.66384319,   2.93285288,  -0.67999685,  14.99905082,
   8.11134953 , -8.13514998,   9.99509999, -3.1398866,   -3.12098119,
   3.14064293 , -0.88864093,  14.46823269 , -9.72785901 , 9.94058661,
  -9.9884102 ,  -2.10945244  , 1.36347725 , 3.13839128 ,  1.94642447,
  14.97404708,  10     ,      9.8221754,   -9.55286118 , -1.93655334,
   1.9425756  ,  0.73671357 , -1.99621949 , 14.99568333 ,  9.44937214,
  -9.5546971  ,  9.88735903,  -1.92921249 ,  0.52145702,  -0.31674049,
  -0.45437399,  14.52875892 , -9.75398966 , -9.55851084 ,-10,
  -1.32680883 , -0.77049664 ,  3.05247276])

