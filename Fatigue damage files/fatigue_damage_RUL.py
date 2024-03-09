import numpy as np
import rainflow as rfc

# Material properties and SN curve parameters for fatigue analysis
material_props  = {'thk': 82.95, 'thk_ref': 25}  # Thickness and reference thickness in mm
sn_curve_params = {'K1': 1 / (10**12.164), 'beta1': 3, 'stress_lim': 52.639,
                   'K2': 1 / (10**15.606), 'beta2': 5
}

chunk_duration = 100    # Duration of each chunk in seconds

class RUL_class():
    
    def __init__(self, port="5556"):
        """Initialize the RUL class with default port and initial values for data storage and calculation.
        
        Parameters:
        - port: String representing the port number for communication.
        """
        self.port = port
        self.time_data = []
        self.axial_stress_data = []
        self.bending_moment_x_data = []
        self.bending_moment_y_data = []
        self.chunk_start_time = None
        self.fatigue_damage = 0  
        self.total_observation_time = 0 


    def update_measurements(self, current_time, measurements):
        """Update the measurement data with new values and process the chunk if the duration has been reached.
        
        Parameters:
        - current_time: The current timestamp of the measurement.
        - measurements: A dictionary containing the measurement values for axial stress and bending moments.
        """
        if self.chunk_start_time is None:
            self.chunk_start_time = current_time

        # Append the measurements
        self.time_data.append(current_time)
        self.axial_stress_data.append(measurements['rootMOOP(1)'])       # Temporary data applied to
        self.bending_moment_x_data.append(measurements['rootMOOP(2)'])   # make the code run. Needs to
        self.bending_moment_y_data.append(measurements['rootMOOP(3)'])   # be changed at later stage.
        
        # Debugging: Print or log the received measurements for verification
        # print(f"Received measurements at {current_time}: Axial Stress: {measurements['rootMOOP(1)']}, Bending Moment X: {measurements['rootMOOP(2)']}, Bending Moment Y: {measurements['rootMOOP(3)']}")

        # Check if the chunk_duration has been reached
        if current_time - self.chunk_start_time >= chunk_duration:
            self.process_chunk(current_time)
            self.reset_data(current_time)
         
            
    def process_chunk(self, current_time):
        """Process the current chunk of data, calculate stress, fatigue damage, and update RUL estimate.
        
        Parameters:
        - current_time: The current timestamp when the chunk processing is triggered.
        """
        # Convert lists to numpy arrays for processing
        # axial_stress = np.array(self.axial_stress_data)
        # bending_moment_x = np.array(self.bending_moment_x_data)
        bending_moment_y = np.array(self.bending_moment_y_data)
        
        # Initialize axial_stress and bending_moment_x arrays with zeros
        axial_stress = np.zeros_like(bending_moment_y)
        bending_moment_x = np.zeros_like(bending_moment_y)

        # Calculate stress and fatigue damage for the chunk
        stress_mpa = self.calculate_stress(axial_stress, bending_moment_x, bending_moment_y, material_props)
        fatigue_damage_chunk = self.calculate_fatigue_damage(stress_mpa, sn_curve_params, material_props)

        # Update the global fatigue_damage variable
        self.fatigue_damage += fatigue_damage_chunk
        
        # Update the total observation time
        if self.time_data:
            observed_chunk_duration = current_time - self.chunk_start_time
            self.total_observation_time += observed_chunk_duration

        RUL_estimate = self.estimate_RUL(self.fatigue_damage, self.total_observation_time)
        print(f"Fatigue Damage: {self.fatigue_damage:.2e}, RUL: {RUL_estimate:.6f} years")


    def reset_data(self, current_time):
        """Reset the measurement data storage for the next chunk of data and update the chunk start time.
        
        Parameters:
        - current_time: The current timestamp to set as the new start time for the next chunk.
        """
        self.time_data = []
        self.axial_stress_data = []
        self.bending_moment_x_data = []
        self.bending_moment_y_data = []
        self.chunk_start_time = current_time


    def calculate_stress(self, axial_force, bending_moment_x, bending_moment_y, material_props):
        """Calculate the total stress from axial and bending moments.

        Parameters:
        - axial_force: Axial force in kN.
        - bending_moment_x, bending_moment_y: Bending moments in kN-m.
        - material_props: Dictionary of material properties including thickness.

        Returns:
        - sigma_total_mpa: Total stress in megapascals (MPa).
        """
        # Material and geometric properties
        thk_mm = material_props['thk']
        thk_m = thk_mm / 1000                                    # Convert thickness to meters
        r_outer_m = 10 / 2                                       # Outer radius for the tower in meters
        r_inner_m = r_outer_m - thk_m
        A = np.pi * (r_outer_m**2 - r_inner_m**2)                # Cross-sectional area
        I_yy = I_xx = np.pi / 4 * (r_outer_m**4 - r_inner_m**4)  # Moment of inertia

        # Stress calculations
        sigma_axial = axial_force / A                            # Axial stress
        M_x = bending_moment_x                                   # Convert bending moment to Nm
        M_y = bending_moment_y 
        sigma_bending_x = M_x * r_outer_m / I_xx
        sigma_bending_y = M_y * r_outer_m / I_yy

        # Total stress
        sigma_total = sigma_axial + sigma_bending_x + sigma_bending_y
        sigma_total_mpa = sigma_total / 1e6                      # Convert to MPa
        return sigma_total_mpa


    def calculate_fatigue_damage(self, sigma_total_mpa, sn_curve_params, material_props):
        """Calculate fatigue damage for a given stress sequence using the SN curve parameters and material properties.
        
        Parameters:
        - sigma_total_mpa: Array of stress values in MPa.
        - sn_curve_params: Dictionary of SN curve parameters.
        - material_props: Dictionary of material properties including thickness and reference thickness.
        
        Returns:
        - fatigue_damage_chunk: Fatigue damage calculated for the given stress sequence.
        """
        fatigue_damage_chunk = 0
        K1, beta1, stress_lim, K2, beta2 = sn_curve_params.values()
        thk, thk_ref = material_props['thk'], material_props['thk_ref']

        # Rainflow counting
        cycles = list(rfc.count_cycles(sigma_total_mpa))
        for cycle in cycles:
            s, n = cycle[0], cycle[1]                             # Stress range and count
            beta, K = (beta1, K1) if s > stress_lim else (beta2, K2)
            Ns = (1 / K) * (s * (thk / thk_ref)**0.2)**(-beta)    # SN curve equation
            fatigue_damage_chunk += n / Ns

        return fatigue_damage_chunk


    def estimate_RUL(self, fatigue_damage, total_time_observed):
        """Estimate the Remaining Useful Life (RUL) based on cumulative fatigue damage and total time observed.
        
        Parameters:
        - fatigue_damage: Cumulative fatigue damage.
        - total_time_observed: Total time observed in seconds.
        
        Returns:
        - RUL_years: Estimated remaining useful life in years. If fatigue damage is not positive, returns an error message.
        """
        # print(f"TOTAL TIME OBSERVED: {total_time_observed}")
        if fatigue_damage <= 0:
            return "RUL cannot be estimated yet or fatigue damage calculation is incorrect."
        
        RUL_seconds = (1 - fatigue_damage) * total_time_observed / fatigue_damage
        RUL_years = RUL_seconds / (365.25 * 24 * 3600)            # Convert seconds to years
        return RUL_years



if __name__ == "__main__":
    RUL_instance = RUL_class()



        
