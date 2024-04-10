# fatigue_damage_RUL.py
import numpy as np
import rainflow as rfc
from real_time_server import RealTimeServer_class
from real_time_server import socketio, app

# # Material properties and SN curve parameters for fatigue analysis
# material_props  = {'thk': 82.95, 'thk_ref': 25}  # Thickness and reference thickness in mm
# sn_curve_params = {'K1': 1 / (10**12.164), 'beta1': 3, 'stress_lim': 52.639,
#                    'K2': 1 / (10**15.606), 'beta2': 5
# }

# Material properties and SN curve parameters for fatigue analysis
material_props  = {'thk': 100, 'diameter': 5.2}  # Thickness in mm and diamater of the blade-root in m
sn_curve_params = {'S0': 605, 'b': 0.14}  # Normalized stress 'S0' and slope 'b' for DD6

chunk_duration = 100    # Duration of each chunk in seconds

class RUL_class():
    def __init__(self, port="5556", emit_callback=None):
        """Initialize the RUL class with default port and initial values for data storage and calculation.
        
        Parameters:
        - port: String representing the port number for communication.
        """
        self.port = port
        print(f"RUL_class initialized with emit_callback: {emit_callback}")
        self.emit_callback = emit_callback
        self.time_data = []
        self.bending_moment_blades = {'blade1': [], 'blade2': [], 'blade3': []}
        self.chunk_start_time = None
        self.fatigue_damage_blades = {'blade1': 0, 'blade2': 0, 'blade3': 0}
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
        self.bending_moment_blades['blade1'].append(measurements['rootMOOP(1)'])  # For blade 1
        self.bending_moment_blades['blade2'].append(measurements['rootMOOP(2)'])  # For blade 2
        self.bending_moment_blades['blade3'].append(measurements['rootMOOP(3)'])  # For blade 3

        # Debugging: Print or log the received measurements for verification
        # print(f"Received measurements at {current_time}: Axial Stress: {measurements['rootMOOP(1)']}, Bending Moment X: {measurements['rootMOOP(2)']}, Bending Moment Y: {measurements['rootMOOP(3)']}")

        # Check if the chunk_duration has been reached
        if current_time - self.chunk_start_time >= chunk_duration:
            self.process_chunk(current_time)
            self.reset_data(current_time)
         
            
    def process_chunk(self, current_time):
        """Process the current chunk of data, calculate stress, fatigue damage, and update RUL estimate."""
        
        blade_info_lines = []  # List to collect each blade's information
        rul_values = {}  # Dictionary to store RUL values for emitting

        # Calculate for each blade separately...
        for blade, moments in self.bending_moment_blades.items():
            bending_moment_x = np.array(moments)
            axial_stress = np.zeros_like(bending_moment_x)      # Assuming axial stress is zero
            bending_moment_y = np.zeros_like(bending_moment_x)  # Assuming bending moment y is zero
            
            stress_mpa = self.calculate_stress(axial_stress, bending_moment_x, bending_moment_y, material_props)
            fatigue_damage_chunk = self.calculate_fatigue_damage(stress_mpa, sn_curve_params, material_props)

            self.fatigue_damage_blades[blade] += fatigue_damage_chunk

            # Update observation time for the chunk
            if self.time_data:
                observed_chunk_duration = current_time - self.chunk_start_time
                self.total_observation_time += observed_chunk_duration

            # Estimate the RUL once and store it
            RUL_estimate = self.estimate_RUL(self.fatigue_damage_blades[blade], self.total_observation_time)
            rul_values[blade] = RUL_estimate  # Store the value for socket emission

            # Format the blade information for terminal output
            blade_number = blade.replace('blade', '')  # Extract blade number
            blade_info_lines.append(f"Blade {blade_number}, Fatigue Damage: {self.fatigue_damage_blades[blade]:.2e}, RUL: {RUL_estimate:.6f} years")

        # Print all blade information with proper formatting
        if blade_info_lines:
            print("\n" + "\n".join(blade_info_lines) + "\n")

        # Reset data for the next chunk
        self.reset_data(current_time)
        
        if self.emit_callback:
            #print(f"Invoking emit_callback with rul_values: {rul_values}")
            self.emit_callback(rul_values)

    def reset_data(self, current_time):
        """Reset the measurement data storage for the next chunk of data and update the chunk start time.
        
        Parameters:
        - current_time: The current timestamp to set as the new start time for the next chunk.
        """
        self.time_data = []
        for blade in self.bending_moment_blades.keys():
            self.bending_moment_blades[blade] = []
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
        thk_mm, diameter_m = material_props['thk'], material_props['diameter']
        thk_m = thk_mm / 1000                                    # Convert thickness to meters
        r_outer_m = diameter_m / 2                               # Outer radius in meters
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
    #    fatigue_damage_chunk = 0
    #    K1, beta1, stress_lim, K2, beta2 = sn_curve_params.values()
    #    thk, thk_ref = material_props['thk'], material_props['thk_ref']

    #    # Rainflow counting
    #    cycles = list(rfc.count_cycles(sigma_total_mpa))
    #    for cycle in cycles:
    #        s, n = cycle[0], cycle[1]                             # Stress range and count
    #        beta, K = (beta1, K1) if s > stress_lim else (beta2, K2)
    #        Ns = (1 / K) * (s * (thk / thk_ref)**0.2)**(-beta)    # SN curve equation
    #        fatigue_damage_chunk += n / Ns

    #    return fatigue_damage_chunk
        
        fatigue_damage_chunk = 0
        S0, b = sn_curve_params.values()

        # Rainflow counting
        cycles = list(rfc.count_cycles(sigma_total_mpa))
        for cycle in cycles:
            S = cycle[0]  # Stress range
            n = cycle[1]  # Count
            N = 10**((1 - S/S0) / b) 
            fatigue_damage_chunk += n / N

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
    real_time_server_instance = RealTimeServer_class()
