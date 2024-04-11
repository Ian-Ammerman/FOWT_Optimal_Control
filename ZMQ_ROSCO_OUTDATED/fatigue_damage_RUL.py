import numpy as np
import time as pytime
import rainflow as rfc
from rosco.toolbox.control_interface import wfc_zmq_server


# Material properties and SN curve parameters for fatigue analysis
material_props  = {'thk': 82.95, 'thk_ref': 25}  # Thickness and reference thickness in mm
sn_curve_params = {'K1': 1 / (10**12.164), 'beta1': 3, 'stress_lim': 52.639,
                   'K2': 1 / (10**15.606), 'beta2': 5
}


sim_speed = 100        # Simulation speed factor for real-time simulation
chunk_duration = 600    # Duration of each chunk in seconds

class RUL_class():
    
    def __init__(self, port="5556"):
        self.port = port
        self.time_data = []
        self.axial_stress_data = []
        self.bending_moment_x_data = []
        self.bending_moment_y_data = []
        self.chunk_start_time = None
        self.fatigue_damage = 0  

    def update_measurements(self, current_time, measurements):
        if self.chunk_start_time is None:
            self.chunk_start_time = current_time

        # Append the measurements
        self.time_data.append(current_time)
        self.axial_stress_data.append(measurements['rootMOOP(1)'])
        self.bending_moment_x_data.append(measurements['rootMOOP(2)'])
        self.bending_moment_y_data.append(measurements['rootMOOP(3)'])
        
        # Debugging: Print or log the received measurements for verification
        print(f"Received measurements at {current_time}: Axial Stress: {measurements['rootMOOP(1)']}, Bending Moment X: {measurements['rootMOOP(2)']}, Bending Moment Y: {measurements['rootMOOP(3)']}")

        # Check if the chunk_duration has been reached
        if current_time - self.chunk_start_time >= chunk_duration:
            self.process_chunk()
            self.reset_data()
            
    def process_chunk(self):
    # Convert lists to numpy arrays for processing
        axial_stress = np.array(self.axial_stress_data)
        bending_moment_x = np.array(self.bending_moment_x_data)
        bending_moment_y = np.array(self.bending_moment_y_data)

        # Calculate stress and fatigue damage for the chunk
        stress_mpa = self.calculate_stress(axial_stress, bending_moment_x, bending_moment_y, material_props)
        fatigue_damage_chunk = self.calculate_fatigue_damage(stress_mpa, sn_curve_params, material_props)

        # Update the global fatigue_damage variable
        self.fatigue_damage += fatigue_damage_chunk

        # Print or log the updated fatigue damage and any other relevant information

    def reset_data(self):
        self.time_data = []
        self.axial_stress_data = []
        self.bending_moment_x_data = []
        self.bending_moment_y_data = []
        self.chunk_start_time = None

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
        M_x = bending_moment_x * 1000                            # Convert bending moment to Nm
        M_y = bending_moment_y * 1000
        sigma_bending_x = M_x * r_outer_m / I_xx
        sigma_bending_y = M_y * r_outer_m / I_yy

        # Total stress
        sigma_total = sigma_axial + sigma_bending_x + sigma_bending_y
        sigma_total_mpa = sigma_total / 1e6                      # Convert to MPa
        return sigma_total_mpa

    def calculate_fatigue_damage(self, sigma_total_mpa, sn_curve_params, material_props):
        """Calculate fatigue damage for a given stress sequence using SN curve parameters.

        Parameters:
        - sigma_total_mpa: Array of stress values in MPa.
        - sn_curve_params: Dictionary of SN curve parameters.
        - material_props: Dictionary of material properties.

        Returns:
        - fatigue_damage_chunk: Fatigue damage for the given stress sequence.
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
        """Estimate the Remaining Useful Life (RUL) based on fatigue damage.

        Parameters:
        - fatigue_damage: Cumulative fatigue damage.
        - total_time_observed: Total time observed in seconds.

        Returns:
        - RUL_years: Estimated remaining useful life in years.
        """
        if fatigue_damage <= 0:
            return "RUL cannot be estimated yet or fatigue damage calculation is incorrect."
        
        RUL_seconds = (1 - fatigue_damage) * total_time_observed / fatigue_damage
        RUL_years = RUL_seconds / (365.25 * 24 * 3600)            # Convert seconds to years
        return RUL_years

    def simulate_real_time(self, file_path, material_props, sn_curve_params, sim_speed, chunk_duration):
        """Simulate real-time fatigue analysis using OpenFAST output data.

        Parameters:
        - file_path: Path to the OpenFAST output file.
        - material_props: Dictionary of material properties.
        - sn_curve_params: Dictionary of SN curve parameters.
        - simspeed: Factor to speed up the simulation.
        """
        time, bending_moment_x, bending_moment_y, axial_stress = self.update_measurements(self, current_time, measurements)
        real_time_increment = 0.0125  # Time increment per frame in the simulation
        total_frames = len(time)

        # Calculate chunk_size in terms of number of data points
        chunk_size = int(chunk_duration / real_time_increment)
        
        self.fatigue_damage = 0  # Reset global fatigue damage
        chunk_start_index = 0

        for frame_index in range(0, total_frames, chunk_size):
            # Ensure the chunk does not exceed the total number of frames
            chunk_end_index = min(frame_index + chunk_size, total_frames)
            
            # Process a chunk of data for stress calculation and fatigue analysis
            stress_mpa_chunk = self.calculate_stress(
                axial_stress[chunk_start_index:chunk_end_index],
                bending_moment_x[chunk_start_index:chunk_end_index],
                bending_moment_y[chunk_start_index:chunk_end_index],
                material_props
            )
            fatigue_damage_chunk = self.calculate_fatigue_damage(stress_mpa_chunk, sn_curve_params, material_props)
            fatigue_damage += fatigue_damage_chunk

            # Print RUL estimate
            current_time = chunk_end_index * real_time_increment
            RUL_estimate = self.estimate_RUL(fatigue_damage, current_time)
            print(f"Time Elapsed: {current_time/60:.0f}min - Fatigue Damage: {fatigue_damage:.2e}, RUL: {RUL_estimate:.2f} years")

            # Update chunk_start_index for the next chunk
            chunk_start_index = chunk_end_index

            # Speed up the simulation
            pytime.sleep(chunk_duration / sim_speed)
            
    def main(self):
        self.simulate_real_time(file_path, material_props, sn_curve_params, sim_speed, chunk_duration)


if __name__ == "__main__":
    RUL_instance = RUL_class()
    RUL_instance.main()


        
