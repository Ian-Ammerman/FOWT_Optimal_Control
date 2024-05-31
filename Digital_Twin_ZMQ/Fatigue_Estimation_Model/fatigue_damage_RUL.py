# fatigue_damage_RUL.py
import os
import time
import numpy as np
import pandas as pd
import rainflow as rfc

# Material properties and SN curve parameters for fatigue analysis
material_props_blade  = {'thk': 100, 'diameter': 5.2}  
#sn_curve_params_blade = {'S0': 412.98, 'b': 0.101}  # Normalized stress 'S0' and slope 'b'
sn_curve_params_blade = {'S0': 404.33, 'b': 0.133}   # Normalized stress 'S0' and slope 'b'

material_props_tower = {'thk': 82.95, 'thk_ref': 25, 'diameter': 10} 
sn_curve_params_tower = {'K1': 1/(10**12.164), 'beta1': 3, 
                         'stress_lim': 52.639, 'K2': 1/(10**15.606), 'beta2': 5}

class RUL_class():
    def __init__(self, port="5556", emit_callback=None, chunk_duration=50, nominal_design_life_years=20):
        """Initialize the RUL class with default port and initial values for data storage and calculation.
        
        Parameters:
        - port: String representing the port number for communication.
        """
        self.port = port
        print(f"RUL_class initialized with emit_callback: {emit_callback}")
        self.emit_callback = emit_callback
        self.chunk_duration = chunk_duration  # Set the chunk duration here
        self.nominal_design_life_years = nominal_design_life_years  # Set nominal design life
        
        self.time_data = []
        self.chunk_start_time = None
        self.total_observation_time = 0
        
        self.initial_rul_years = nominal_design_life_years
        self.fatigue_damage_blades_openfast = {'blade1': 0, 'blade2': 0, 'blade3': 0}
        self.fatigue_damage_tower_openfast = 0

        self.bending_moment_tower = {'TwrBsFzt': [], 'TwrBsMxt': [], 'TwrBsMyt': []}
        self.bending_moment_blades = {
            'blade1': {'RootFzb': [], 'RootMxb': [], 'RootMyb': []},
            'blade2': {'RootFzb': [], 'RootMxb': [], 'RootMyb': []},
            'blade3': {'RootFzb': [], 'RootMxb': [], 'RootMyb': []},
        }
        self.blade_pitch = {'BlPitchCMeas': []}
        
        self.csv_file_path = os.path.join("Digital_Twin_ZMQ", "Outputs", "rul_values.csv")
        print(f"CSV file path set to: {self.csv_file_path}")
        self.initialize_csv()
        
    def update_measurements(self, current_time, data, save_to_csv=False, csv_file_path=None):
        #print("CSV Measurements received:", csv_measurements)

        if self.chunk_start_time is None:
            self.chunk_start_time = current_time
            
        self.time_data.append(current_time)   
        self.blade_pitch['BlPitchCMeas'].append(data.get('BlPitchCMeas', 0))

        self.bending_moment_tower['TwrBsFzt'].append(data.get('TwrBsFzt', 0))
        self.bending_moment_tower['TwrBsMxt'].append(data.get('TwrBsMxt', 0))
        self.bending_moment_tower['TwrBsMyt'].append(data.get('TwrBsMyt', 0))

        for blade_num in range(1, 4):
            blade_key = f'blade{blade_num}'
            self.bending_moment_blades[blade_key]['RootFzb'].append(data.get(f'RootFzb{blade_num}', 0))
            self.bending_moment_blades[blade_key]['RootMxb'].append(data.get(f'RootMxb{blade_num}', 0))
            self.bending_moment_blades[blade_key]['RootMyb'].append(data.get(f'RootMyb{blade_num}', 0))
        
        if current_time - self.chunk_start_time >= self.chunk_duration:
            self.process_chunk(current_time, save_to_csv)

    def initialize_csv(self):
        """Initialize the CSV file with initial RUL and fatigue damage values."""
        data_to_save = {
            'Time': 0,
            'OpenFAST_RUL_blade1': self.initial_rul_years,
            'OpenFAST_RUL_blade2': self.initial_rul_years,
            'OpenFAST_RUL_blade3': self.initial_rul_years,
            'OpenFAST_RUL_Tower': self.initial_rul_years,
            'Fatigue_blade1': 0,
            'Fatigue_blade2': 0,
            'Fatigue_blade3': 0,
            'Fatigue_Tower': 0
        }
        df = pd.DataFrame([data_to_save])
        df.to_csv(self.csv_file_path, mode='w', header=True, index=False)
        print(f"Initialized new CSV with headers at: {self.csv_file_path}")
    
    def save_rul_to_csv(self, rul_values, fatigue_values, current_time):
        """
        Save the RUL and fatigue values to a CSV file.
        
        Parameters:
        - rul_values: Dictionary containing RUL values for blades and tower.
        - fatigue_values: Dictionary containing fatigue values for blades and tower.
        - current_time: Current time to include in the CSV file.
        """
        data_to_save = {'Time': current_time}
        data_to_save.update(rul_values)
        data_to_save.update(fatigue_values)

        df = pd.DataFrame([data_to_save])

        # Append new data to the existing file
        df.to_csv(self.csv_file_path, mode='a', header=False, index=False)
        print(f"Appended RUL and fatigue values to CSV at: {self.csv_file_path}")


    def process_chunk(self, current_time, save_to_csv=False):
        """Process the current chunk of data, calculate stress, fatigue damage, and update RUL estimate."""
        print(f"Processing chunk at time: {current_time}")
        blade_info_lines_openfast = []
        tower_info_lines_openfast = []
        fatigue_values_blade_openfast = {}
        fatigue_values_tower_openfast = {}
        rul_values_blade_openfast = {}
        rul_values_tower_openfast = {}
        
        if self.time_data:
            observed_chunk_duration = current_time - self.chunk_start_time
            self.total_observation_time += observed_chunk_duration
            print(f"Observed Chunk Duration: {observed_chunk_duration}, Total Observed Time blades: {self.total_observation_time}")

        # Calculate for each blade root using OpenFAST data
        for blade_num in range(1, 4):
            blade_key = f'blade{blade_num}'
            
            RootFzb = np.array(self.bending_moment_blades[blade_key]['RootFzb'])
            RootMxb = np.array(self.bending_moment_blades[blade_key]['RootMxb'])
            RootMyb = np.array(self.bending_moment_blades[blade_key]['RootMyb'])
            
            BlPitchCMeas = np.array(self.blade_pitch['BlPitchCMeas'])
            
            stress_mpa = self.calculate_stress_blades(RootFzb, RootMxb, RootMyb, BlPitchCMeas, material_props_blade)
            fatigue_damage_chunk = self.calculate_fatigue_damage_blades(stress_mpa, sn_curve_params_blade)
            self.fatigue_damage_blades_openfast[blade_key] += fatigue_damage_chunk
            
            RUL_blades_openfast = self.estimate_RUL(self.fatigue_damage_blades_openfast[blade_key])
            rul_values_blade_openfast[f'OpenFAST_RUL_{blade_key}'] = RUL_blades_openfast
            fatigue_values_blade_openfast[f'Fatigue_{blade_key}'] = self.fatigue_damage_blades_openfast[blade_key]
            blade_info_lines_openfast.append(f"OpenFAST - {blade_key}, Fatigue Damage: {self.fatigue_damage_blades_openfast[blade_key]:.2e}, RUL: {rul_values_blade_openfast[f'OpenFAST_RUL_{blade_key}']:.6f} years")
        
        # Calculate for tower base using OpenFAST data
        TwrBsFzt = np.array(self.bending_moment_tower['TwrBsFzt'])
        TwrBsMxt = np.array(self.bending_moment_tower['TwrBsMxt'])
        TwrBsMyt = np.array(self.bending_moment_tower['TwrBsMyt'])

        stress_mpa = self.calculate_stress_tower(TwrBsFzt, TwrBsMxt, TwrBsMyt, material_props_tower)
        fatigue_damage_chunk = self.calculate_fatigue_damage_tower(stress_mpa, sn_curve_params_tower, material_props_tower)
        self.fatigue_damage_tower_openfast += fatigue_damage_chunk
        
        RUL_tower_openfast = self.estimate_RUL(self.fatigue_damage_tower_openfast)
        rul_values_tower_openfast['OpenFAST_RUL_Tower'] = RUL_tower_openfast
        fatigue_values_tower_openfast['Fatigue_Tower'] = self.fatigue_damage_tower_openfast
        tower_info_lines_openfast.append(f"OpenFAST - Tower, Fatigue Damage: {self.fatigue_damage_tower_openfast:.2e}, RUL: {RUL_tower_openfast:.6f} years")
        
        # Printing formatted information
        print("\n".join(blade_info_lines_openfast))
        print("\n".join(tower_info_lines_openfast))

        # Reset data for the next chunk
        self.reset_data(current_time)
        
        if self.emit_callback:
            all_rul_values = {
                'rul_values_blade_openfast': rul_values_blade_openfast,
                'rul_values_tower_openfast': rul_values_tower_openfast
            }
            self.emit_callback(all_rul_values)
            
        if save_to_csv:
            print(f"Calling save_rul_to_csv at {current_time} with file path: {self.csv_file_path}")
            all_rul_values = {**rul_values_blade_openfast, **rul_values_tower_openfast}
            all_fatigue_values = {**fatigue_values_blade_openfast, **fatigue_values_tower_openfast}
            self.save_rul_to_csv(rul_values=all_rul_values, fatigue_values=all_fatigue_values, current_time=current_time)

    def reset_data(self, current_time):
        print(f"Data reset for the next chunk starting at: {current_time}")

        """Reset the measurement data storage for the next chunk of data and update the chunk start time.
        
        Parameters:
        - current_time: The current timestamp to set as the new start time for the next chunk.
        """
        self.chunk_start_time = current_time  # Update chunk start time
        self.time_data = []

        # Reset OpenFAST blade measurements using blade_key
        for blade_num in range(1, 4):  
            blade_key = f'blade{blade_num}'
            self.bending_moment_blades[blade_key] = {'RootFzb': [], 'RootMxb': [], 'RootMyb': []}

        # Reset OpenFAST tower measurements
        self.bending_moment_tower = {'TwrBsFzt': [], 'TwrBsMxt': [], 'TwrBsMyt': []}
        
        # Reset blade pitch measurements
        self.blade_pitch = {'BlPitchCMeas': []}
        
    def calculate_stress_blades(self, RootFzb, RootMxb_edgewise, RootMyb_flapwise, BlPitchCMeas, material_props_blade):
        """Calculate the total stress from axial and bending moments.

        Parameters:
        - axial_force: Axial force in kN.
        - bending_moment_x, bending_moment_y: Bending moments in kN-m.
        - material_props: Dictionary of material properties including thickness.

        Returns:
        - sigma_total_mpa: Total stress in megapascals (MPa).
        """
        #print(f"Calculating stress for blades with axial force: {axial_force}, moments x: {bending_moment_x}, moments y: {bending_moment_y}")
        #print(f"BlPitchCMeas data: {BlPitchCMeas}")
        # Material and geometric properties
        thk_mm, diameter_m = material_props_blade['thk'], material_props_blade['diameter']
        thk_m = thk_mm / 1000                                    # Convert thickness to meters
        r_outer_m = diameter_m / 2                               # Outer radius in meters
        r_inner_m = r_outer_m - thk_m
        theta   = np.radians(270)
        
        A = np.pi * (r_outer_m**2 - r_inner_m**2)                # Cross-sectional area
        I_yy = I_xx = np.pi / 4 * (r_outer_m**4 - r_inner_m**4)  # Moment of inertia

        # Stress calculations
        N_z = RootFzb / A               * 1000                   # Convert axial stress from kN to N  
        bending_x = RootMxb_edgewise    * 1000                   # Convert edgewise moment from kN to N                             
        bending_y = RootMyb_flapwise    * 1000                   # Convert flapwise moment from kN to N      
        M_x = bending_x * r_outer_m / I_xx * np.sin(theta + BlPitchCMeas)
        M_y = bending_y * r_outer_m / I_yy * np.cos(theta + BlPitchCMeas)

        # Total stress
        sigma_total = N_z + M_x + M_y
        sigma_total_mpa = sigma_total / 1e6                      # Convert to MPa
        return sigma_total_mpa

    def calculate_fatigue_damage_blades(self, sigma_total_mpa, sn_curve_params_blade):
        """Calculate fatigue damage for a given stress sequence using the SN curve parameters and material properties.
        
        Parameters:
        - sigma_total_mpa: Array of stress values in MPa.
        - sn_curve_params: Dictionary of SN curve parameters.
        - material_props: Dictionary of material properties including thickness and reference thickness.
        
        Returns:
        - fatigue_damage_chunk: Fatigue damage calculated for the given stress sequence.
        """
        #print(f"Calculating fatigue damage for blades with total stress: {sigma_total_mpa}")
        
        FD = 0
        S0, b = sn_curve_params_blade.values()

        # Rainflow counting
        cycles = rfc.count_cycles(sigma_total_mpa)
        for cycle in cycles:
            S = cycle[0]  # Stress range
            n = cycle[1]  # Count
            N = 10**((1 - S/S0) / b) 
            FD += n / N

        return FD

    def calculate_stress_tower(self, axial_force, moment_sideside_x, moment_foreaft_y, material_props_tower):
        """Calculate the total stress from axial and bending moments.

        Parameters:
        - axial_force: Axial force in kN.
        - bending_moment_x, bending_moment_y: Bending moments in kN-m.
        - material_props: Dictionary of material properties including thickness.

        Returns:
        - sigma_total_mpa: Total stress in megapascals (MPa).
        """
        #print(f"Calculating stress for tower with axial force: {axial_force}, moments x: {moment_sideside_x}, moments y: {moment_foreaft_y}")

        # Material and geometric properties
        thk_mm, diameter = material_props_tower['thk'], material_props_tower['diameter']
        
        thk_m = thk_mm / 1000                                    # Convert thickness to meters
        r_outer = diameter / 2                                   # Outer radius in meters
        r_inner = r_outer - thk_m
        theta   = np.radians(90)
        A = np.pi * (r_outer**2 - r_inner**2)                    # Cross-sectional area
        I_yy = I_xx = np.pi / 4 * (r_outer**4 - r_inner**4)      # Moment of inertia
        
        # Stress calculations
        sigma_axial = axial_force / A  *1000                     # Convert axial force from kN to N 
        M_x = moment_sideside_x        *1000                     # Convert side-side bending moment from kNm to Nm 
        M_y = moment_foreaft_y         *1000                     # Convert fore-aft bending moment from kNm to Nm
        sigma_bending_x = M_x / I_xx * r_outer * np.cos(theta)
        sigma_bending_y = M_y / I_yy * r_outer * np.sin(theta)

        # Total stress
        sigma_total = sigma_axial + sigma_bending_x + sigma_bending_y
        sigma_total_mpa = sigma_total / 1e6                      # Convert to MPa
        return sigma_total_mpa

    def calculate_fatigue_damage_tower(self, sigma_total_mpa, sn_curve_params_tower, material_props_tower):
        """Calculate fatigue damage for a given stress sequence using the SN curve parameters and material properties.
        
        Parameters:
        - sigma_total_mpa: Array of stress values in MPa.
        - sn_curve_params: Dictionary of SN curve parameters.
        - material_props: Dictionary of material properties including thickness and reference thickness.
        
        Returns:
        - fatigue_damage_chunk: Fatigue damage calculated for the given stress sequence.
        """
        #print(f"Calculating fatigue damage for tower with total stress: {sigma_total_mpa}")

        FD = 0
        K1, beta1, stress_lim, K2, beta2 = sn_curve_params_tower.values()
        thk, thk_ref = material_props_tower['thk'], material_props_tower['thk_ref']
        
        cc = rfc.count_cycles(sigma_total_mpa)
        sk = [c[0] for c in cc] # stress range
        n = [c[1] for c in cc] # cycle count
        
        Ns = np.zeros(len(sk)) #initialize damage
        
        for i,s in enumerate(sk):
            if s>stress_lim:
                beta = beta1; K = K1
            else:
                beta = beta2; K = K2
            
            Ns[i] = 1/K*(s*(thk/thk_ref)**0.2)**(-beta)
            
        FD = np.sum(n/Ns)
        
        return FD
        
    def estimate_RUL(self, fatigue_damage):
        """Estimate the Remaining Useful Life (RUL) based on cumulative fatigue damage and nominal design life.
        
        Parameters:
        - fatigue_damage: Cumulative fatigue damage.
        - nominal_design_life_years: Nominal design life in years.
        
        Returns:
        - RUL_years: Estimated remaining useful life in years. If fatigue damage is not positive, returns an error message.
        """
        # print(f"TOTAL TIME OBSERVED: {total_time_observed}")
        if fatigue_damage <= 0: 
            return "RUL cannot be estimated yet or fatigue damage calculation is incorrect."
        
        RUL_years = (1 - fatigue_damage) * self.nominal_design_life_years
        return RUL_years

if __name__ == "__main__":
    RUL_instance = RUL_class()
