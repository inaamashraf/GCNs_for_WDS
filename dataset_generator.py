# -*- coding: utf-8 -*-
"""
Dataset Generator
Copyright: (C) 2019, KIOS Research Center of Excellence
"""
import pandas as pd
import numpy as np
import wntr
import pickle
import os
import sys
import yaml
import shutil
import time


class LeakDatasetCreator:
    def __init__(self, scenario_folder, inp_file):        

        # Read input arguments from yalm file
        self.scenario_folder = scenario_folder
        try:
            with open(os.path.join(os.getcwd(), scenario_folder, "dataset_configuration.yaml"), 'r') as f:
                configs = yaml.safe_load(f.read())
        except Exception as ex:
            print('"dataset_configuration" file not found.')
            print(ex)
            sys.exit()

        print(f'Run input file: "{inp_file}"')

        start_time = configs['times']['StartTime']
        end_time = configs['times']['EndTime']
        self.pressure_sensors = configs['pressure_sensors']

        def get_sensors(configs, field):
            sensors = []
            [sensors.append(str(sens)) for sens in configs[field]]
            return sensors

        self.flow_sensors = get_sensors(configs, 'flow_sensors')
        self.pressure_sensors = get_sensors(configs, 'pressure_sensors')
        self.amrs = get_sensors(configs, 'amrs')
        self.flow_sensors = get_sensors(configs, 'flow_sensors')
        self.level_sensors = get_sensors(configs, 'level_sensors')

        # demand-driven (DD) or pressure dependent demand (PDD)
        Mode_Simulation = 'PDD'  # 'PDD'#'PDD'       

        self.scenario_num = 1
        self.unc_range = np.arange(0, 0.25, 0.05)

        # Load EPANET network file
        self.wn = wntr.network.WaterNetworkModel(inp_file)
        self.wn.options.hydraulic.demand_model = Mode_Simulation

        self.nodes = self.wn.to_graph().nodes()
        self.links = self.wn.link_name_list

        for name, node in self.wn.junctions():
            node.required_pressure = 25  

        # Get time step
        self.time_step = round(self.wn.options.time.hydraulic_timestep)
        # Create time_stamp
        self.time_stamp = pd.date_range(start_time, end_time, freq=str(self.time_step / 60) + "min")

        # Simulation duration in steps
        self.wn.options.time.duration = (len(self.time_stamp) - 1) * 300  # 5min step
        self.TIMESTEPS = int(self.wn.options.time.duration / self.wn.options.time.hydraulic_timestep)


    def create_folder(self, _path_):

        try:
            if os.path.exists(_path_):
                shutil.rmtree(_path_)
            os.makedirs(_path_)
        except Exception as error:
            pass


    def dataset_generator(self):
        # Path of EPANET Input File
        print(f"Dataset Generator run...")        
        
        # Save the water network model to a file before using it in a simulation
        with open(os.path.join(os.getcwd(), self.scenario_folder,'self.wn.pickle'), 'wb') as f:
            pickle.dump(self.wn, f)

        # Run wntr simulator
        sim = wntr.sim.WNTRSimulator(self.wn)
        results = sim.run_sim()
        if results.node["pressure"].empty:
            print("Negative pressures.")
            return -1

        if results:
            decimal_size = 2

            # Create xlsx file with Measurements
            def export_measurements(pressure_sensors, file_out="Measurements_All_Pressures.csv"):
                total_pressures = {'Timestamp': self.time_stamp}

                for j in range(0, self.wn.num_nodes):
                    node_id = self.wn.node_name_list[j]
                    if node_id in pressure_sensors:
                        pres = results.node['pressure'][node_id]
                        pres = pres[:len(self.time_stamp)]
                        pres = [round(elem, decimal_size) for elem in pres]
                        total_pressures[node_id] = pres

                # Create a Pandas dataframe from the data.
                df_pressures = pd.DataFrame(total_pressures)
                # Export as .csv file -- .csv files are much faster parsed by pandas than huge .xlsx files!
                df_pressures.to_csv(os.path.join(self.scenario_folder, file_out), index=False)

            # Export all measurements 
            export_measurements(self.nodes, "Measurements_All_Pressures.csv")

            # Clean up
            os.remove(os.path.join(os.getcwd(), self.scenario_folder,'self.wn.pickle'))
        else:
            print('Results empty.')
            return -1


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: < demand_type: Toy or Real >")
    else:

        scenario_dir = os.path.join(os.getcwd(), "networks", "L-Town", sys.argv[1])
        if not os.path.isdir(scenario_dir):
            os.system('mkdir ' + os.path.join(os.getcwd(), "networks"))
            os.system('mkdir ' + os.path.join(os.getcwd(), "networks", "L-Town"))
            os.system('mkdir ' + os.path.join(os.getcwd(), "networks", "L-Town", sys.argv[1]))
        
        if sys.argv[1] == 'Toy':
            inp_file = os.path.join(scenario_dir, "L-TOWN.inp")
        elif sys.argv[1] == 'Real':
            inp_file = os.path.join(scenario_dir, "L-TOWN_Real.inp")

        t = time.time()

        # Call dataset creator        
        L = LeakDatasetCreator(scenario_dir, inp_file)
        L.dataset_generator()

        print(f'Total Elapsed time is {str(time.time() - t)} seconds.')
