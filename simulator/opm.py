"""Wrap OPM-flow"""
# External imports
from subprocess import call, DEVNULL, run
import os
import shutil

# Internal imports
from simulator.eclipse import eclipse
from misc.system_tools.environ_var import OPMRunEnvironment


class flow(eclipse):
    """
    Class for running OPM flow with Eclipse input files. Inherits eclipse parent class for setting up and running
    simulations, and reading the results.
    """

    def __init__(self,input_file=None,initialize_parent=True):
        if initialize_parent:
            super().__init__(input_file)
        else:
            self.file = input_file['filename']
            self.options = None

    def call_sim(self, folder=None, wait_for_proc=False):
        """
        Call OPM flow simulator via shell.

        Parameters
        ----------
        folder : str
            Folder with runfiles.

        wait_for_proc : bool
            Boolean determining if we wait for the process to be done or not.

        Changelog
        ---------
        - ST 18/10-18
        """
        # Filename
        if folder is not None:
            filename = folder + self.file
        else:
            filename = self.file

        success = True
        try:
            with OPMRunEnvironment(filename, 'PRT', ['End of simulation', 'NOSIM']):
                com = []
                if self.options['mpi']:
                    com.extend(self.options['mpi'].split())
                com.append(self.options['sim_path'] + 'flow')
                if self.options['parsing-strictness']:
                    com.extend(['--parsing-strictness=' + self.options['parsing-strictness']])
                com.extend(['--output-dir=' + folder, *
                           self.options['sim_flag'].split(), filename + '.DATA'])
                if 'sim_limit' in self.options:
                    call(com, stdout=DEVNULL, timeout=self.options['sim_limit'])
                else:
                    call(com, stdout=DEVNULL)
                raise ValueError  # catch errors in run_sim
        except:
            print('\nError in the OPM run.')  # add rerun?
            if not os.path.exists('Crashdump'):
                shutil.copytree(folder, 'Crashdump')
            success = False

        return success

    def check_sim_end(self, finished_member=None):
        """
        Check in RPT file for "End of simulation" to see if OPM flow is done.

        Changelog
        ---------
        - ST 19/10-18
        """
        # Initialize output
        # member = None
        #
        # # Search for output.dat file
        # for file in os.listdir('En_' + str(finished_member)):  # Search within a specific En_folder
        #     if file.endswith('PRT'):  # look in PRT file
        #         with open('En_' + str(finished_member) + os.sep + file, 'r') as fid:
        #             for line in fid:
        #                 if re.search('End of simulation', line):
        #                     # TODO: not do time.sleep()
        #                     # time.sleep(0.1)
        #                     member = finished_member

        return finished_member

    @staticmethod
    def SLURM_HPC_run(num_runs, filename=None):
        """
        HPC run manager for SLURM.

        This function will start num_runs of sim.call_sim() using job arrays in SLURM.
        """
        filename_str = f'"{filename}"' if filename is not None else ""

        slurm_script = f"""\                                                                                  
    #!/bin/bash                                                                                               
    #SBATCH --partition=comp                                                                                  
    #SBATCH --job-name=test_mpi                                                                               
    #SBATCH --array=0-{num_runs - 1}                                                                            
    #SBATCH --time=01:00:00                                                                                   
    #SBATCH --mem=4G                                                                                          
    #SBATCH --cpus-per-task=1                                                                                 
    #SBATCH --export=ALL                                                                                      
    #SBATCH --output=/dev/null                                                                                
    #SBATCH --error=errors.log                                                                                

    # OPTIONAL: load modules here                                                                             
    module load Python                                                                                        
    export LMOD_DISABLE_SAME_NAME_AUTOSWAP=no                                                                 
    module load opm-simulators                                                                                

    source ../../../code/venv/bin/activate                                                                    

    # Set folder based on SLURM_ARRAY_TASK_ID
    folder="En_${{SLURM_ARRAY_TASK_ID}}"
                                          
    python simulator/opm.py "$folder" {filename_str}                                              
    """
        script_name = "submit_test_parallel_mpi.sh"
        with open(script_name, "w") as f:
            f.write(slurm_script)

        # Make it executable (optional):
        os.chmod(script_name, 0o755)

        #    print(f"Created SLURM script: {script_name}")
        #    print(f"Submitting array job with {num_runs} tasks...")

        # Submit the script to SLURM
        cmd = ["sbatch", script_name]
        run(cmd, capture_output=True, text=True)

        # remove the sh file
        os.remove(script_name)


class ebos(eclipse):
    """
    Class for running OPM ebos with Eclipse input files. Inherits eclipse parent class for setting up and running
    simulations, and reading the results.
    """

    def call_sim(self, folder=None, wait_for_proc=False):
        """
        Call OPM flow simulator via shell.

        Parameters
        ----------
        folder : str
            Folder with runfiles.

        wait_for_proc : bool
            Determines whether to wait for the process to be done or not.

        Changelog
        ---------
        - RJL 27/08-19
        """
        # Filename
        if folder is not None:
            filename = folder + self.file
        else:
            filename = self.file

        # Run simulator
        if 'sim_path' not in self.options.keys():
            self.options['sim_path'] = ''

        with OPMRunEnvironment(filename, 'OUT', 'Timing receipt'):
            with open(filename+'.OUT', 'w') as f:
                call([self.options['sim_path'] + 'ebos', '--output-dir=' + folder,
                     *self.options['sim_flag'].split(), filename + '.DATA'], stdout=f)

    def check_sim_end(self, finished_member=None):
        """
        Check in RPT file for "End of simulation" to see if OPM ebos is done.

        Changelog
        ---------
        - RJL 27/08-19
        """
        # Initialize output
        # member = None
        #
        # # Search for output.dat file
        # for file in os.listdir('En_' + str(finished_member)):  # Search within a specific En_folder
        #     if file.endswith('OUT'):  # look in OUT file
        #         with open('En_' + str(finished_member) + os.sep + file, 'r') as fid:
        #             for line in fid:
        #                 if re.search('Timing receipt', line):
        #                     # TODO: not do time.sleep()
        #                     # time.sleep(0.1)
        #                     member = finished_member

        return finished_member


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python opm.py <folder> <filename>")
        sys.exit(1)

    folder = sys.argv[1]
    filename = sys.argv[2]
    sim = flow(input_file={'filename': filename},initialize_parent=False)
    success = sim.call_sim(folder=folder)
    #print("Success!" if success else "Failed.")
