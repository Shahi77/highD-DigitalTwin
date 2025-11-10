"""
Utility functions for SUMO simulation
"""
import os
import sys
import subprocess

import traci


def check_sumo_env():
    """Check if SUMO environment is properly set up"""
    if 'SUMO_HOME' not in os.environ:
        sys.exit("Please declare environment variable 'SUMO_HOME'")
    
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if tools not in sys.path:
        sys.path.append(tools)
    
    print(f" SUMO_HOME: {os.environ['SUMO_HOME']}")


def start_sumo(config_file, nogui=False, gui=False):
    """
    Start SUMO simulation
    
    Args:
        config_file: Path to .sumocfg file
        nogui: Deprecated, use gui parameter instead
        gui: If True, use sumo-gui, otherwise use sumo
    """
    # Determine which binary to use
    use_gui = gui if not nogui else False
    
    sumo_binary = "sumo-gui" if use_gui else "sumo"
    
    # Build command
    sumo_cmd = [sumo_binary, "-c", config_file]
    
    print(f"Starting SUMO: {' '.join(sumo_cmd)}")
    
    # Start SUMO
    max_retries = 3
    for attempt in range(max_retries):
        try:
            traci.start(sumo_cmd)
            print(" SUMO started successfully")
            return
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Retry attempt {attempt + 1}/{max_retries}")
                import time
                time.sleep(1)
            else:
                print(f" Failed to start SUMO after {max_retries} attempts")
                raise e


def running(demo_mode=True, current_step=0, max_step=None):
    """
    Check if simulation should continue running
    
    Args:
        demo_mode: If True, run until max_step. If False, run while vehicles exist
        current_step: Current simulation step
        max_step: Maximum number of steps to run (only used if demo_mode=True)
    
    Returns:
        bool: True if simulation should continue, False otherwise
    """
    try:
        # Check if TraCI is connected
        if not traci.isLoaded():
            return False
        
        # In demo mode, run until max_step
        if demo_mode:
            if max_step is not None and current_step >= max_step:
                return False
            return True
        
        # In non-demo mode, run while vehicles exist
        min_expected = traci.simulation.getMinExpectedNumber()
        return min_expected > 0
        
    except traci.exceptions.FatalTraCIError:
        return False
    except Exception as e:
        print(f"Error in running check: {e}")
        return False