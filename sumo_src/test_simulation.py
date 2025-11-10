"""
Test simulation script - Works without highD dataset
This will add random vehicles to test your SUMO setup
"""
import random
import sys
import os
from utils import check_sumo_env, start_sumo, running

check_sumo_env()
import traci

def test_simulation():
    print(" Starting test simulation...")
    
    # Start SUMO
    start_sumo("sumo_cfg/freeway.sumo.cfg", False, gui=True)
    
    step = 0
    vehicles_added = 0
    max_vehicles = 50  # Add 50 test vehicles
    
    print(f"Will add {max_vehicles} test vehicles...")
    
    while step < 5000:  # Run for 5000 steps (~20 minutes sim time)
        traci.simulationStep()
        
        # Add a vehicle every 20 steps (5 seconds)
        if step % 20 == 0 and vehicles_added < max_vehicles:
            vehicle_id = f"test_{vehicles_added}"
            vehicle_type = f"car{random.randint(0, 999)}"
            lane = random.randint(0, 2)  # Random lane (0, 1, or 2)
            speed = random.uniform(31, 35)  # Random speed 31-35 m/s
            
            try:
                traci.vehicle.add(
                    vehID=vehicle_id,
                    routeID="platoon_route",
                    typeID=vehicle_type,
                    departSpeed=8,      # spawn slow
                    departPos=10,
                    departLane=lane,
                )
                traci.vehicle.setMaxSpeed(vehicle_id, 10)     # clamp max to 10 m/s
                traci.vehicle.setSpeed(vehicle_id, 8)         # hold initial lower speed
                vehicles_added += 1
                print(f"Added vehicle {vehicle_id} on lane {lane}")
            except Exception as e:
                print(f" Failed to add vehicle {vehicle_id}: {e}")
        
        # Print status every 100 steps
        if step % 100 == 0:
            try:
                vehicle_list = traci.edge.getLastStepVehicleIDs("E0")
                print(f"Step {step}: {len(vehicle_list)} vehicles on highway, {vehicles_added} total added")
            except:
                pass
        
        step += 1
    
    print(f"\n Test complete! Added {vehicles_added} vehicles")
    traci.close()

if __name__ == "__main__":
    try:
        test_simulation()
    except KeyboardInterrupt:
        print("\n Simulation stopped by user")
        traci.close()
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
        try:
            traci.close()
        except:
            pass