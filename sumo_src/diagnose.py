#!/usr/bin/env python3
"""
Diagnostic script to check your SUMO setup
"""

import os
import sys
import csv

def check_file(path, description, min_size=None):
    """Check if a file exists and optionally check its size"""
    if os.path.exists(path):
        size = os.path.getsize(path)
        size_str = f"{size:,} bytes"
        if min_size and size < min_size:
            print(f"  {description}: {path} (too small: {size_str})")
            return False
        else:
            print(f"{description}: {path} ({size_str})")
            return True
    else:
        print(f" {description}: {path} NOT FOUND")
        return False


def check_highd_data():
    """Check highD dataset files"""
    print("\n Checking highD Dataset:")
    print("-" * 60)
    
    meta_path = "./data/highd/dataset/02_tracksMeta.csv"
    tracks_path = "./data/highd/dataset/02_tracks.csv"
    
    meta_ok = check_file(meta_path, "TracksMeta", min_size=1000)
    tracks_ok = check_file(tracks_path, "Tracks", min_size=10000)
    
    if meta_ok and tracks_ok:
        # Count vehicles
        try:
            with open(meta_path, 'r') as f:
                reader = csv.DictReader(f)
                direction_1_count = sum(1 for row in reader if row['drivingDirection'] == '1')
            
            with open(tracks_path, 'r') as f:
                reader = csv.DictReader(f)
                unique_ids = set(row['id'] for row in reader)
            
            print(f"   Vehicles (direction 1): {direction_1_count}")
            print(f"   Total unique vehicle IDs: {len(unique_ids)}")
        except Exception as e:
            print(f"  Error reading data: {e}")
    
    return meta_ok and tracks_ok


def check_sumo_files():
    """Check SUMO configuration files"""
    print("\n Checking SUMO Files:")
    print("-" * 60)
    
    files_ok = True
    
    # Check main directory
    files_ok &= check_file("sumo_cfg/net.xml", "Network", min_size=500)
    files_ok &= check_file("sumo_cfg/route.xml", "Routes", min_size=100)
    files_ok &= check_file("sumo_cfg/freeway.sumo.cfg", "Config", min_size=100)
    files_ok &= check_file("sumo_cfg/vTypeDistributions.add.xml", "Vehicle Types", min_size=100000)
    
    # Check vehicle configs
    check_file("veh_config/car.config.txt", "Car Config", min_size=100)
    check_file("veh_config/truck.config.txt", "Truck Config", min_size=100)
    
    return files_ok


def check_network_content():
    """Check network file content"""
    print("\n Checking Network Content:")
    print("-" * 60)
    
    try:
        with open("sumo_cfg/net.xml", 'r') as f:
            content = f.read()
            
            # Check for edge E0
            if 'id="E0"' in content:
                print(" Edge E0 found")
                
                # Extract length
                import re
                length_match = re.search(r'edge id="E0".*?length="([\d.]+)"', content)
                if length_match:
                    length = float(length_match.group(1))
                    print(f"   Length: {length}m")
                    if length < 1000:
                        print(f"    Warning: Edge is quite short ({length}m)")
                
                # Count lanes
                lane_count = content.count('lane id="E0_')
                print(f"   Lanes: {lane_count}")
            else:
                print(" Edge E0 NOT FOUND in network")
                return False
        
        return True
    except Exception as e:
        print(f" Error reading network: {e}")
        return False


def check_route_content():
    """Check route file content"""
    print("\n  Checking Route Content:")
    print("-" * 60)
    
    try:
        with open("sumo_cfg/route.xml", 'r') as f:
            content = f.read()
            
            if 'id="platoon_route"' in content:
                print(" Route 'platoon_route' found")
                
                if 'edges="E0"' in content:
                    print("  References edge E0")
                else:
                    print(" Does not reference E0")
            else:
                print(" Route 'platoon_route' NOT FOUND")
                return False
        
        return True
    except Exception as e:
        print(f" Error reading routes: {e}")
        return False


def check_vehicle_types():
    """Check vehicle type distribution"""
    print("\n Checking Vehicle Types:")
    print("-" * 60)
    
    try:
        with open("sumo_cfg/vTypeDistributions.add.xml", 'r') as f:
            content = f.read()
            
            car_count = content.count('id="car')
            truck_count = content.count('id="truck')
            
            print(f" Car types: {car_count}")
            print(f" Truck types: {truck_count}")
            
            if car_count < 100:
                print(f"     Warning: Very few car types ({car_count})")
            if truck_count < 100:
                print(f"     Warning: Very few truck types ({truck_count})")
            
            return car_count > 0 and truck_count > 0
    except Exception as e:
        print(f" Error reading vehicle types: {e}")
        return False


def check_environment():
    """Check environment variables"""
    print("\n Checking Environment:")
    print("-" * 60)
    
    if 'SUMO_HOME' in os.environ:
        sumo_home = os.environ['SUMO_HOME']
        print(f" SUMO_HOME: {sumo_home}")
        
        # Check if SUMO binaries exist
        sumo_bin = os.path.join(sumo_home, '../../bin/sumo')
        sumo_gui_bin = os.path.join(sumo_home, '../../bin/sumo-gui')
        
        if os.path.exists(os.path.normpath(sumo_bin)):
            print("    sumo binary found")
        if os.path.exists(os.path.normpath(sumo_gui_bin)):
            print("    sumo-gui binary found")
        
        return True
    else:
        print(" SUMO_HOME not set")
        print("   Set it with: export SUMO_HOME=/opt/homebrew/opt/sumo/share/sumo")
        return False


def main():
    """Run all diagnostics"""
    print("=" * 60)
    print("  SUMO highD Simulation Diagnostics")
    print("=" * 60)
    
    checks = [
        ("Environment", check_environment()),
        ("SUMO Files", check_sumo_files()),
        ("Network Content", check_network_content()),
        ("Route Content", check_route_content()),
        ("Vehicle Types", check_vehicle_types()),
        ("highD Data", check_highd_data()),
    ]
    
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    
    all_ok = True
    for name, status in checks:
        status_str = " PASS" if status else "❌ FAIL"
        print(f"{status_str} {name}")
        all_ok &= status
    
    print("\n" + "=" * 60)
    if all_ok:
        print(" All checks passed! You're ready to run the simulation.")
        print("\nRun: python3 main.py")
    else:
        print(" Some checks failed. Please fix the issues above.")
        print("\nTo fix:")
        print("  1. Run: bash complete_setup.sh")
        print("  2. Download highD dataset if needed")
        print("  3. Run this diagnostic again")
    print("=" * 60)


if __name__ == "__main__":
    main()