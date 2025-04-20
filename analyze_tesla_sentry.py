#!/usr/bin/env python3
# Tesla Sentry Mode Analysis Tools
# Copyright (c) 2025 Ariel Jacob
# MIT License - See LICENSE file for details

import os
import json
import argparse
from collections import defaultdict

def analyze_tesla_sentry(root_folder):
    """
    Analyze Tesla Sentry Mode recordings to find camera events and reasons.
    
    This function scans the given directory and its subdirectories for event.json files,
    extracting information about which cameras triggered events and why.
    
    Args:
        root_folder (str): Path to the root folder containing Tesla Sentry Mode recordings
        
    Returns:
        dict: A dictionary containing:
            - 'camera_events': Count of events per camera
            - 'camera_folders': List of event folders per camera
            - 'reason_counts': Count of events per reason
    """
    camera_events = defaultdict(int)
    camera_folders = defaultdict(set)
    reason_counts = defaultdict(int)
    
    # Walk through all subfolders and files
    for dirpath, dirnames, filenames in os.walk(root_folder):
        if 'event.json' in filenames:
            event_path = os.path.join(dirpath, 'event.json')
            try:
                with open(event_path, 'r') as f:
                    event_data = json.load(f)
                camera = str(event_data.get('camera', 'Unknown'))
                reason = str(event_data.get('reason', 'Unknown'))
                folder_name = os.path.basename(dirpath)
                
                camera_events[camera] += 1
                camera_folders[camera].add(folder_name)
                reason_counts[reason] += 1
            except Exception as e:
                print(f"Error processing {event_path}: {e}")
    
    # Convert sets to sorted lists for output
    camera_folders = {k: sorted(list(v)) for k, v in camera_folders.items()}
    
    return {
        'camera_events': dict(camera_events),
        'camera_folders': camera_folders,
        'reason_counts': dict(reason_counts)
    }

def main():
    """
    Main entry point for the Tesla Sentry Mode analysis tool.
    
    Parses command line arguments, analyzes Tesla Sentry Mode recordings,
    and generates a report of camera events and reasons.
    """
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description='Analyze Tesla Sentry Mode recordings to find camera events and reasons'
    )
    parser.add_argument('root_folder', nargs='?', default=None,
                        help='Root folder containing Tesla Sentry Mode recordings (e.g., D:\\TeslaCam)')
    
    args = parser.parse_args()
    
    # Check if root_folder was provided
    if args.root_folder is None:
        parser.print_help()
        return
    
    root_folder = args.root_folder
    
    # Check if the folder exists
    if not os.path.isdir(root_folder):
        print(f"Error: The folder '{root_folder}' does not exist.")
        return
    
    print(f"\nAnalyzing Tesla Sentry Mode recordings in {root_folder}...")
    results = analyze_tesla_sentry(root_folder)
    
    # Print camera counts and folders in the requested format
    print("\n=== Camera Event Counts ===\n")
    for camera, count in sorted(results['camera_events'].items()):
        print(f"camera {camera}: {count} events")
        folders = ','.join(results['camera_folders'][camera])
        print(f"{folders}\n")
    
    # Print reason counts
    print("\n=== Reason Counts ===\n")
    for reason, count in sorted(results['reason_counts'].items()):
        print(f"reason {reason}: {count} times")
    
    # Save results to a file
    output_file = os.path.join(root_folder, "SentryModeAnalysis.txt")
    with open(output_file, 'w') as f:
        f.write("=== Camera Event Counts ===\n\n")
        for camera, count in sorted(results['camera_events'].items()):
            f.write(f"camera {camera}: {count} events\n")
            folders = ','.join(results['camera_folders'][camera])
            f.write(f"{folders}\n\n")
        
        f.write("\n=== Reason Counts ===\n\n")
        for reason, count in sorted(results['reason_counts'].items()):
            f.write(f"reason {reason}: {count} times\n")
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
