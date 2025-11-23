#!/usr/bin/env python3
"""
Find Graph Location Script
==========================
This script shows you exactly where the graph files will be saved
and helps you locate them after generation.
"""

import os
import glob

def show_current_directory():
    """Show current working directory and contents"""
    current_dir = os.getcwd()
    print("🔍 CURRENT WORKING DIRECTORY:")
    print(f"   {current_dir}")
    
    print("\n📁 FILES IN CURRENT DIRECTORY:")
    files = os.listdir(current_dir)
    
    # Show Python files
    python_files = [f for f in files if f.endswith('.py')]
    if python_files:
        print("   📄 Python Scripts:")
        for file in sorted(python_files):
            print(f"      {file}")
    
    # Show PNG files (graphs)
    png_files = [f for f in files if f.endswith('.png')]
    if png_files:
        print("   📊 Graph Files (PNG):")
        for file in sorted(png_files):
            file_path = os.path.join(current_dir, file)
            file_size = os.path.getsize(file_path)
            print(f"      {file} ({file_size:,} bytes)")
    else:
        print("   📊 Graph Files (PNG): None found")
    
    # Show CSV files (data)
    csv_files = [f for f in files if f.endswith('.csv')]
    if csv_files:
        print("   📈 Data Files (CSV):")
        for file in sorted(csv_files):
            print(f"      {file}")
    
    return current_dir

def find_mtbm_graphs():
    """Find MTBM graph files anywhere in the system"""
    print("\n🔍 SEARCHING FOR MTBM GRAPH FILES...")
    
    # Graph file patterns to search for
    graph_patterns = [
        '*mtbm*.png',
        '*time_series*.png',
        '*deviation*.png',
        '*performance*.png',
        '*correlation*.png',
        '1_mtbm_time_series.png',
        '2_mtbm_deviation_analysis.png',
        '3_mtbm_performance_dashboard.png',
        '4_mtbm_correlation_matrix.png'
    ]
    
    found_files = []
    
    # Search in current directory and subdirectories
    current_dir = os.getcwd()
    for pattern in graph_patterns:
        matches = glob.glob(os.path.join(current_dir, '**', pattern), recursive=True)
        found_files.extend(matches)
    
    # Remove duplicates
    found_files = list(set(found_files))
    
    if found_files:
        print("   ✅ FOUND GRAPH FILES:")
        for file_path in sorted(found_files):
            file_size = os.path.getsize(file_path)
            print(f"      {file_path} ({file_size:,} bytes)")
    else:
        print("   ❌ NO GRAPH FILES FOUND")
        print("      Run one of these commands to generate graphs:")
        print("      python create_graphs_direct.py")
        print("      python generate_mtbm_graphs.py")
        print("      python mtbm_comprehensive_plotting.py")
    
    return found_files

def show_expected_locations():
    """Show where graph files should be created"""
    print("\n📍 EXPECTED GRAPH FILE LOCATIONS:")
    
    current_dir = os.getcwd()
    expected_files = [
        '1_mtbm_time_series.png',
        '2_mtbm_deviation_analysis.png', 
        '3_mtbm_performance_dashboard.png',
        '4_mtbm_correlation_matrix.png',
        'mtbm_data.csv'
    ]
    
    for filename in expected_files:
        full_path = os.path.join(current_dir, filename)
        exists = os.path.exists(full_path)
        status = "✅ EXISTS" if exists else "❌ NOT FOUND"
        print(f"   {status}: {full_path}")

def main():
    """Main execution"""
    print("🚀 MTBM GRAPH LOCATION FINDER")
    print("=" * 50)
    
    # Show current directory and contents
    current_dir = show_current_directory()
    
    # Find existing graph files
    found_files = find_mtbm_graphs()
    
    # Show expected locations
    show_expected_locations()
    
    print("\n💡 INSTRUCTIONS:")
    print("   1. Graph files will be saved in the current working directory")
    print(f"   2. Current directory: {current_dir}")
    print("   3. To generate graphs, run: python create_graphs_direct.py")
    print("   4. After generation, run this script again to see the files")
    
    print("\n🔧 TROUBLESHOOTING:")
    print("   - If graphs aren't in expected location, check your current directory")
    print("   - Make sure you're running scripts from the MTBM-Machine-Learning folder")
    print("   - Files might be in your user home directory if run from there")

if __name__ == "__main__":
    main()
