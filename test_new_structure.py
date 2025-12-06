#!/usr/bin/env python3
"""
Test New Repository Structure
============================

This script tests the new organized repository structure and
provides guidance on using the reorganized files.
"""

import os
import sys

def test_folder_structure():
    """Test if the new folder structure exists"""
    
    print("🧪 TESTING NEW REPOSITORY STRUCTURE")
    print("=" * 50)
    
    # Expected folders
    expected_folders = [
        'core',
        'core/frameworks',
        'core/visualization', 
        'core/data_processing',
        'tools',
        'tools/testing',
        'outputs',
        'outputs/graphs',
        'outputs/models',
        'outputs/reports',
        'docs',
        'docs/guides',
        'config'
    ]
    
    print("📁 FOLDER STRUCTURE TEST:")
    all_folders_exist = True
    
    for folder in expected_folders:
        exists = os.path.exists(folder)
        status = "✅" if exists else "❌"
        print(f"  {status} {folder}")
        if not exists:
            all_folders_exist = False
    
    return all_folders_exist

def check_file_locations():
    """Check if files are in their new locations"""
    
    print(f"\n📄 FILE LOCATION TEST:")
    
    # Expected file locations
    expected_files = {
        'core/frameworks/unified_mtbm_ml_framework.py': 'Unified Framework',
        'core/frameworks/avn2400_advanced_measurement_ml.py': 'AVN 2400 Framework',
        'core/visualization/create_graphs_direct.py': 'Graph Creator',
        'core/visualization/mtbm_comprehensive_plotting.py': 'Comprehensive Plotting',
        'core/data_processing/load_real_data.py': 'Data Loader',
        'tools/quickstart_demo.py': 'Quick Demo',
        'tools/find_graph_location.py': 'File Finder',
        'config/requirements.txt': 'Dependencies'
    }
    
    files_found = 0
    total_files = len(expected_files)
    
    for file_path, description in expected_files.items():
        exists = os.path.exists(file_path)
        status = "✅" if exists else "❌"
        print(f"  {status} {description}: {file_path}")
        if exists:
            files_found += 1
    
    return files_found, total_files

def show_usage_examples():
    """Show how to use the new structure"""
    
    print(f"\n🚀 USAGE EXAMPLES:")
    print("=" * 30)
    
    examples = [
        ("Generate Graphs", "python core/visualization/create_graphs_direct.py"),
        ("Run Demo", "python tools/quickstart_demo.py"),
        ("Find Files", "python tools/find_graph_location.py"),
        ("Load Data", "python core/data_processing/load_real_data.py"),
        ("Unified Framework", "python core/frameworks/unified_mtbm_ml_framework.py")
    ]
    
    for description, command in examples:
        print(f"  📊 {description}:")
        print(f"     {command}")
        print()

def main():
    """Main test function"""
    
    # Test folder structure
    folders_ok = test_folder_structure()
    
    # Check file locations
    files_found, total_files = check_file_locations()
    
    # Show usage examples
    show_usage_examples()
    
    # Summary
    print("📊 REORGANIZATION SUMMARY:")
    print("=" * 30)
    print(f"  Folder Structure: {'✅ Complete' if folders_ok else '⚠️ Incomplete'}")
    print(f"  Files Organized: {files_found}/{total_files} files in new locations")
    
    if files_found < total_files:
        print(f"\n⚠️ MANUAL STEPS NEEDED:")
        print(f"  Some files may need to be manually moved to their new locations.")
        print(f"  See NEW_REPOSITORY_STRUCTURE.md for complete guidance.")
    else:
        print(f"\n🎉 REORGANIZATION SUCCESSFUL!")
        print(f"  Your repository is now professionally organized!")
    
    print(f"\n📚 DOCUMENTATION:")
    print(f"  - NEW_REPOSITORY_STRUCTURE.md - Complete structure guide")
    print(f"  - REPOSITORY_REORGANIZATION_PLAN.md - Planning document")

if __name__ == "__main__":
    main()
