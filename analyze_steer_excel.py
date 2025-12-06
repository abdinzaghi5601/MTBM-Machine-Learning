"""
Analyze Steer-cyl-cal-rev8..xls Excel File
===========================================

Reads and analyzes the steering cylinder calculation Excel file.
"""

import pandas as pd
import xlrd
from pathlib import Path

excel_file = Path(r"C:\Users\abdul\Desktop\ML for Tunneling\Steer-cyl-cal-rev8..xls")

print("="*80)
print("ANALYZING: Steer-cyl-cal-rev8..xls")
print("="*80)

# First, get all sheet names
try:
    excel_file_obj = pd.ExcelFile(excel_file)
    sheet_names = excel_file_obj.sheet_names
    
    print(f"\nFound {len(sheet_names)} sheets:")
    for i, sheet in enumerate(sheet_names, 1):
        print(f"  {i}. {sheet}")
    
    # Analyze each sheet
    for sheet_name in sheet_names:
        print(f"\n{'='*80}")
        print(f"SHEET: {sheet_name}")
        print(f"{'='*80}")
        
        try:
            # Read the sheet
            df = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)
            
            print(f"\nDimensions: {df.shape[0]} rows × {df.shape[1]} columns")
            
            # Show first 20 rows
            print(f"\nFirst 20 rows:")
            print("-"*80)
            pd.set_option('display.max_rows', 20)
            pd.set_option('display.max_columns', 15)
            pd.set_option('display.width', 120)
            print(df.head(20).to_string())
            
            # Look for key terms
            print(f"\n\nSearching for key terms (first 50 rows):")
            print("-"*80)
            key_terms = ['cylinder', 'pitch', 'yaw', 'stroke', 'mounting', 'diameter', 
                        'pipe', 'length', 'vertical', 'horizontal', 'target', 'laser']
            
            found_terms = {}
            for row_idx in range(min(50, len(df))):
                for col_idx in range(min(15, len(df.columns))):
                    cell_value = str(df.iloc[row_idx, col_idx]).lower()
                    for term in key_terms:
                        if term in cell_value:
                            if term not in found_terms:
                                found_terms[term] = []
                            found_terms[term].append((row_idx+1, col_idx+1, df.iloc[row_idx, col_idx]))
            
            if found_terms:
                for term, locations in found_terms.items():
                    print(f"\n'{term.upper()}' found at:")
                    for row, col, value in locations[:5]:  # Show first 5 occurrences
                        print(f"  Row {row}, Col {col}: {value}")
            else:
                print("  No key terms found in first 50 rows")
                
        except Exception as e:
            print(f"Error reading sheet '{sheet_name}': {e}")
    
    # Try to read with xlrd for formula extraction
    print(f"\n{'='*80}")
    print("ATTEMPTING FORMULA EXTRACTION (if available)")
    print(f"{'='*80}")
    
    try:
        workbook = xlrd.open_workbook(str(excel_file), formatting_info=False)
        print(f"\nWorkbook opened successfully with xlrd")
        print(f"Number of sheets: {workbook.nsheets}")
        
        # Check a specific sheet for formulas (e.g., the first calculation sheet)
        for sheet_idx in range(min(3, workbook.nsheets)):
            sheet = workbook.sheet_by_index(sheet_idx)
            print(f"\nSheet '{sheet.name}': {sheet.nrows} rows × {sheet.ncols} columns")
            
            # Look for formula cells in first 30 rows
            formula_count = 0
            for row_idx in range(min(30, sheet.nrows)):
                for col_idx in range(min(15, sheet.ncols)):
                    cell = sheet.cell(row_idx, col_idx)
                    if cell.ctype == xlrd.XL_CELL_FORMULA:
                        formula_count += 1
                        if formula_count <= 5:  # Show first 5 formulas
                            print(f"  Formula at Row {row_idx+1}, Col {col_idx+1}: {cell.value}")
            
            if formula_count > 5:
                print(f"  ... and {formula_count - 5} more formulas")
                
    except Exception as e:
        print(f"Could not extract formulas: {e}")
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")

except Exception as e:
    print(f"Error opening Excel file: {e}")
    import traceback
    traceback.print_exc()

