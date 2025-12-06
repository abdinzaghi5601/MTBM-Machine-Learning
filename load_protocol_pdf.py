#!/usr/bin/env python3
"""
Load MTBM Data from Protocol PDF
=================================

This script extracts data from MTBM protocol PDF files.

Your PDF file: C:\\Users\\abdul\\Desktop\\ML for Tunneling\\3000 Measure Protocol.pdf

Usage:
    python load_protocol_pdf.py
"""

import pandas as pd
import numpy as np
import sys
import os

# Try to import PDF reader
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("⚠️ PyPDF2 not installed. Install with: pip install PyPDF2")

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    print("⚠️ pdfplumber not installed. Install with: pip install pdfplumber")

try:
    import tabula
    TABULA_AVAILABLE = True
except ImportError:
    TABULA_AVAILABLE = False
    print("⚠️ tabula-py not installed. Install with: pip install tabula-py")


def extract_tables_from_pdf(pdf_path):
    """
    Extract tables from PDF using multiple methods

    Args:
        pdf_path: Path to your PDF file

    Returns:
        List of DataFrames (one per table found)
    """

    print("=" * 70)
    print("📄 EXTRACTING DATA FROM PDF")
    print("=" * 70)
    print(f"\nFile: {pdf_path}")

    if not os.path.exists(pdf_path):
        print(f"\n❌ Error: File not found!")
        print(f"Looking for: {pdf_path}")
        print("\nPlease check:")
        print("1. File path is correct")
        print("2. File exists in the specified location")
        return None

    all_tables = []

    # Method 1: Try tabula-py (best for structured tables)
    if TABULA_AVAILABLE:
        print("\n🔧 Method 1: Using tabula-py...")
        try:
            tables = tabula.read_pdf(
                pdf_path,
                pages='all',
                multiple_tables=True,
                lattice=True  # For tables with visible borders
            )

            if tables:
                print(f"✅ Found {len(tables)} tables with tabula-py")
                all_tables.extend(tables)
            else:
                print("⚠️ No tables found with tabula-py")

        except Exception as e:
            print(f"⚠️ tabula-py error: {e}")

    # Method 2: Try pdfplumber (good for various PDF types)
    if PDFPLUMBER_AVAILABLE and len(all_tables) == 0:
        print("\n🔧 Method 2: Using pdfplumber...")
        try:
            import pdfplumber

            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages, 1):
                    print(f"   Scanning page {i}...")

                    tables = page.extract_tables()

                    if tables:
                        print(f"   ✅ Found {len(tables)} tables on page {i}")

                        for table in tables:
                            if table:
                                df = pd.DataFrame(table[1:], columns=table[0])
                                all_tables.append(df)

        except Exception as e:
            print(f"⚠️ pdfplumber error: {e}")

    # Method 3: Extract text and parse manually
    if len(all_tables) == 0 and PDF_AVAILABLE:
        print("\n🔧 Method 3: Extracting text from PDF...")
        print("⚠️ No structured tables found.")
        print("   You may need to manually export data from PDF to Excel/CSV")

        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)

                print(f"\nPDF Info:")
                print(f"   Pages: {len(pdf_reader.pages)}")

                # Extract text from first 2 pages
                print("\nFirst page preview:")
                print("-" * 70)
                text = pdf_reader.pages[0].extract_text()
                print(text[:500])  # First 500 characters
                print("-" * 70)

        except Exception as e:
            print(f"⚠️ Text extraction error: {e}")

    if len(all_tables) > 0:
        print(f"\n✅ Total tables extracted: {len(all_tables)}")
        return all_tables
    else:
        print("\n❌ Could not extract tables from PDF")
        print("\n💡 Recommended Solutions:")
        print("1. Convert PDF to Excel manually and use load_real_data.py")
        print("2. Copy-paste data from PDF to Excel")
        print("3. Use PDF software to export tables")
        return None


def load_protocol_pdf_manual():
    """
    Guide for manually loading protocol data
    """

    print("\n" + "=" * 70)
    print("📋 MANUAL DATA EXTRACTION GUIDE")
    print("=" * 70)

    print("\nYour protocol PDF contains measurement data.")
    print("Here's how to get it into the ML system:\n")

    print("OPTION 1: Convert PDF to Excel (Recommended)")
    print("-" * 70)
    print("1. Open your PDF: '3000 Measure Protocol.pdf'")
    print("2. Use Adobe Acrobat (if available):")
    print("   File → Export To → Spreadsheet → Microsoft Excel")
    print("\n3. Or use online converter:")
    print("   - https://www.adobe.com/acrobat/online/pdf-to-excel.html")
    print("   - https://www.ilovepdf.com/pdf_to_excel")
    print("\n4. Save as: 'AVN3000_Data.xlsx'")
    print("5. Then run: python load_real_data.py")

    print("\n\nOPTION 2: Manual Copy-Paste")
    print("-" * 70)
    print("1. Open PDF and select data table")
    print("2. Copy the data (Ctrl+C)")
    print("3. Open Excel and paste (Ctrl+V)")
    print("4. Clean up the data:")
    print("   - Remove header rows if needed")
    print("   - Ensure columns are properly separated")
    print("   - Save as 'AVN3000_Data.xlsx'")
    print("5. Run: python load_real_data.py")

    print("\n\nOPTION 3: Use Tabula (Free Software)")
    print("-" * 70)
    print("1. Download Tabula: https://tabula.technology/")
    print("2. Open your PDF in Tabula")
    print("3. Select the data table")
    print("4. Export as CSV")
    print("5. Run: python load_real_data.py")

    print("\n" + "=" * 70)


def main():
    """Main function"""

    print("\n" + "=" * 70)
    print("🚀 MTBM PROTOCOL PDF LOADER")
    print("=" * 70)

    # ================================================
    # YOUR PDF FILE PATH - EDIT THIS!
    # ================================================

    pdf_file = r"C:\Users\abdul\Desktop\ML for Tunneling\3000 Measure Protocol.pdf"

    # Alternative: Use relative path if PDF is in same folder
    # pdf_file = "3000 Measure Protocol.pdf"

    # ================================================

    print(f"\nTarget PDF: {pdf_file}")

    # Check if PDF libraries are available
    if not (TABULA_AVAILABLE or PDFPLUMBER_AVAILABLE or PDF_AVAILABLE):
        print("\n❌ No PDF libraries installed!")
        print("\nPlease install at least one:")
        print("   pip install tabula-py")
        print("   pip install pdfplumber")
        print("   pip install PyPDF2")
        print("\nOr use manual extraction (see guide below)")
        load_protocol_pdf_manual()
        return

    # Try to extract tables
    tables = extract_tables_from_pdf(pdf_file)

    if tables:
        print("\n" + "=" * 70)
        print("📊 EXTRACTED TABLES")
        print("=" * 70)

        for i, df in enumerate(tables, 1):
            print(f"\nTable {i}:")
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {list(df.columns)}")
            print(f"\nFirst few rows:")
            print(df.head())

            # Save table
            output_file = f"protocol_table_{i}.csv"
            df.to_csv(output_file, index=False)
            print(f"\n💾 Saved to: {output_file}")

        print("\n" + "=" * 70)
        print("✅ NEXT STEPS")
        print("=" * 70)
        print("\n1. Review the extracted CSV files")
        print("2. Choose the file with your measurement data")
        print("3. Edit load_real_data.py to use that CSV file:")
        print("   USE_CSV = True")
        print("   csv_file = 'protocol_table_1.csv'")
        print("4. Run: python load_real_data.py")

    else:
        # Show manual guide
        load_protocol_pdf_manual()

        print("\n" + "=" * 70)
        print("📝 QUICK START TEMPLATE")
        print("=" * 70)
        print("\nCreate a CSV/Excel file with these columns:")
        print("-" * 70)

        # Create template
        template = pd.DataFrame({
            'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'Chainage': [10.5, 11.2, 12.8],
            'Ground_Type': ['Clay', 'Sand', 'Clay'],
            'Thrust_kN': [1250, 1450, 1300],
            'Torque_kNm': [210, 245, 220],
            'RPM': [8.5, 8.2, 8.7],
            'Speed_mm_min': [35.2, 28.3, 34.1],
            'Pressure_bar': [130, 145, 135]
        })

        print(template)

        # Save template
        template.to_csv('mtbm_data_template.csv', index=False)
        template.to_excel('mtbm_data_template.xlsx', index=False)

        print("\n💾 Template files created:")
        print("   - mtbm_data_template.csv")
        print("   - mtbm_data_template.xlsx")
        print("\nFill this template with your protocol data!")
        print("Then run: python load_real_data.py")


if __name__ == "__main__":
    main()
