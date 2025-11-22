# How to Use Protocol PDF Data - Simple Guide

## 🎯 Your File Location
```
C:\Users\abdul\Desktop\ML for Tunneling\3000 Measure Protocol.pdf
```

---

## ⚡ FASTEST METHOD (5 Minutes)

### Option 1: Convert PDF to Excel (Recommended)

**Step 1:** Open your PDF
- File: `3000 Measure Protocol.pdf`

**Step 2:** Convert to Excel using one of these:

**A. Adobe Acrobat (if you have it):**
1. Open PDF in Adobe Acrobat
2. File → Export To → Spreadsheet → Microsoft Excel
3. Save as: `AVN3000_Data.xlsx`

**B. Online Converter (Free):**
1. Go to: https://www.ilovepdf.com/pdf_to_excel
2. Upload your PDF: `3000 Measure Protocol.pdf`
3. Click "Convert to Excel"
4. Download the Excel file
5. Save as: `AVN3000_Data.xlsx`

**C. Microsoft Word:**
1. Open PDF in Microsoft Word (it will convert automatically)
2. Copy the table
3. Paste into Excel
4. Save as: `AVN3000_Data.xlsx`

**Step 3:** Use the Excel file

Now edit `load_real_data.py` at **line 90**:

```python
# EDIT THIS SECTION FOR YOUR DATA
USE_EXCEL = True
excel_file = r'C:\Users\abdul\Desktop\ML for Tunneling\AVN3000_Data.xlsx'
sheet_name = 'Sheet1'  # Or whatever sheet name you have
```

**Step 4:** Run it!
```bash
python load_real_data.py
```

---

## 🔧 Option 2: Manual Copy-Paste (10 Minutes)

1. **Open your PDF** (`3000 Measure Protocol.pdf`)

2. **Find the data table** in the PDF

3. **Select and copy** the table (Ctrl+A to select, Ctrl+C to copy)

4. **Open Excel** (blank workbook)

5. **Paste** the data (Ctrl+V)

6. **Clean up if needed:**
   - Remove extra header rows
   - Make sure columns are properly aligned
   - Check that numbers look correct

7. **Save as:** `AVN3000_Data.xlsx`

8. **Edit `load_real_data.py`** (line 90):
```python
USE_EXCEL = True
excel_file = r'C:\Users\abdul\Desktop\ML for Tunneling\AVN3000_Data.xlsx'
sheet_name = 'Sheet1'
```

9. **Run:**
```bash
python load_real_data.py
```

---

## 🤖 Option 3: Automatic PDF Extraction

**Step 1:** Install PDF tools
```bash
pip install tabula-py pdfplumber PyPDF2
```

**Step 2:** Run the PDF loader
```bash
python load_protocol_pdf.py
```

This will automatically:
- Extract tables from your PDF
- Save them as CSV files
- Tell you which file to use

**Step 3:** Use the extracted CSV

The script will create files like:
- `protocol_table_1.csv`
- `protocol_table_2.csv`

Then edit `load_real_data.py` (line 95):
```python
USE_EXCEL = False
USE_CSV = True
csv_file = 'protocol_table_1.csv'  # Use the one with your data
```

**Step 4:** Run:
```bash
python load_real_data.py
```

---

## 📋 What Data Do You Need?

From your AVN 3000 protocol, look for tables with these columns:

### Required (Minimum):
- **Date/Time** - When measurements were taken
- **Ground Type** - Clay, Sand, Rock, etc.
- **Thrust Force** - Usually in kN
- **Advance Speed** - Usually in mm/min

### Recommended (For Better Accuracy):
- **Torque** - kN·m
- **RPM** - Revolutions per minute
- **Earth Pressure** - bar
- **Chainage** - Position in tunnel (meters)

### Optional (Nice to Have):
- **Horizontal Deviation** - mm
- **Vertical Deviation** - mm
- **Cutter Wear** - mm

---

## 🎯 Example: What Your Data Should Look Like

After converting from PDF, your Excel should look like:

| Date       | Chainage | Ground | Thrust | Torque | RPM  | Speed | Pressure |
|------------|----------|--------|--------|--------|------|-------|----------|
| 2024-01-01 | 10.5     | Clay   | 1250   | 210    | 8.5  | 35.2  | 130      |
| 2024-01-01 | 11.2     | Clay   | 1280   | 215    | 8.7  | 36.1  | 132      |
| 2024-01-02 | 12.8     | Sand   | 1450   | 245    | 8.2  | 28.3  | 145      |
| 2024-01-02 | 14.1     | Sand   | 1480   | 250    | 8.0  | 27.8  | 148      |

---

## ⚙️ Mapping Your Column Names

Once you have your Excel file, you need to tell the script what your columns are called.

In `load_real_data.py` find **line 35-75** (the `column_mapping` section):

### Your PDF might have columns like:
- "Messung Nr." → `timestamp`
- "Station" → `chainage`
- "Boden" or "Soil" → `geological_type`
- "Vorschubkraft" or "Thrust" → `thrust_force`
- "Drehmoment" or "Torque" → `torque`
- "Drehzahl" or "RPM" → `rpm`
- "Vorschubgeschwindigkeit" or "Speed" → `advance_speed`

### Update the mapping:

```python
column_mapping = {
    # Replace the left side with YOUR exact column names from the PDF
    'Messung Nr.': 'timestamp',           # Your date column
    'Station': 'chainage',                # Your position column
    'Boden': 'geological_type',          # Your ground type column
    'Vorschubkraft': 'thrust_force',     # Your thrust column
    'Drehmoment': 'torque',              # Your torque column
    'Drehzahl': 'rpm',                   # Your RPM column
    'Vorschubgeschwindigkeit': 'advance_speed',  # Your speed column
}
```

---

## 🚀 Complete Workflow

### Step-by-Step:

1. **Convert PDF to Excel**
   - Use online tool: https://www.ilovepdf.com/pdf_to_excel
   - Upload: `3000 Measure Protocol.pdf`
   - Download Excel file
   - Save as: `AVN3000_Data.xlsx`

2. **Open `load_real_data.py` in text editor**

3. **Edit line 90-93:**
```python
USE_EXCEL = True
excel_file = r'C:\Users\abdul\Desktop\ML for Tunneling\AVN3000_Data.xlsx'
sheet_name = 'Sheet1'
```

4. **Edit line 35-75 (column mapping):**
   - Look at your Excel column names
   - Update the mapping to match

5. **Save the file**

6. **Run:**
```bash
cd "C:\Users\abdul\Desktop\ML for Tunneling\MTBM-Machine-Learning"
python load_real_data.py
```

7. **If successful, you'll see:**
```
✅ Successfully loaded X rows
✅ Cleaned data: X rows
💾 Saved cleaned data to: cleaned_mtbm_data.csv
```

8. **Train model:**
```bash
python train_with_real_data.py
```

9. **Make predictions:**
```bash
python make_predictions.py
```

---

## 🆘 Troubleshooting

### Problem: "File not found"
**Solution:** Check file path. Use full path with `r` prefix:
```python
excel_file = r'C:\Users\abdul\Desktop\ML for Tunneling\AVN3000_Data.xlsx'
```

### Problem: "Column not found"
**Solution:**
1. Open your Excel file
2. Write down exact column names
3. Update `column_mapping` with those exact names

### Problem: PDF conversion looks messy
**Solution:**
1. Try different converter
2. Or manually copy-paste table by table
3. Or use Option 3 (automatic extraction)

### Problem: Data has weird characters
**Solution:**
1. Open Excel file
2. Find and replace weird characters
3. Save and try again

---

## 💡 Pro Tips

1. **Check your Excel first** - Open the converted Excel and make sure data looks good before running Python

2. **Start small** - If you have lots of data, try with just 100 rows first to make sure it works

3. **Save intermediate files** - Keep copies of your Excel files in case you need to start over

4. **Column names matter** - The exact spelling and spacing must match in `column_mapping`

---

## ✅ Quick Checklist

Before running `load_real_data.py`:

- [ ] I converted my PDF to Excel
- [ ] Excel file is saved in the right location
- [ ] I can open the Excel file and see the data
- [ ] I updated `excel_file` path in load_real_data.py (line 92)
- [ ] I updated `column_mapping` with my column names (line 35-75)
- [ ] I saved the changes to load_real_data.py

---

## 🎯 Summary

**Easiest path:**
1. Convert PDF to Excel (5 minutes)
2. Edit 2 lines in `load_real_data.py`
3. Run `python load_real_data.py`
4. Done!

**You DON'T need to:**
- Write any code
- Understand Python
- Install special PDF tools (unless you want automatic extraction)

**You just need to:**
- Convert PDF to Excel (many free tools available)
- Edit 2 sections in the script (file path and column names)
- Run the script

**That's it!** 🚀
