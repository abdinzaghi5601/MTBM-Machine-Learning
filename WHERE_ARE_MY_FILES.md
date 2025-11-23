# Where Are My Generated Files? 📂

## 🎯 Quick Answer

When you run Python scripts, files are saved in the **directory where you ran the command**.

---

## 📍 File Locations

### If you ran from the repository folder:
```
C:\Users\abdul\Desktop\ML for Tunneling\MTBM-Machine-Learning\
```

Your files are here:
- `mtbm_comprehensive_data.csv`
- `mtbm_time_series_analysis.png`
- `mtbm_deviation_analysis.png`
- `mtbm_performance_dashboard.png`
- `mtbm_correlation_matrix.png`

### If you ran from the parent folder:
```
C:\Users\abdul\Desktop\ML for Tunneling\
```

Your files are here:
- `avn2400_measurement_data.csv` ✅ (Found here!)
- Other generated files

---

## 🔍 How to Find Your Files

### Method 1: File Explorer (Easiest)

1. **Open File Explorer** (Windows + E)

2. **Navigate to:**
   ```
   C:\Users\abdul\Desktop\ML for Tunneling\
   ```

3. **Sort by "Date Modified"** (newest files at top)

4. **Look for files created today:**
   - `*.png` files (visualization images)
   - `*.csv` files (data files)
   - Files with "mtbm" or "avn" in the name

### Method 2: Search

1. **Open File Explorer**

2. **Go to:** `C:\Users\abdul\Desktop\ML for Tunneling\`

3. **In search box, type:** `*.png` or `*.csv`

4. **Sort by date modified**

### Method 3: Command Line

```bash
# Navigate to the folder
cd "C:\Users\abdul\Desktop\ML for Tunneling"

# List all PNG files
dir *.png /s

# List all CSV files
dir *.csv /s

# List files modified today
dir /OD
```

---

## 📊 Generated Files from Your Analysis

Based on the output you showed, you should have these files:

### 1. **Visualization Files (PNG images)**

**Time Series Analysis:**
```
mtbm_time_series_analysis.png
```
- 24 subplots showing all parameters over time
- Color-coded by parameter group
- Professional styling with grid lines

**Deviation Analysis:**
```
mtbm_deviation_analysis.png
```
- Scatter plot with tolerance circles
- Trend analysis over tunnel length
- Steering response analysis
- Quality statistics

**Performance Dashboard:**
```
mtbm_performance_dashboard.png
```
- Speed vs Pressure optimization
- Drilling efficiency trends
- Pressure balance monitoring
- Temperature warnings

**Correlation Matrix:**
```
mtbm_correlation_matrix.png
```
- Full parameter correlation heatmap
- Top 10 strongest correlations
- Statistical significance analysis

### 2. **Data Files (CSV)**

**Comprehensive Data:**
```
mtbm_comprehensive_data.csv
```
- 1,000 records of MTBM operation data
- All 23 parameters included
- Ready for further analysis

**AVN 2400 Measurement Data:**
```
avn2400_measurement_data.csv
```
- Precision measurement data
- Sub-millimeter accuracy analysis
- Quality control metrics

---

## 📂 Expected File Structure

```
C:\Users\abdul\Desktop\ML for Tunneling\
│
├── MTBM-Machine-Learning\          # Your repository
│   ├── (Python scripts here)
│   └── (Maybe some outputs here too)
│
├── mtbm_time_series_analysis.png   # Visualization 1
├── mtbm_deviation_analysis.png     # Visualization 2
├── mtbm_performance_dashboard.png  # Visualization 3
├── mtbm_correlation_matrix.png     # Visualization 4
├── mtbm_comprehensive_data.csv     # Generated data
└── avn2400_measurement_data.csv    # AVN 2400 results ✅
```

---

## 🎯 How to Open Your Files

### For PNG Images:

**Method 1: Double-click**
- Just double-click the .png file
- Opens in Windows Photo Viewer or default image viewer

**Method 2: Open with specific app**
- Right-click → Open with → Choose:
  - Paint (for editing)
  - Photos (for viewing)
  - Browser (Chrome, Edge, etc.)

**Method 3: In PowerPoint (for presentations)**
- Open PowerPoint
- Insert → Pictures
- Navigate to your PNG files
- Insert into presentation

### For CSV Files:

**Method 1: Excel (Recommended)**
- Double-click the .csv file
- Opens in Excel automatically
- You can view, sort, filter, and analyze

**Method 2: Python**
```python
import pandas as pd

# Load and view
df = pd.read_csv('mtbm_comprehensive_data.csv')
print(df.head())
print(df.describe())
```

**Method 3: Text Editor**
- Right-click → Open with → Notepad
- See raw CSV format

---

## 📸 Viewing Your Visualizations

### Quick Preview:

1. **Navigate to folder**
2. **Change view to "Large Icons" or "Extra Large Icons"**
3. **You'll see thumbnail previews of all PNG files**
4. **Click once to see larger preview in sidebar**

### Professional Review:

1. **Open in browser for full quality:**
   - Drag PNG file onto Chrome/Edge
   - Use zoom for details
   - Right-click → Print for high-quality PDF

2. **Create presentation:**
   - Open PowerPoint
   - Insert all 4 PNG images
   - Add titles and annotations
   - Present to management!

---

## 🔧 If Files Are Not Where Expected

### Check the current directory:

When you run a Python script, it saves files in the "current working directory".

**Find where you ran the script from:**

```bash
# In Python
import os
print("Files are saved here:")
print(os.getcwd())
```

**Common locations:**

1. **Repository folder:**
   ```
   C:\Users\abdul\Desktop\ML for Tunneling\MTBM-Machine-Learning\
   ```

2. **Parent folder:**
   ```
   C:\Users\abdul\Desktop\ML for Tunneling\
   ```

3. **User home folder (if you ran from anywhere else):**
   ```
   C:\Users\abdul\
   ```

---

## 📋 Quick File Finder Script

Create this file: `find_my_files.py`

```python
import os
import glob
from datetime import datetime, timedelta

# Search paths
search_paths = [
    r"C:\Users\abdul\Desktop\ML for Tunneling",
    r"C:\Users\abdul\Desktop\ML for Tunneling\MTBM-Machine-Learning",
    os.getcwd()
]

# File patterns to look for
patterns = ['*.png', '*.csv', 'mtbm*', 'avn*']

print("=" * 70)
print("🔍 SEARCHING FOR YOUR GENERATED FILES")
print("=" * 70)

# Get files from last 24 hours
yesterday = datetime.now() - timedelta(days=1)

for path in search_paths:
    print(f"\n📂 Searching in: {path}")

    if os.path.exists(path):
        for pattern in patterns:
            files = glob.glob(os.path.join(path, pattern))

            for file in files:
                # Check if file was modified in last 24 hours
                mtime = datetime.fromtimestamp(os.path.getmtime(file))

                if mtime > yesterday:
                    size = os.path.getsize(file) / 1024  # KB
                    print(f"   ✅ {os.path.basename(file)}")
                    print(f"      Full path: {file}")
                    print(f"      Size: {size:.1f} KB")
                    print(f"      Modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
                    print()
    else:
        print(f"   ⚠️ Path not found")

print("=" * 70)
```

**Run it:**
```bash
python find_my_files.py
```

This will show you **exactly** where all your files are!

---

## 💡 Pro Tips

### 1. Always know where your files go:

**Before running a script:**
```python
import os
print("Working directory:", os.getcwd())
```

**Or specify exact path:**
```python
# Instead of:
df.to_csv('output.csv')

# Use:
df.to_csv(r'C:\Users\abdul\Desktop\ML for Tunneling\MTBM-Machine-Learning\output.csv')
```

### 2. Create an output folder:

```python
import os

# Create outputs folder
output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save files there
df.to_csv(f'{output_dir}/data.csv')
plt.savefig(f'{output_dir}/plot.png')
```

### 3. Add timestamps to filenames:

```python
from datetime import datetime

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f'mtbm_analysis_{timestamp}.png'
plt.savefig(filename)
```

---

## ✅ Checklist - Find Your Files

- [ ] Open File Explorer
- [ ] Navigate to `C:\Users\abdul\Desktop\ML for Tunneling\`
- [ ] Sort by "Date Modified" (newest first)
- [ ] Look for files created today
- [ ] Check both parent folder and MTBM-Machine-Learning subfolder
- [ ] Open PNG files to view visualizations
- [ ] Open CSV files in Excel to view data

---

## 🎯 Summary

**Your files are most likely here:**

```
C:\Users\abdul\Desktop\ML for Tunneling\
```

**Confirmed file found:**
- `avn2400_measurement_data.csv` ✅

**Expected files (look for these):**
- `mtbm_time_series_analysis.png`
- `mtbm_deviation_analysis.png`
- `mtbm_performance_dashboard.png`
- `mtbm_correlation_matrix.png`
- `mtbm_comprehensive_data.csv`

**To find them:**
1. Open File Explorer
2. Go to `C:\Users\abdul\Desktop\ML for Tunneling\`
3. Sort by date (newest files at top)
4. Look for PNG and CSV files

**Your visualizations are ready to view and share with management! 📊🎉**
