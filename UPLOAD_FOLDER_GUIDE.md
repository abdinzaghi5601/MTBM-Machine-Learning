# 📁 Upload Here Folder - Complete Guide

## 🎯 What Is This?

The **"Upload Here"** folder is your dedicated drop zone for CSV data files. Just put your files here, run the program, and it automatically finds them!

---

## 🚀 Super Simple Usage

### Step 1: Put Your CSV File Here
```
ML for Tunneling/
└── Upload Here/          ← PUT FILES HERE!
    └── your_data.csv
```

**How to add files:**
- Drag and drop into the "Upload Here" folder
- Copy-paste your CSV files
- Move files from other locations
- Double-click `Open Upload Folder.bat` to open it quickly

### Step 2: Run the Analysis
```bash
cd MTBM-Machine-Learning
python auto_protocol_analyzer.py
```

### Step 3: Select Your File
The program shows:
```
================================================================================
INTERACTIVE FILE SELECTION
================================================================================

📂 Found 2 file(s) in 'Upload Here' folder

📁 Found 2 CSV file(s):

  [1] tunnel_data.csv
      Path: C:\...\Upload Here
      Size: 245.3 KB

  [2] section_A.csv
      Path: C:\...\Upload Here
      Size: 128.5 KB

  [0] Enter custom path
  [G] Generate sample data
  [Q] Quit

Select file (number/0/G/Q):
```

Just type `1` or `2` to select your file!

---

## 💡 Why This Is Better

### Before (Hard Way):
1. Remember the exact file path
2. Type long path like: `C:\Users\abdul\Desktop\Project\Data\file.csv`
3. Easy to make typos
4. Annoying with spaces in path

### Now (Easy Way):
1. Drop file in "Upload Here" folder
2. Run program
3. Type `1` or `2`
4. Done! ✅

---

## 📋 Search Priority

The program searches folders in this order:

1. **"Upload Here" folder** ⭐ (Highest priority - SEARCHES HERE FIRST!)
2. `data/raw/` folder
3. `data/` folder
4. Current directory
5. Parent directory

**Files in "Upload Here" are shown first!** This makes it easy to identify your recently uploaded files.

---

## 🎯 Best Practices

### ✅ Good File Names
```
tunnel_section_A_nov24.csv
daily_monitoring_2024-11-24.csv
project_alpha_chainage_100-200.csv
```

### ❌ Avoid
```
data.csv          (too generic)
Copy of file.csv  (confusing)
temp (1).csv      (unclear)
```

### 📂 Organization Tips

**During Active Work:**
```
Upload Here/
├── current_section.csv
├── today_data.csv
└── test_run.csv
```

**After Analysis:**
Move analyzed files to organize:
```
data/
├── raw/
│   └── original_files/
└── processed/
    └── analyzed_files/
```

---

## 🔧 Advanced Features

### Multiple Files
You can have many files in "Upload Here":
```
Upload Here/
├── section_A.csv
├── section_B.csv
├── section_C.csv
├── daily_nov23.csv
└── daily_nov24.csv
```

The program shows them all with file sizes to help you pick the right one.

### Auto-Cleanup Option
After analysis, you can:
- Keep files in "Upload Here" for quick re-analysis
- Move to `data/processed/` for archival
- Delete if no longer needed

The program never modifies or deletes your original files!

---

## 📊 Typical Workflow

### Daily Monitoring
```
1. Export CSV from MTBM system
2. Copy to "Upload Here" folder
3. Run: python auto_protocol_analyzer.py
4. Select today's file
5. Review results
6. Move file to data/processed/ when done
```

### Project Analysis
```
1. Collect all section CSV files
2. Put them in "Upload Here" folder
3. Run analysis on each section
4. Compare results
5. Archive analyzed files
```

---

## 🎨 Quick Access Methods

### Method 1: Double-Click Shortcut (Windows)
**Double-click: `Open Upload Folder.bat`**
- Opens "Upload Here" folder instantly
- Located in main "ML for Tunneling" folder

### Method 2: Windows Explorer
```
Navigate to: ML for Tunneling\Upload Here
```

### Method 3: File Explorer Address Bar
```
Paste: %USERPROFILE%\Desktop\ML for Tunneling\Upload Here
```

### Method 4: Create Desktop Shortcut
1. Right-click "Upload Here" folder
2. Send to → Desktop (create shortcut)
3. Rename to "MTBM Data Upload"

---

## ⚙️ Technical Details

### Folder Location
```
C:\Users\abdul\Desktop\ML for Tunneling\Upload Here\
```

### Search Pattern
The interactive helper searches for `*.csv` files using:
```python
upload_dir = default_dir.parent / 'Upload Here'
csv_files = list(upload_dir.glob('*.csv'))
```

### File Requirements
- ✅ Must have `.csv` extension
- ✅ Can be any size
- ✅ Can have any name
- ✅ Multiple files OK

### What Happens
1. Program scans "Upload Here" first
2. Shows files with sizes
3. You select by number
4. Program reads the file
5. Auto-detects protocol
6. Runs complete analysis
7. **Original file is never modified**

---

## 🎯 Common Scenarios

### Scenario 1: New User - First Test
```
1. Run program without files: python auto_protocol_analyzer.py
2. Select [G] to generate sample data
3. Program creates test CSV
4. Complete your first analysis!
```

### Scenario 2: Regular Use
```
1. Drop today's CSV in "Upload Here"
2. Run: python auto_protocol_analyzer.py
3. Select [1] for your file
4. Select diameter
5. Press Enter for defaults
6. Check results!
```

### Scenario 3: Multiple Sections
```
Upload Here/
├── section_1.csv
├── section_2.csv
└── section_3.csv

Run program 3 times, selecting different file each time.
```

### Scenario 4: Comparison Analysis
```
1. Upload multiple time periods
2. Analyze each separately
3. Compare integrated_summary_*.txt files
4. Track trends over time
```

---

## ❓ FAQ

### Q: Can I have spaces in file names?
**A:** Yes! The program handles spaces automatically.
```
✅ "tunnel data section A.csv"
✅ "Daily Report 2024-11-24.csv"
```

### Q: What if folder is empty?
**A:** The program will:
- Skip the empty "Upload Here" folder
- Search other locations
- Offer to generate sample data
- Let you enter custom path

### Q: Can I upload files from network drives?
**A:** Yes! Either:
- Copy to "Upload Here" first (recommended)
- Or select [0] and enter network path

### Q: Do I need to delete old files?
**A:** No, but you can:
- Keep frequently used files
- Move old files to `data/processed/`
- Or delete when no longer needed

### Q: Can I organize files in subfolders?
**A:** Currently, program searches top level only:
```
✅ Upload Here/file.csv         (Will find)
❌ Upload Here/subfolder/file.csv  (Won't find)
```

If you need subfolders, use [0] to enter custom path.

---

## 🎉 Benefits Summary

### Time Savings
- ❌ Before: Type long paths, risk typos
- ✅ Now: Drop file, type 1, done!

### Error Prevention
- ❌ Before: "File not found" errors common
- ✅ Now: Program finds files automatically

### User-Friendly
- ❌ Before: Need command-line skills
- ✅ Now: Drag, drop, click!

### Organization
- ❌ Before: Files scattered everywhere
- ✅ Now: One central upload location

---

## 📝 Example Session

```
> cd MTBM-Machine-Learning
> python auto_protocol_analyzer.py

================================================================================
🚀 AUTOMATIC MTBM PROTOCOL ANALYZER - INTERACTIVE MODE
================================================================================

================================================================================
INTERACTIVE FILE SELECTION
================================================================================

📂 Found 1 file(s) in 'Upload Here' folder

📁 Found 1 CSV file(s):

  [1] tunnel_section_A.csv
      Path: C:\Users\abdul\Desktop\ML for Tunneling\Upload Here
      Size: 245.3 KB

  [0] Enter custom path
  [G] Generate sample data
  [Q] Quit

Select file (number/0/G/Q): 1
✅ Selected: tunnel_section_A.csv

[continues with diameter selection, analysis...]
```

---

## 🎯 Quick Commands

**Open folder:**
```bash
# Windows - Double-click
Open Upload Folder.bat

# Or in terminal
cd "Upload Here"
dir
```

**Check what's in folder:**
```bash
# Windows
dir "Upload Here"

# Mac/Linux
ls -lh "Upload Here/"
```

**Move analyzed file:**
```bash
# After analysis
move "Upload Here\file.csv" "MTBM-Machine-Learning\data\processed\"
```

---

## ✅ Checklist

Before first use:
- [ ] "Upload Here" folder exists
- [ ] Can access the folder
- [ ] Have CSV files to analyze

For each analysis:
- [ ] CSV file in "Upload Here" folder
- [ ] Run program from MTBM-Machine-Learning directory
- [ ] Select file from list
- [ ] Choose pipe diameter
- [ ] Review results

---

## 🚀 You're Ready!

**The Simplest Workflow:**
1. Drop CSV in "Upload Here"
2. Run `python auto_protocol_analyzer.py`
3. Select file and diameter
4. Get results!

**That's it!** No complex paths, no typing errors, just simple drag-and-drop! 🎉

---

**Created:** November 2024
**Version:** 1.0
**Status:** ✅ Ready to use
