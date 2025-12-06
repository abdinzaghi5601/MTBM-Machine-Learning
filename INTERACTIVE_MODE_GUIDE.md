# Interactive Mode Guide

## 🎯 Super Easy - Just Run Without Arguments!

No need to remember complex command-line arguments anymore. Just run the program and answer simple questions!

---

## 🚀 Easiest Way to Use

### Step 1: Open Terminal/PowerShell

**Windows PowerShell:**
```powershell
cd "C:\Users\abdul\Desktop\ML for Tunneling\MTBM-Machine-Learning"
```

**Mac/Linux Terminal:**
```bash
cd ~/Desktop/ML\ for\ Tunneling/MTBM-Machine-Learning
```

### Step 2: Run the Program (No Arguments Needed!)

```bash
python auto_protocol_analyzer.py
```

**That's it!** The program will guide you through everything.

---

## 📋 What Happens in Interactive Mode

### 1. File Selection
The program shows you all CSV files it found:

```
================================================================================
INTERACTIVE FILE SELECTION
================================================================================

📁 Found 3 CSV file(s):

  [1] tunnel_data_section_A.csv
      Path: C:\...\data\raw
      Size: 245.3 KB

  [2] tunnel_data_section_B.csv
      Path: C:\...\data\raw
      Size: 189.7 KB

  [3] mtbm_comprehensive_data.csv
      Path: C:\...\ML for Tunneling
      Size: 512.8 KB

  [0] Enter custom path
  [G] Generate sample data
  [Q] Quit

Select file (number/0/G/Q):
```

**Your options:**
- Type `1`, `2`, `3`, etc. to select a file
- Type `0` to enter a custom file path
- Type `G` to generate sample data for testing
- Type `Q` to quit

### 2. Pipe Diameter Selection
Choose your pipe diameter:

```
================================================================================
PIPE DIAMETER SELECTION
================================================================================

Common pipe diameters:
  [1] 600 mm  (Tolerances: ±20mm vertical, ±25mm horizontal)
  [2] 800 mm  (Tolerances: ±25mm vertical, ±40mm horizontal)
  [3] 1000 mm (Tolerances: ±30mm vertical, ±100mm horizontal)
  [4] 1200 mm (Tolerances: ±30mm vertical, ±100mm horizontal)
  [5] 1400 mm (Tolerances: ±50mm vertical, ±200mm horizontal)
  [0] Enter custom diameter

Select diameter (1-5 or 0 for custom):
```

Just type the number! Or `0` for a custom diameter.

### 3. Anomaly Detection Sensitivity
Choose how sensitive the ML detection should be:

```
================================================================================
ANOMALY DETECTION SENSITIVITY
================================================================================

Options:
  [1] Low    - ~5% flagged (fewer false alarms)
  [2] Medium - ~10% flagged (balanced) - RECOMMENDED
  [3] High   - ~15% flagged (catch everything)

Select sensitivity (1/2/3, default 2):
```

Just press `Enter` for recommended medium sensitivity, or type `1`/`2`/`3`.

### 4. Analysis Options
Quick yes/no questions:

```
Skip anomaly detection? (y/N):
```

Type `n` or just press `Enter` to include anomaly detection.

```
Skip tolerance compliance? (y/N):
```

Type `n` or just press `Enter` to include tolerance checking.

### 5. Analysis Runs!
The program then runs the complete analysis automatically.

---

## 💡 Example Session

Here's what a complete session looks like:

```
> python auto_protocol_analyzer.py

================================================================================
🚀 AUTOMATIC MTBM PROTOCOL ANALYZER - INTERACTIVE MODE
================================================================================

================================================================================
INTERACTIVE FILE SELECTION
================================================================================

📁 Found 2 CSV file(s):

  [1] tunnel_data.csv
      Path: C:\Users\abdul\Desktop\ML for Tunneling\data\raw
      Size: 245.3 KB

  [2] test_data.csv
      Path: C:\Users\abdul\Desktop\ML for Tunneling\MTBM-Machine-Learning
      Size: 89.2 KB

  [0] Enter custom path
  [G] Generate sample data
  [Q] Quit

Select file (number/0/G/Q): 1
✅ Selected: tunnel_data.csv

================================================================================
PIPE DIAMETER SELECTION
================================================================================

Common pipe diameters:
  [1] 600 mm
  [2] 800 mm
  [3] 1000 mm
  [4] 1200 mm
  [5] 1400 mm
  [0] Enter custom diameter

Select diameter (1-5 or 0 for custom): 2
✅ Selected diameter: 800 mm

================================================================================
ANOMALY DETECTION SENSITIVITY
================================================================================

Options:
  [1] Low    - ~5% flagged
  [2] Medium - ~10% flagged - RECOMMENDED
  [3] High   - ~15% flagged

Select sensitivity (1/2/3, default 2):
✅ Sensitivity: medium

Skip anomaly detection? (y/N): n
Skip tolerance compliance? (y/N): n

================================================================================
STARTING ANALYSIS
================================================================================
[... analysis proceeds ...]
```

---

## 🎯 When to Use Interactive Mode

### Perfect For:
- ✅ First-time users
- ✅ When you forget command-line arguments
- ✅ Quick ad-hoc analyses
- ✅ Exploring different files
- ✅ Testing different settings

### Not Needed When:
- ❌ Batch processing (use command-line arguments)
- ❌ Automation scripts
- ❌ When you already know all parameters

---

## 🔧 Advanced: Combine Interactive and Command-Line

You can mix both approaches! Provide some arguments and the program will only ask for missing ones:

```bash
# Provide diameter, program asks for file
python auto_protocol_analyzer.py --diameter 800

# Provide data file, program asks for diameter
python auto_protocol_analyzer.py --data tunnel.csv

# Force interactive mode even with arguments
python auto_protocol_analyzer.py --interactive
python auto_protocol_analyzer.py -i
```

---

## 📋 Interactive vs Command-Line

### Interactive Mode (Easy!)
```bash
python auto_protocol_analyzer.py
# Answer 4-5 simple questions
```

**Pros:**
- ✅ No need to remember arguments
- ✅ Sees all available files
- ✅ Shows options with descriptions
- ✅ Validates input immediately
- ✅ Can't make syntax errors

**Cons:**
- ❌ Slower for repeated analyses
- ❌ Can't automate in scripts

### Command-Line Mode (Fast!)
```bash
python auto_protocol_analyzer.py --data tunnel.csv --diameter 800 --sensitivity medium
```

**Pros:**
- ✅ Very fast for repeated use
- ✅ Can use in scripts
- ✅ Can batch process
- ✅ Can pipe/redirect output

**Cons:**
- ❌ Must remember arguments
- ❌ Easy to make typos
- ❌ Must know file paths

---

## 💡 Tips for Interactive Mode

### 1. Finding Your CSV Files
The program automatically searches in:
- `data/raw/` folder
- `data/` folder
- Current directory
- Parent directory

**Pro tip:** Put your CSV files in `data/raw/` for easy access!

### 2. Custom File Paths
When entering custom paths:
- You can copy-paste the full path
- Quotes are automatically handled (OK with or without)
- Both forward slashes `/` and backslashes `\` work

**Examples that work:**
```
C:\Users\abdul\Desktop\tunnel.csv
C:/Users/abdul/Desktop/tunnel.csv
"C:\Users\abdul\Desktop\tunnel.csv"
'/Users/abdul/Desktop/tunnel.csv'
```

### 3. Default Values
Most questions have smart defaults:
- Diameter selection: No default (must choose)
- Sensitivity: Medium (just press Enter)
- Skip anomaly: No (just press Enter)
- Skip tolerance: No (just press Enter)

**Quick analysis:** Just select file, diameter, and press Enter for everything else!

### 4. Generating Test Data
Option `[G]` generates realistic sample data:
- Perfect for learning the system
- Tests all features
- Uses AVN2400 protocol by default
- 1,000 samples (you can specify different amount)

---

## 🚀 Quick Start Examples

### Example 1: First-Time User
```bash
python auto_protocol_analyzer.py
# Select [G] to generate sample data
# Select [2] for 800mm diameter
# Press Enter for all other options
# Done! Check the results
```

### Example 2: Analyze Your Data
```bash
python auto_protocol_analyzer.py
# Select your CSV file number
# Select your pipe diameter
# Press Enter for all other options
# Done!
```

### Example 3: High Sensitivity Investigation
```bash
python auto_protocol_analyzer.py
# Select your problem section CSV
# Select diameter
# Select [3] for high sensitivity
# Press Enter for skip questions
# Review detailed anomaly results
```

---

## 📊 After Analysis

Results are saved to:
```
outputs/AVN[protocol]/auto_analysis/
├── integrated_summary_*.txt ⭐ READ THIS FIRST!
├── complete_results_*.csv
├── anomaly_detection/
└── tolerance_compliance/
```

The program tells you exactly where to find your results!

---

## ❓ Troubleshooting Interactive Mode

### "No CSV files found"
**Solution:**
- Put CSV files in `data/raw/` folder
- Or select `[0]` and enter full path
- Or select `[G]` to generate test data

### "File not found" when entering custom path
**Solution:**
- Check the path is correct
- Use full absolute path
- Make sure file ends with `.csv`
- Try copy-pasting the path from File Explorer

### Want to cancel?
- Type `Q` in file selection
- Or press `Ctrl+C` anytime

### Made a mistake in selection?
- Press `Ctrl+C` and start over
- Or let it run and adjust next time

---

## 🎯 Summary

**Interactive Mode = Super Easy!**

1. Run: `python auto_protocol_analyzer.py`
2. Answer 4-5 simple questions
3. Get complete analysis!

**No command-line arguments to remember!**
**No complex syntax!**
**Just answer questions and go!**

---

**Perfect for beginners and quick analyses!** 🚀

**Created**: November 2024
**Version**: 1.0
**Status**: ✅ Ready to use
