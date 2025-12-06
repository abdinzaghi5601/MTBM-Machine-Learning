# Essential Files Guide

## 🎯 What You Actually Need

This guide shows you the essential files you need and what you can ignore.

---

## ⭐ MUST READ (Start Here)

| File | Purpose | Read Time |
|------|---------|-----------|
| **START_HERE.md** | Your entry point - read this first! | 5 min |
| **INTERACTIVE_MODE_GUIDE.md** | How to run programs (no command-line needed) | 10 min |
| **AUTO_ANALYZER_GUIDE.md** | Complete usage guide for automatic analyzer | 15 min |

**Total**: 30 minutes to be fully operational ✅

---

## 🐍 Python Programs (What to Run)

### Main Programs (In Order of Importance)

| File | Purpose | Usage |
|------|---------|-------|
| **auto_protocol_analyzer.py** ⭐⭐⭐ | Automatic complete analysis | `python auto_protocol_analyzer.py` |
| full_analysis.py | Complete suite (manual protocol) | `python full_analysis.py --protocol AVN2400 --data file.csv --diameter 800` |
| analyze_protocol.py | Standard analysis only | `python analyze_protocol.py --protocol AVN2400 --data file.csv` |
| analyze_with_anomalies.py | Anomaly detection only | `python analyze_with_anomalies.py --protocol AVN2400 --data file.csv` |

### Core Modules (Used Automatically - Don't Run Directly)

| File | Purpose |
|------|---------|
| protocol_configs.py | Protocol definitions |
| deviation_anomaly_detector.py | ML algorithms |
| pipe_bore_tolerances.py | Tolerance standards |
| mtbm_comprehensive_plotting.py | Visualization |
| interactive_helpers.py | Interactive mode functions |

### Configuration

| File | Purpose |
|------|---------|
| requirements.txt | Python dependencies |

---

## 📚 Documentation (By Category)

### Essential Documentation

| File | Purpose |
|------|---------|
| START_HERE.md | Entry point |
| INTERACTIVE_MODE_GUIDE.md | How to use interactive mode |
| AUTO_ANALYZER_GUIDE.md | Automatic analyzer guide |
| PROJECT_COMPLETE.md | Complete system overview |

### Feature Guides

| File | Purpose |
|------|---------|
| COMPLETE_ANALYSIS_GUIDE.md | Full analysis system |
| ANOMALY_DETECTION_QUICKSTART.md | ML anomaly detection |
| MULTI_PROTOCOL_QUICKSTART.md | Multi-protocol support |

### Reference

| File | Purpose |
|------|---------|
| NAVIGATION_GUIDE.md | Find any file quickly |
| FINAL_STATUS.md | Project completion status |
| QUICK_START.md | General overview |
| README.md | Project introduction |

### Archive (Session History)

| File | Purpose |
|------|---------|
| SESSION_SUMMARY.md | What was built |
| WHATS_NEW.md | Changelog |

---

## 📁 Directories

### Essential Directories

```
MTBM-Machine-Learning/
├── data/
│   ├── raw/           ← PUT YOUR CSV FILES HERE
│   └── processed/
│
├── outputs/           ← ALL RESULTS GO HERE
│   ├── AVN800/
│   ├── AVN1200/
│   ├── AVN2400/
│   └── AVN3000/
│
├── docs/             ← Technical deep-dive guides
│   ├── MULTI_PROTOCOL_GUIDE.md
│   ├── PLOT_INTERPRETATION_GUIDE.md
│   ├── ANOMALY_DETECTION_GUIDE.md
│   └── CODE_STRUCTURE_GUIDE.md
│
└── archive/          ← OLD FILES (CAN IGNORE)
    ├── old_scripts/
    └── old_docs/
```

### What You Can Ignore

- `archive/` - Old development files
- `legacy/` - Legacy code
- `tools/` - Utility scripts
- `sql/` - Database integration (optional)
- `core/` - Internal modules
- `dashboards/` - Dashboard prototypes (optional)

---

## 🎯 Quick Start Flow

### 1. Installation
```bash
cd MTBM-Machine-Learning
pip install -r requirements.txt
```

### 2. Put Your Data
Put CSV files in `data/raw/` folder

### 3. Run Analysis (Interactive Mode)
```bash
python auto_protocol_analyzer.py
```
Then answer the questions!

### 4. Check Results
```
outputs/AVN[protocol]/auto_analysis/integrated_summary_*.txt
```

---

## 🗑️ What Was Cleaned Up

### Moved to `archive/old_scripts/`
- FINAL_REORGANIZATION.py
- create_graphs_direct.py
- find_graph_location.py
- generate_mtbm_graphs.py
- load_protocol_pdf.py
- load_real_data.py
- make_predictions.py
- plot_real_mtbm_data.py
- quickstart_demo.py
- reorganize_repository.py
- simple_test.py
- test_graph.py
- test_new_structure.py
- train_with_real_data.py
- unified_mtbm_ml_framework.py
- avn2400_advanced_measurement_ml.py

### Moved to `archive/old_docs/`
- COMPLETE_FILE_DOCUMENTATION.md
- COMPREHENSIVE_ML_FRAMEWORK_SUMMARY.md
- DOCUMENTATION_STATUS.md
- GITHUB_PUSH_GUIDE.md
- NEW_REPOSITORY_STRUCTURE.md
- PLOTTING_GUIDE.md
- PRACTICAL_GUIDE.md
- PROTOCOL_PDF_GUIDE.md
- REAL_DATA_IMPLEMENTATION.md
- REORGANIZATION_COMPLETE.md
- REPOSITORY_REORGANIZATION_PLAN.md
- WHERE_ARE_MY_FILES.md

**These files are still available if needed, just archived for cleaner structure!**

---

## ✅ What Remains (Clean & Essential)

### Python Programs: 9 files
- 4 main programs (run these)
- 5 core modules (used automatically)

### Documentation: 13 files
- 4 essential guides
- 3 feature guides
- 4 reference docs
- 2 archive/history

### Plus:
- requirements.txt
- README.md

**Total: ~25 essential files** (down from 50+)

---

## 🎯 Daily Usage

### 90% of the Time:
```bash
python auto_protocol_analyzer.py
```

### Check Results:
```bash
cat outputs/AVN*/auto_analysis/integrated_summary_*.txt
```

### Read Documentation:
1. START_HERE.md
2. INTERACTIVE_MODE_GUIDE.md
3. Done!

---

## 📊 File Importance Ranking

### Priority 1 (MUST HAVE) ⭐⭐⭐
- auto_protocol_analyzer.py
- interactive_helpers.py
- protocol_configs.py
- deviation_anomaly_detector.py
- pipe_bore_tolerances.py
- START_HERE.md
- INTERACTIVE_MODE_GUIDE.md
- requirements.txt

### Priority 2 (VERY USEFUL) ⭐⭐
- full_analysis.py
- analyze_protocol.py
- analyze_with_anomalies.py
- mtbm_comprehensive_plotting.py
- AUTO_ANALYZER_GUIDE.md
- PROJECT_COMPLETE.md
- COMPLETE_ANALYSIS_GUIDE.md

### Priority 3 (REFERENCE) ⭐
- docs/*.md (technical guides)
- NAVIGATION_GUIDE.md
- ANOMALY_DETECTION_QUICKSTART.md
- MULTI_PROTOCOL_QUICKSTART.md

### Priority 4 (ARCHIVE)
- archive/* (old files, can delete if needed)
- SESSION_SUMMARY.md
- FINAL_STATUS.md

---

## 💡 Recommendation

### For Beginners:
**Read only:**
1. START_HERE.md
2. INTERACTIVE_MODE_GUIDE.md

**Run:**
```bash
python auto_protocol_analyzer.py
```

**That's it!** Everything else is reference material for later.

### For Advanced Users:
Read the technical guides in `docs/` folder when you need to:
- Understand the ML algorithms
- Customize the code
- Interpret complex plots
- Work with multiple protocols

---

## 🎉 Summary

**Essential Files:** ~25
**Archive Files:** ~30 (moved to archive/)
**Total Cleanup:** Much cleaner and easier to navigate!

**You only need to focus on:**
- 1 program: `auto_protocol_analyzer.py`
- 2 guides: `START_HERE.md` + `INTERACTIVE_MODE_GUIDE.md`
- 1 command: `python auto_protocol_analyzer.py`

**Everything else is optional reference material!** ✅

---

**Created**: November 2024
**Version**: 1.0
**Status**: ✅ Cleaned and organized
