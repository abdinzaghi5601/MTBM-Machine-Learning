# MTBM ML Framework - Navigation Guide

## 🗺️ Quick Navigation Map

```
┌─────────────────────────────────────────────────────────────┐
│                    ML for Tunneling                         │
│                  (Main Project Folder)                      │
└─────────────────────────────────────────────────────────────┘
            │
            ├── 📄 START HERE
            │   ├── PROJECT_COMPLETE.md ⭐ ← Overview of everything built
            │   ├── AUTO_ANALYZER_GUIDE.md ⭐ ← Easiest way to use system
            │   ├── QUICK_START.md ← General getting started
            │   └── README.md ← Project overview
            │
            ├── 📄 FEATURE-SPECIFIC GUIDES
            │   ├── COMPLETE_ANALYSIS_GUIDE.md ← Full integrated analysis
            │   ├── ANOMALY_DETECTION_QUICKSTART.md ← ML anomaly detection
            │   └── MULTI_PROTOCOL_QUICKSTART.md ← Multi-protocol support
            │
            ├── 📁 MTBM-Machine-Learning (CODE REPOSITORY) ⭐
            │   │
            │   ├── 🐍 MAIN PROGRAMS (Use These!)
            │   │   ├── auto_protocol_analyzer.py ⭐⭐⭐ ← AUTOMATIC (Best!)
            │   │   ├── full_analysis.py ⭐⭐ ← Complete suite
            │   │   ├── analyze_protocol.py ⭐ ← Standard analysis
            │   │   └── analyze_with_anomalies.py ⭐ ← Anomaly detection
            │   │
            │   ├── 🔧 CORE MODULES (Used by main programs)
            │   │   ├── protocol_configs.py ← Protocol definitions
            │   │   ├── deviation_anomaly_detector.py ← ML algorithms
            │   │   ├── pipe_bore_tolerances.py ← Tolerance standards
            │   │   └── mtbm_comprehensive_plotting.py ← Visualization
            │   │
            │   ├── 📁 data/ ← YOUR DATA GOES HERE
            │   │   ├── raw/ ← Original CSV files
            │   │   └── processed/ ← Processed data
            │   │
            │   ├── 📁 outputs/ ← ALL RESULTS SAVED HERE
            │   │   ├── AVN800/ ← Protocol-specific results
            │   │   ├── AVN1200/
            │   │   ├── AVN2400/
            │   │   └── AVN3000/
            │   │       ├── plots/ ← Standard analysis
            │   │       ├── auto_analysis/ ← Automatic analyzer results
            │   │       │   ├── anomaly_detection/
            │   │       │   ├── tolerance_compliance/
            │   │       │   ├── integrated_summary_*.txt ⭐ ← READ THIS FIRST
            │   │       │   └── complete_results_*.csv
            │   │       └── complete_analysis/ ← Full analysis results
            │   │
            │   ├── 📁 docs/ ← DETAILED TECHNICAL GUIDES
            │   │   ├── MULTI_PROTOCOL_GUIDE.md (25 pages)
            │   │   ├── PLOT_INTERPRETATION_GUIDE.md (23 pages)
            │   │   ├── ANOMALY_DETECTION_GUIDE.md (30 pages)
            │   │   └── CODE_STRUCTURE_GUIDE.md (19 pages)
            │   │
            │   ├── 📁 legacy/ ← Old code (ignore)
            │   ├── 📁 tools/ ← Utility scripts
            │   ├── 📁 sql/ ← Database integration (optional)
            │   │
            │   └── 📄 requirements.txt ← Python dependencies
            │
            └── 📁 docs/ ← Same detailed guides (convenience copy)

```

---

## 🎯 "I Want To..."

### → Analyze My Data (EASIEST)
**What**: Run complete automatic analysis
**File**: `MTBM-Machine-Learning/auto_protocol_analyzer.py`
**Guide**: `AUTO_ANALYZER_GUIDE.md`
**Command**:
```bash
cd MTBM-Machine-Learning
python auto_protocol_analyzer.py --data your_data.csv --diameter 800
```

---

### → Understand What Was Built
**What**: Project overview and capabilities
**File**: `PROJECT_COMPLETE.md` ⭐
**Contains**:
- All system capabilities
- All files created
- All documentation
- Quick start examples

---

### → Learn How to Use Each Feature

| Feature | Quick Guide | Detailed Guide |
|---------|-------------|----------------|
| **Automatic Analysis** | `AUTO_ANALYZER_GUIDE.md` | - |
| **Complete Analysis** | `COMPLETE_ANALYSIS_GUIDE.md` | - |
| **Multi-Protocol** | `MULTI_PROTOCOL_QUICKSTART.md` | `docs/MULTI_PROTOCOL_GUIDE.md` |
| **Anomaly Detection** | `ANOMALY_DETECTION_QUICKSTART.md` | `docs/ANOMALY_DETECTION_GUIDE.md` |
| **Plot Interpretation** | - | `docs/PLOT_INTERPRETATION_GUIDE.md` |
| **Code Reference** | - | `docs/CODE_STRUCTURE_GUIDE.md` |

---

### → Find My Analysis Results
**Location**: `MTBM-Machine-Learning/outputs/[PROTOCOL]/`

**Example**: After running automatic analyzer on AVN2400 data:
```
MTBM-Machine-Learning/outputs/AVN2400/auto_analysis/
├── integrated_summary_YYYYMMDD_HHMMSS.txt ⭐ READ THIS FIRST
├── complete_results_YYYYMMDD_HHMMSS.csv
├── anomaly_detection/
│   ├── anomalies_*.png ← Visualizations
│   ├── anomaly_report_*.txt
│   └── anomaly_results_*.csv
└── tolerance_compliance/
    ├── compliance_*.png
    ├── compliance_report_*.txt
    └── compliance_results_*.csv
```

---

### → Understand the Plots
**Guide**: `docs/PLOT_INTERPRETATION_GUIDE.md` (23 pages)
**Contains**:
- What each plot shows
- What is "good" vs "alarming"
- Decision-making workflows
- Quality thresholds
- Real-world examples

---

### → Learn About ML Algorithms
**Quick**: `ANOMALY_DETECTION_QUICKSTART.md`
**Detailed**: `docs/ANOMALY_DETECTION_GUIDE.md` (30 pages)
**Contains**:
- 5 ML algorithm explanations
- How ensemble voting works
- Sensitivity settings
- Interpreting anomaly scores
- Feature engineering details

---

### → Work with Different Protocols
**Quick**: `MULTI_PROTOCOL_QUICKSTART.md`
**Detailed**: `docs/MULTI_PROTOCOL_GUIDE.md` (25 pages)
**Contains**:
- AVN 800/1200/2400/3000 differences
- Protocol-specific parameters
- Custom thresholds
- When to use which protocol

---

### → Modify the Code
**Guide**: `docs/CODE_STRUCTURE_GUIDE.md` (19 pages)
**Contains**:
- Code architecture
- Key classes and functions
- How to customize thresholds
- Adding new features
- Integration examples

---

## 📋 File Type Reference

### Python Programs You Run

| File | Purpose | When to Use |
|------|---------|-------------|
| `auto_protocol_analyzer.py` | 🌟 Automatic everything | Most of the time! |
| `full_analysis.py` | Complete suite | When you know protocol |
| `analyze_protocol.py` | Standard analysis only | Quick checks |
| `analyze_with_anomalies.py` | Anomaly detection only | Focus on ML |

### Python Modules (Don't Run Directly)

| File | Purpose |
|------|---------|
| `protocol_configs.py` | Protocol definitions & thresholds |
| `deviation_anomaly_detector.py` | ML algorithms (5 models) |
| `pipe_bore_tolerances.py` | Industry tolerance standards |
| `mtbm_comprehensive_plotting.py` | Visualization functions |

### Documentation Files

| File | Type | Pages | Content |
|------|------|-------|---------|
| `PROJECT_COMPLETE.md` | Overview | 10 | Everything built, how to use |
| `AUTO_ANALYZER_GUIDE.md` | User Guide | 12 | Automatic analyzer |
| `COMPLETE_ANALYSIS_GUIDE.md` | User Guide | 8 | Complete analysis |
| `ANOMALY_DETECTION_QUICKSTART.md` | Quick Ref | 6 | ML quick start |
| `MULTI_PROTOCOL_QUICKSTART.md` | Quick Ref | 5 | Protocol quick start |
| `QUICK_START.md` | Getting Started | 6 | General overview |
| `README.md` | Overview | 7 | Project introduction |
| `docs/MULTI_PROTOCOL_GUIDE.md` | Technical | 25 | Complete protocol guide |
| `docs/PLOT_INTERPRETATION_GUIDE.md` | Technical | 23 | Understanding results |
| `docs/ANOMALY_DETECTION_GUIDE.md` | Technical | 30 | ML deep dive |
| `docs/CODE_STRUCTURE_GUIDE.md` | Technical | 19 | Code reference |

---

## 🚀 Common Workflows

### Workflow 1: First Time User
```
1. Read: PROJECT_COMPLETE.md
2. Read: AUTO_ANALYZER_GUIDE.md
3. Run: python auto_protocol_analyzer.py --generate-sample --diameter 800
4. Check: outputs/AVN*/auto_analysis/integrated_summary_*.txt
5. Review: Anomaly and tolerance plots
6. Try with your own data!
```

### Workflow 2: Daily Monitoring
```
1. Run: python auto_protocol_analyzer.py --data daily.csv --diameter 800
2. Check: integrated_summary_*.txt for critical findings
3. If issues: Review anomaly_report_*.txt
4. If issues: Check compliance_report_*.txt
5. Save reports for records
```

### Workflow 3: Problem Investigation
```
1. Run with high sensitivity:
   python auto_protocol_analyzer.py --data problem.csv --diameter 800 --sensitivity high
2. Read: anomaly_report_*.txt
3. Open: anomalies_*.png to visualize
4. Review: complete_results_*.csv for specific chainages
5. Cross-reference with site logs
```

### Workflow 4: Understanding Results
```
1. Open: integrated_summary_*.txt
2. If unclear: Read docs/PLOT_INTERPRETATION_GUIDE.md
3. Check: Quality thresholds in docs/MULTI_PROTOCOL_GUIDE.md
4. Compare: Your results vs. quality standards
5. Make decisions based on severity and compliance
```

---

## 📊 Decision Tree: Which File Do I Need?

```
❓ What do you want to do?
│
├─ 🎯 Analyze data
│  │
│  ├─ Don't know protocol → auto_protocol_analyzer.py ⭐
│  ├─ Know protocol → full_analysis.py
│  └─ Just anomalies → analyze_with_anomalies.py
│
├─ 📖 Learn how to use
│  │
│  ├─ Complete overview → PROJECT_COMPLETE.md ⭐
│  ├─ Quick start → AUTO_ANALYZER_GUIDE.md
│  ├─ Specific feature → See feature-specific guides
│  └─ Deep technical → docs/*.md
│
├─ 📊 Understand results
│  │
│  ├─ What do plots mean? → docs/PLOT_INTERPRETATION_GUIDE.md
│  ├─ What is good/bad? → COMPLETE_ANALYSIS_GUIDE.md
│  └─ Protocol thresholds? → docs/MULTI_PROTOCOL_GUIDE.md
│
├─ 🔧 Modify code
│  │
│  ├─ How does it work? → docs/CODE_STRUCTURE_GUIDE.md
│  ├─ Change thresholds? → protocol_configs.py + guide
│  └─ Add features? → docs/CODE_STRUCTURE_GUIDE.md
│
└─ ❓ General questions
   │
   ├─ What was built? → PROJECT_COMPLETE.md
   ├─ How to start? → QUICK_START.md
   └─ Project overview? → README.md
```

---

## 🎯 Quick Reference Cards

### Card 1: Most Important Files

| Priority | File | Purpose |
|----------|------|---------|
| ⭐⭐⭐ | `auto_protocol_analyzer.py` | Run this to analyze data |
| ⭐⭐⭐ | `AUTO_ANALYZER_GUIDE.md` | Learn how to use it |
| ⭐⭐⭐ | `PROJECT_COMPLETE.md` | Understand what was built |
| ⭐⭐ | `integrated_summary_*.txt` | Your analysis results |
| ⭐⭐ | `complete_results_*.csv` | Your data with flags |

### Card 2: Most Useful Commands

```bash
# 1. Automatic complete analysis
python auto_protocol_analyzer.py --data file.csv --diameter 800

# 2. Generate test data
python auto_protocol_analyzer.py --generate-sample --diameter 800

# 3. High sensitivity investigation
python auto_protocol_analyzer.py --data file.csv --diameter 800 --sensitivity high

# 4. Quick analysis (skip ML)
python auto_protocol_analyzer.py --data file.csv --diameter 800 --skip-anomaly

# 5. Check latest results
cat outputs/AVN*/auto_analysis/integrated_summary_*.txt
```

### Card 3: Where Are My Files?

```
Your CSV data → MTBM-Machine-Learning/data/raw/
Analysis results → MTBM-Machine-Learning/outputs/[PROTOCOL]/auto_analysis/
Plots/visualizations → outputs/[PROTOCOL]/auto_analysis/anomaly_detection/
                       outputs/[PROTOCOL]/auto_analysis/tolerance_compliance/
Summary report → integrated_summary_*.txt (in auto_analysis/)
Full results → complete_results_*.csv (in auto_analysis/)
```

---

## 📞 Getting Help

### For Each Feature:

**Automatic Analyzer**
- Quick: `AUTO_ANALYZER_GUIDE.md`
- Issues: Check troubleshooting section in guide

**Anomaly Detection**
- Quick: `ANOMALY_DETECTION_QUICKSTART.md`
- Deep: `docs/ANOMALY_DETECTION_GUIDE.md`

**Multi-Protocol**
- Quick: `MULTI_PROTOCOL_QUICKSTART.md`
- Deep: `docs/MULTI_PROTOCOL_GUIDE.md`

**Understanding Plots**
- Guide: `docs/PLOT_INTERPRETATION_GUIDE.md`
- Examples: See "Good vs Alarming" sections

**Code Issues**
- Reference: `docs/CODE_STRUCTURE_GUIDE.md`
- Dependencies: Check `requirements.txt`

---

## ✅ Checklist for New Users

**Before First Use:**
- [ ] Read `PROJECT_COMPLETE.md`
- [ ] Read `AUTO_ANALYZER_GUIDE.md`
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Test with sample data: `--generate-sample`

**For Each Analysis:**
- [ ] Know your pipe diameter
- [ ] Have CSV data ready
- [ ] Run automatic analyzer
- [ ] Check integrated summary first
- [ ] Review visualizations
- [ ] Save results for records

**Understanding Results:**
- [ ] Read integrated summary
- [ ] Check critical findings section
- [ ] Review anomaly percentage
- [ ] Check tolerance compliance
- [ ] Compare vs quality standards
- [ ] Consult `docs/PLOT_INTERPRETATION_GUIDE.md` if needed

---

## 🎓 Learning Path

**Beginner (Day 1)**
1. Read: `PROJECT_COMPLETE.md`
2. Read: `QUICK_START.md`
3. Run: Sample data test
4. Review: Generated outputs

**Intermediate (Week 1)**
1. Read: `AUTO_ANALYZER_GUIDE.md`
2. Read: `COMPLETE_ANALYSIS_GUIDE.md`
3. Test: With your real data
4. Learn: `docs/PLOT_INTERPRETATION_GUIDE.md`

**Advanced (Month 1)**
1. Read: `docs/ANOMALY_DETECTION_GUIDE.md`
2. Read: `docs/MULTI_PROTOCOL_GUIDE.md`
3. Read: `docs/CODE_STRUCTURE_GUIDE.md`
4. Customize: Thresholds and features

---

## 🗺️ You Are Here

```
START
  ↓
PROJECT_COMPLETE.md ← Overview of everything ⭐
  ↓
AUTO_ANALYZER_GUIDE.md ← How to use the system ⭐
  ↓
Run: auto_protocol_analyzer.py ← Analyze your data
  ↓
Check: integrated_summary_*.txt ← Review results
  ↓
If issues → Read specific guides for help
  ↓
PRODUCTION USE
```

---

**Total Files**:
- 📄 Documentation: 12 files (75+ pages)
- 🐍 Python Code: 7 main programs + modules
- 📁 Organized outputs: Protocol-specific folders

**You Have Everything You Need!** 🚀

**Next Step**: Open `AUTO_ANALYZER_GUIDE.md` and start analyzing! ⭐
