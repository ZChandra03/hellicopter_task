# Bayesian GRU Fitting Pipeline - File Index

## 📋 Overview
Complete toolkit for fitting a detuned Bayesian observer model to a trained GRU network, revealing what "internal model" the network learned during training.

---

## 🚀 Getting Started Files

### **QUICKSTART.md** (START HERE!)
**Purpose:** Concise guide to get you running in 5 minutes  
**Size:** 5.7 KB  
**Read if:** You want to start immediately  
**Contains:**
- Minimal setup instructions
- Three ways to run the pipeline
- Quick interpretation guide
- Common troubleshooting

---

## 📚 Documentation Files

### **README_fitting.md**
**Purpose:** Comprehensive user manual  
**Size:** 8.9 KB  
**Read if:** You want to understand all options and details  
**Contains:**
- Detailed usage instructions
- Configuration options
- Output file descriptions
- Advanced usage patterns
- Troubleshooting guide
- FAQ section

### **bayesian_gru_fitting_plan.md**
**Purpose:** Technical methodology document  
**Size:** 14 KB  
**Read if:** You want to understand the theory and implementation  
**Contains:**
- Five-phase implementation plan
- Mathematical background
- Likelihood derivation
- Optimization strategies
- Alternative approaches
- Expected outcomes analysis

---

## 🛠️ Executable Scripts

### **run_full_pipeline.py** ⭐ RECOMMENDED
**Purpose:** Master script that runs everything automatically  
**Size:** 3.0 KB  
**Run:** `python run_full_pipeline.py`  
**Does:**
1. Checks setup
2. Fits Bayesian parameters
3. Generates comprehensive report
**Use when:** First time running or want full automation

### **fit_bayesian_to_gru.py** ⭐ CORE SCRIPT
**Purpose:** Main fitting algorithm  
**Size:** 31 KB  
**Run:** `python fit_bayesian_to_gru.py`  
**Does:**
1. Loads trained GRU and runs inference
2. Collects predictions with probabilities
3. Fits σ_belief and bias via maximum likelihood
4. Generates comparison plots
**Use when:** You want just the fitting without checks/reports

### **check_setup.py**
**Purpose:** Pre-flight verification script  
**Size:** 5.3 KB  
**Run:** `python check_setup.py`  
**Does:**
- Verifies all files are in place
- Checks Python dependencies
- Lists available test variants
- Provides helpful error messages
**Use when:** Debugging setup issues or first-time setup

### **generate_report.py**
**Purpose:** Creates human-readable analysis report  
**Size:** 15 KB  
**Run:** `python generate_report.py`  
**Does:**
- Interprets fitted parameters
- Explains what GRU learned
- Provides recommendations
- Generates text summary
**Use when:** After fitting, to get interpretations

---

## 📂 Expected Directory Structure

```
your_project/
│
├── models/
│   └── trained_gru/              # Your trained model
│       ├── final.pt               # ← Required
│       └── hp.json                # ← Required
│
├── variants/
│   ├── sigma_1/                   # Test configs
│   │   └── testConfig_*.csv       # ← Required
│   └── sigma_2/                   # More test configs
│       └── testConfig_*.csv
│
├── rnn_models.py                  # ← Required (GRU definition)
│
├── [Copy these scripts from outputs/]
├── run_full_pipeline.py           # Master script
├── fit_bayesian_to_gru.py         # Core fitting
├── check_setup.py                 # Setup checker
├── generate_report.py             # Report generator
│
└── results/
    └── bayesian_fitting/          # Created automatically
        ├── gru_predictions.csv
        ├── fitted_parameters.json
        ├── fitting_report.txt
        ├── comparison_*.png
        └── ...
```

---

## 🎯 Recommended Workflow

### First Time Use
```bash
# 1. Copy scripts to your project directory
cp /path/to/outputs/*.py /path/to/your/project/
cp /path/to/outputs/*.md /path/to/your/project/docs/  # optional

# 2. Verify setup
python check_setup.py

# 3. Run full pipeline
python run_full_pipeline.py

# 4. Read the report
cat results/bayesian_fitting/fitting_report.txt
```

### Subsequent Runs
```bash
# If you just want to refit with different settings
python fit_bayesian_to_gru.py

# Then regenerate report
python generate_report.py
```

### Quick Checks
```bash
# Just verify everything is in place
python check_setup.py
```

---

## 📊 Output Files (Created Automatically)

After running the pipeline, you'll find these in `results/bayesian_fitting/`:

### Data Files
- **gru_predictions.csv** - All GRU predictions (choices + probabilities)
- **fitted_parameters.json** - Fitted σ_belief and bias with metadata
- **comparison_summary.json** - Summary statistics (accuracies, kappa)
- **detailed_comparison.csv** - Trial-by-trial GRU vs Bayesian comparison

### Plots
- **comparison_report_head.png** - Report accuracy: GRU vs Fitted vs Optimal
- **comparison_hazard_head.png** - Hazard accuracy: GRU vs Fitted vs Optimal
- **probability_distributions.png** - Choice probability histograms
- **accuracy_deltas.png** - Performance differences (GRU - Fitted)

### Reports
- **fitting_report.txt** - Comprehensive human-readable analysis

---

## 🔧 Configuration

Key variables in `fit_bayesian_to_gru.py`:

```python
MODEL_NAME = "trained_gru"     # Folder name in models/
VARIANT_NAME = "sigma_2"       # Which test set (sigma_1, sigma_2, beta_1p0, etc.)
N_TEST_CFGS = 20              # Number of test configs to use
DEVICE = "cuda"/"cpu"          # Automatically detected
```

---

## 📖 Which File Should I Read?

**"I just want to run it!"**  
→ Start with: **QUICKSTART.md**

**"I want to understand all the options"**  
→ Read: **README_fitting.md**

**"I want to understand the methodology"**  
→ Read: **bayesian_gru_fitting_plan.md**

**"Something isn't working"**  
→ Run: `python check_setup.py`  
→ Then read: **QUICKSTART.md** troubleshooting section

**"I ran it, now what do the results mean?"**  
→ Check: `results/bayesian_fitting/fitting_report.txt`  
→ Then read: **README_fitting.md** "Interpreting Results" section

---

## 🎓 Key Concepts

### What This Pipeline Does
1. **Collects GRU behavior** - Runs your trained GRU on test data
2. **Fits Bayesian parameters** - Finds σ and bias that best match GRU choices
3. **Compares strategies** - Shows GRU vs optimal Bayesian vs fitted Bayesian
4. **Interprets results** - Tells you what "internal model" the GRU learned

### Parameters Being Fitted

**σ_belief (sigma belief)**
- How much noise the observer thinks there is
- Reveals if GRU is over/underconfident about evidence reliability

**bias (prior bias)**
- Prior expectation about volatility
- Positive = expect more switches
- Negative = expect fewer switches

### Why This Matters
Understanding what internal model your GRU learned helps you:
- Verify it learned the intended strategy
- Debug suboptimal performance
- Compare different architectures
- Understand generalization properties

---

## 📝 Dependencies

Required Python packages:
```bash
pip install numpy pandas torch scipy matplotlib scikit-learn
```

All standard packages, no exotic dependencies!

---

## ⏱️ Expected Runtime

| Step | Time | Notes |
|------|------|-------|
| Setup check | <1 sec | Very fast |
| GRU inference | 1-2 min | Depends on # configs |
| Parameter fitting | 10-20 min | Most time here |
| Report generation | <1 sec | Very fast |
| **Total** | **~15-25 min** | For 20 test configs |

Reduce `N_TEST_CFGS` if you want faster results during development.

---

## 🐛 Troubleshooting Quick Reference

| Problem | Solution | File |
|---------|----------|------|
| Don't know where to start | Read QUICKSTART.md | QUICKSTART.md |
| Setup issues | Run check_setup.py | check_setup.py |
| Need more detail | Read README_fitting.md | README_fitting.md |
| Want methodology | Read plan document | bayesian_gru_fitting_plan.md |
| Fitting failed | Check error, adjust bounds | fit_bayesian_to_gru.py |
| Poor fit quality | Try extended models | README_fitting.md |
| Interpret results | Read fitting report | results/.../fitting_report.txt |

---

## 🎯 Success Criteria

You'll know it worked when:
1. ✓ No errors during execution
2. ✓ Files appear in `results/bayesian_fitting/`
3. ✓ Plots show all three curves (GRU, Fitted, Optimal)
4. ✓ fitting_report.txt contains interpretations
5. ✓ Fitted accuracy ≈ GRU accuracy (within ~5%)

---

## 🚨 When Things Go Wrong

### Fitting Quality Issues
If fitted Bayesian ≠ GRU behavior:
1. **This is actually informative!** Means GRU isn't doing pure Bayesian inference
2. Consider extended models (see README_fitting.md)
3. Analyze disagreement trials
4. Check representational similarity

### Parameter Interpretation
- **σ way off:** GRU may have learned adaptive noise estimation
- **Large bias:** Check training data balance, may be adaptive
- **Low kappa:** GRU uses different strategy, not necessarily bad!

---

## 📚 Additional Resources

- **Plan Document:** Full methodology and theory
- **README:** Comprehensive usage guide
- **QUICKSTART:** Get running immediately
- **Output Report:** Automatic interpretation of YOUR results

---

## 🎓 Learning Path

1. **Beginner:** QUICKSTART.md → run_full_pipeline.py → read output report
2. **Intermediate:** README_fitting.md → adjust configs → compare results
3. **Advanced:** bayesian_gru_fitting_plan.md → extend models → custom analysis

---

## ✅ Final Checklist Before Running

- [ ] Trained GRU checkpoint exists (final.pt)
- [ ] Hyperparameters file exists (hp.json)
- [ ] Test configs available (testConfig_*.csv)
- [ ] rnn_models.py in directory
- [ ] All scripts copied to project
- [ ] Dependencies installed
- [ ] Ran check_setup.py

If all checked, run: `python run_full_pipeline.py`

---

## 📞 Need Help?

1. Run `python check_setup.py` for diagnostics
2. Check troubleshooting in QUICKSTART.md
3. Read relevant section in README_fitting.md
4. Review plan document for methodology questions

---

**Ready to discover what your GRU learned?**  
Start with: `python run_full_pipeline.py`
