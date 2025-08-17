# Drug Efficacy Statistical Analyzer

![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![SciPy](https://img.shields.io/badge/SciPy-Statistical%20Analysis-8CAAE6?style=for-the-badge&logo=scipy)

A Streamlit-powered statistical tool for analyzing simulated clinical trial data to evaluate drug treatment efficacy.



## 🚀 Features

- **Interactive Clinical Trial Simulation**
  - Adjustable parameters: sample size, effect sizes, variance
  - Real-time statistical analysis updates

- **Comprehensive Statistical Testing**
  - Automated test selection (parametric/non-parametric)
  - Paired t-tests vs Wilcoxon signed-rank for within-group analysis
  - Independent t-tests vs Mann-Whitney U for between-group comparison

- **Diagnostic Checks**
  - Shapiro-Wilk normality tests
  - Levene's variance equality tests
  - Effect size calculations (Cohen's d)

- **Visual Analytics**
  - Interactive distribution plots
  - Comparative boxplots
  - Results summary dashboard

## 📊 Sample Output
- Normality check results (Shapiro–Wilk & KS test)

- Within-group analysis (significant or not)

- Between-group comparison (which drug is more effective)

- Effect sizes for practical significance

## 📝 Usage

- Adjust simulation parameters in the sidebar

- View simulated data distributions

- Analyze within-group drug efficacy

- Compare improvements between treatments

- Interpret statistical conclusions

- Download analysis report





📚 Documentation

- **Statistical Methods Used:**

  - Paired t-test

  - Wilcoxon signed-rank test

  - Cohen's d effect size

🤝 Contributing

Contributions are welcome! Please open an issue or submit a PR for any:

- Bug fixes

- Additional statistical tests

- Enhanced visualization options

- Documentation improvements

📜 License

MIT License - See LICENSE for details.

Developed by Yougharta Reghis

## 🛠️ Installation


```bash
1. Clone the repository:
git clone  https://github.com/YoughartaR123/drug_performance_analytics.git
cd pharmastat

2. Create and activate a virtual environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. Install dependencies:
pip install -r requirements.txt

4. Run the Streamlit app:
streamlit run drug_affect.py

