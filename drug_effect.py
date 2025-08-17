import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import wilcoxon, mannwhitneyu, ttest_rel, ttest_ind, shapiro, levene
import math

# ------------------------------
# App Configuration
# ------------------------------
st.set_page_config(
    page_title="Drug Efficacy Analyzer",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ------------------------------
# Helper Functions
# ------------------------------
def ks_with_sample_params(x):
    """K-S test against normal distribution with sample parameters"""
    mu, sigma = np.mean(x), np.std(x, ddof=1)
    stat, p = stats.kstest(x, 'norm', args=(mu, sigma))
    return stat, p, mu, sigma


def cohens_d_paired(x_before, x_after):
    """Calculate Cohen's d for paired samples"""
    d = x_after - x_before
    mean_diff = np.mean(d)
    sd_diff = np.std(d, ddof=1)
    return mean_diff / sd_diff if sd_diff != 0 else np.nan


def cohens_d_independent(x, y):
    """Calculate Cohen's d for independent samples"""
    nx, ny = len(x), len(y)
    mx, my = np.mean(x), np.mean(y)
    sx2 = np.var(x, ddof=1)
    sy2 = np.var(y, ddof=1)
    pooled_sd = math.sqrt(((nx - 1) * sx2 + (ny - 1) * sy2) / (nx + ny - 2))
    return (mx - my) / pooled_sd if pooled_sd != 0 else np.nan


def plot_distribution(data, title, names, colors):
    """Plot distribution comparison"""
    fig, ax = plt.subplots(figsize=(10, 6))
    for d, name, color in zip(data, names, colors):
        sns.histplot(d, kde=True, ax=ax, label=name, color=color, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel('Reaction Time (ms)')
    ax.legend()
    st.pyplot(fig)


# ------------------------------
# Main App
# ------------------------------
st.title("💊 Drug Efficacy Statistical Analyzer")
st.markdown("""
This application simulates a clinical trial comparing two drugs (A and B) for reducing reaction times.
It performs comprehensive statistical analysis including:
- Within-group comparisons (pre vs post treatment)
- Between-group comparisons of improvements
- Normality and variance checks
- Effect size calculations
""")

with st.sidebar:
    st.header("Simulation Parameters")
    st.subheader("Experimental Design")
    n = st.slider("Participants per group", 10, 100, 30)
    alpha = st.slider("Significance level (α)", 0.01, 0.10, 0.05, 0.01)

    st.subheader("Drug A Characteristics")
    mu_a_before = st.slider("Drug A - Baseline mean", 200, 300, 260)
    mu_a_effect = st.slider("Drug A - Effect size", -30, 0, -10)

    st.subheader("Drug B Characteristics")
    mu_b_before = st.slider("Drug B - Baseline mean", 200, 300, 265)
    mu_b_effect = st.slider("Drug B - Effect size", -30, 0, -8)

    sigma = st.slider("Standard deviation", 5, 20, 10)
    seed = st.number_input("Random seed", 42)

    st.markdown("---")
    st.caption("Adjust parameters in real-time to see how they impact statistical results")

# ------------------------------
# Data Simulation
# ------------------------------
np.random.seed(seed)

drug_a_before = np.random.normal(mu_a_before, sigma, n)
drug_a_after = drug_a_before + np.random.normal(mu_a_effect, sigma, n)

drug_b_before = np.random.normal(mu_b_before, sigma, n)
drug_b_after = drug_b_before + np.random.normal(mu_b_effect, sigma, n)

# Create DataFrame for display
df = pd.DataFrame({
    'Drug': ['A'] * n + ['B'] * n,
    'Participant': list(range(1, n + 1)) * 2,
    'Before': np.concatenate([drug_a_before, drug_b_before]),
    'After': np.concatenate([drug_a_after, drug_b_after])
})

# ------------------------------
# Data Visualization
# ------------------------------
st.header("📊 Simulated Clinical Trial Data")
st.caption(f"Simulated data for {n} participants per drug group")

col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("Sample Data")
    st.dataframe(df.head(10), height=300)

with col2:
    st.subheader("Distribution Comparison")
    plot_distribution(
        [drug_a_before, drug_a_after, drug_b_before, drug_b_after],
        "Reaction Time Distributions",
        ['Drug A (Before)', 'Drug A (After)', 'Drug B (Before)', 'Drug B (After)'],
        ['skyblue', 'royalblue', 'lightcoral', 'firebrick']
    )


# ------------------------------
# Statistical Analysis Functions
# ------------------------------
def evaluate_within_group(name, before, after):
    """Perform within-group statistical analysis"""
    results = {}
    diffs = after - before

    # Normality checks
    sw_stat, sw_p = shapiro(diffs)
    ks_stat, ks_p, mu_d, sigma_d = ks_with_sample_params(diffs)

    # Test selection
    if sw_p > alpha:
        t_stat, p_val = ttest_rel(after, before)
        test_used = "Paired t-test"
        d_val = cohens_d_paired(before, after)
    else:
        try:
            w_stat, p_val = wilcoxon(after, before, alternative='two-sided')
            test_used = "Wilcoxon signed-rank"
            d_val = None
        except:
            p_val = np.nan
            test_used = "Wilcoxon failed"
            d_val = None

    # Format results
    results = {
        'mean_before': np.mean(before),
        'mean_after': np.mean(after),
        'mean_diff': np.mean(diffs),
        'sw_p': sw_p,
        'ks_p': ks_p,
        'test_used': test_used,
        'p_value': p_val,
        'cohen_d': d_val,
        'significant': p_val <= alpha if not np.isnan(p_val) else False
    }
    return results, diffs


def evaluate_between_groups(improve_a, improve_b):
    """Perform between-group statistical analysis"""
    results = {}

    # Normality checks
    sw_a = shapiro(improve_a)
    sw_b = shapiro(improve_b)
    ks_a = ks_with_sample_params(improve_a)
    ks_b = ks_with_sample_params(improve_b)

    # Variance equality
    levene_stat, levene_p = levene(improve_a, improve_b)

    # Test selection
    normal_a = sw_a[1] > alpha
    normal_b = sw_b[1] > alpha
    equal_var = levene_p > alpha

    if normal_a and normal_b and equal_var:
        t_stat, p_val = ttest_ind(improve_a, improve_b, equal_var=True)
        test_used = "Independent t-test (pooled)"
        d_val = cohens_d_independent(improve_a, improve_b)
    else:
        u_stat, p_val = mannwhitneyu(improve_a, improve_b, alternative='two-sided')
        test_used = "Mann-Whitney U"
        d_val = None

    # Format results
    results = {
        'mean_improve_a': np.mean(improve_a),
        'mean_improve_b': np.mean(improve_b),
        'sw_p_a': sw_a[1],
        'sw_p_b': sw_b[1],
        'levene_p': levene_p,
        'test_used': test_used,
        'p_value': p_val,
        'cohen_d': d_val,
        'significant': p_val <= alpha
    }
    return results


# ------------------------------
# Analysis Execution
# ------------------------------
st.header("📈 Statistical Analysis Results")

# Within-group analysis
st.subheader("Within-Group Efficacy Analysis")
st.markdown("""
**Hypothesis**:  
H₀: μ_before = μ_after (No significant change)  
H₁: μ_before ≠ μ_after (Significant change)
""")

col1, col2 = st.columns(2)
results_a, diffs_a = evaluate_within_group("Drug A", drug_a_before, drug_a_after)
results_b, diffs_b = evaluate_within_group("Drug B", drug_b_before, drug_b_after)

with col1:
    st.markdown("#### Drug A Results")
    st.metric("Mean Before", f"{results_a['mean_before']:.1f} ms")
    st.metric("Mean After", f"{results_a['mean_after']:.1f} ms",
              delta=f"{results_a['mean_diff']:.1f} ms")
    st.write(f"**Normality Check**:")
    st.write(f"- Shapiro-Wilk p-value: {results_a['sw_p']:.4f}")
    st.write(f"- Kolmogorov-Smirnov p-value: {results_a['ks_p']:.4f}")
    st.write(f"**Test Used**: {results_a['test_used']}")
    st.write(f"**p-value**: {results_a['p_value']:.4f}")
    if results_a['cohen_d'] is not None:
        st.write(f"**Cohen's d**: {results_a['cohen_d']:.3f}")
    st.write(f"**Conclusion**: {'Significant change' if results_a['significant'] else 'No significant change'}")

with col2:
    st.markdown("#### Drug B Results")
    st.metric("Mean Before", f"{results_b['mean_before']:.1f} ms")
    st.metric("Mean After", f"{results_b['mean_after']:.1f} ms",
              delta=f"{results_b['mean_diff']:.1f} ms")
    st.write(f"**Normality Check**:")
    st.write(f"- Shapiro-Wilk p-value: {results_b['sw_p']:.4f}")
    st.write(f"- Kolmogorov-Smirnov p-value: {results_b['ks_p']:.4f}")
    st.write(f"**Test Used**: {results_b['test_used']}")
    st.write(f"**p-value**: {results_b['p_value']:.4f}")
    if results_b['cohen_d'] is not None:
        st.write(f"**Cohen's d**: {results_b['cohen_d']:.3f}")
    st.write(f"**Conclusion**: {'Significant change' if results_b['significant'] else 'No significant change'}")

# Between-group analysis
st.subheader("Between-Group Comparison of Improvements")
st.markdown("""
**Hypothesis**:  
H₀: μ_improve_A = μ_improve_B (Drugs equally effective)  
H₁: μ_improve_A ≠ μ_improve_B (Drugs differ in effectiveness)  
*(Improvement = Before - After)*
""")

improve_a = drug_a_before - drug_a_after
improve_b = drug_b_before - drug_b_after
results_between = evaluate_between_groups(improve_a, improve_b)

col1, col2 = st.columns(2)
with col1:
    st.metric("Drug A Mean Improvement", f"{results_between['mean_improve_a']:.1f} ms")
    st.write(f"Shapiro-Wilk p-value: {results_between['sw_p_a']:.4f}")

with col2:
    st.metric("Drug B Mean Improvement", f"{results_between['mean_improve_b']:.1f} ms")
    st.write(f"Shapiro-Wilk p-value: {results_between['sw_p_b']:.4f}")

st.write(f"**Levene's Test (variance equality) p-value**: {results_between['levene_p']:.4f}")
st.write(f"**Test Used**: {results_between['test_used']}")
st.write(f"**p-value**: {results_between['p_value']:.4f}")
if results_between['cohen_d'] is not None:
    st.write(f"**Cohen's d**: {results_between['cohen_d']:.3f}")
st.write(
    f"**Conclusion**: {'Significant difference' if results_between['significant'] else 'No significant difference'}")

# Improvement plot
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(x=['Drug A'] * n + ['Drug B'] * n,
            y=np.concatenate([improve_a, improve_b]),
            palette=['skyblue', 'lightcoral'])
ax.set_title("Improvement Comparison Between Groups")
ax.set_ylabel("Improvement (ms)")
st.pyplot(fig)

# ------------------------------
# Summary and Interpretation
# ------------------------------
st.header("📝 Summary & Interpretation")
st.subheader("Key Findings")

col1, col2, col3 = st.columns(3)
col1.metric("Drug A Effectiveness",
            "Significant" if results_a['significant'] else "Not Significant",
            delta=f"Δ = {results_a['mean_diff']:.1f} ms")

col2.metric("Drug B Effectiveness",
            "Significant" if results_b['significant'] else "Not Significant",
            delta=f"Δ = {results_b['mean_diff']:.1f} ms")

col3.metric("Drug Comparison",
            "A > B" if results_between['mean_improve_a'] > results_between['mean_improve_b'] else "B > A",
            "Different" if results_between['significant'] else "Similar")

st.subheader("Statistical Guidance")
st.markdown("""
1. **Within-Group Analysis**:
   - Shapiro-Wilk test determines normality of differences
   - Normal: Paired t-test (parametric)
   - Non-normal: Wilcoxon signed-rank test (non-parametric)

2. **Between-Group Analysis**:
   - Requires normality of improvements AND equal variances
   - Normal + equal variance: Independent t-test
   - Violated assumptions: Mann-Whitney U test

3. **Effect Sizes**:
   - Cohen's d quantifies practical significance
   - |d| < 0.2: Negligible, 0.2-0.5: Small, 0.5-0.8: Medium, >0.8: Large
""")

st.caption("""
*Note: This simulation assumes normally distributed reaction times with equal variances across groups. \
Real-world data may require different statistical approaches.*
""")

# ------------------------------
# Export Results
# ------------------------------
if st.button("📥 Download Analysis Report"):
    # Create report (simplified)
    report = f"""
    Drug Efficacy Statistical Analysis Report

    Parameters:
    - Participants per group: {n}
    - Significance level: {alpha}
    - Random seed: {seed}

    Drug A:
    - Baseline mean: {mu_a_before} ms
    - Effect size: {mu_a_effect} ms

    Drug B:
    - Baseline mean: {mu_b_before} ms
    - Effect size: {mu_b_effect} ms

    Within-Group Results:
    - Drug A: {'Significant' if results_a['significant'] else 'Not significant'} (p={results_a['p_value']:.4f})
    - Drug B: {'Significant' if results_b['significant'] else 'Not significant'} (p={results_b['p_value']:.4f})

    Between-Group Results:
    - {'Significant difference' if results_between['significant'] else 'No significant difference'} (p={results_between['p_value']:.4f})
    """
    st.download_button("Download Report", report, "drug_efficacy_report.txt")