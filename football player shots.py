import pandas as pd
import numpy as np
import scipy.stats as st

# Matplotlib backend for saving figures without opening GUI windows
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# FIFA World Cup 2026
# Objective 1 - Shooting Volume
#
# Research Question:
# Among players who appeared in at least one FIFA World Cup
# 2026 match, is the average number of shots greater than 2?
#
# H0: mu = 2
# Ha: mu > 2
# ============================================================


# ============================================================
# 1. Read the CSV file
# ============================================================

file_name = "football player data-Sheet.csv"

# The CSV file contains an extra header row.
df = pd.read_csv(
    file_name,
    sep="\t",
    encoding="gbk",
    header=1
)

print(df.columns.tolist())

print("=" * 60)
print("1. ORIGINAL DATA")
print("=" * 60)

print("Original number of rows:", len(df))

print("\nFirst five rows:")
print(df.head())


# ============================================================
# 2. Select the variables needed
# ============================================================

# Player = player name
# 90s    = playing time in 90-minute equivalents
# Sh     = total shots

df = df[["Player", "90s", "Sh"]].copy()


# ============================================================
# 3. Convert numerical variables
# ============================================================

df["90s"] = pd.to_numeric(
    df["90s"],
    errors="coerce"
)

df["Sh"] = pd.to_numeric(
    df["Sh"],
    errors="coerce"
)


# ============================================================
# 4. Check missing values
# ============================================================

print("\n" + "=" * 60)
print("2. MISSING VALUE CHECK")
print("=" * 60)

print(df.isnull().sum())


# Remove rows with missing values
df = df.dropna(
    subset=["Player", "90s", "Sh"]
)


# ============================================================
# 5. Data Wrangling
# ============================================================

# Keep only players who appeared in at least one match.
#
# 90s > 0 means the player had playing time.

population = df[df["90s"] > 0].copy()


print("\n" + "=" * 60)
print("3. DATA WRANGLING")
print("=" * 60)

print("Population size:", len(population))


# ============================================================
# 6. Define Population
# ============================================================

N = len(population)

print("\nPopulation:")
print(
    "Players who appeared in at least one "
    "FIFA World Cup 2026 match."
)

print("N =", N)


# ============================================================
# 7. Simple Random Sampling
# ============================================================

sample_size = 100

sample = population.sample(
    n=sample_size,
    random_state=42
).copy()


print("\n" + "=" * 60)
print("4. SIMPLE RANDOM SAMPLING")
print("=" * 60)

print("Population size:", N)

print("Sample size:", len(sample))

print("\nRandom sample:")
print(sample)


# ============================================================
# 8. Extract Shots variable
# ============================================================

shots = sample["Sh"].to_numpy()


# ============================================================
# 9. Descriptive Statistics
# ============================================================

print("\n" + "=" * 60)
print("5. DESCRIPTIVE STATISTICS")
print("=" * 60)

n = len(shots)

sample_mean = np.mean(shots)

sample_median = np.median(shots)

sample_sd = np.std(
    shots,
    ddof=1
)

sample_min = np.min(shots)

sample_max = np.max(shots)


print("Sample size:", n)

print(
    "Mean:",
    round(sample_mean, 3)
)

print(
    "Median:",
    round(sample_median, 3)
)

print(
    "Standard deviation:",
    round(sample_sd, 3)
)

print(
    "Minimum:",
    sample_min
)

print(
    "Maximum:",
    sample_max
)


# ============================================================
# 10. Figure 1 - Histogram
# ============================================================

plt.figure(figsize=(8, 5))

plt.hist(
    shots,
    bins=10
)

# Benchmark line
plt.axvline(
    2,
    linestyle="--",
    linewidth=2,
    label="Benchmark = 2"
)

# Sample mean line
plt.axvline(
    sample_mean,
    linestyle="-",
    linewidth=2,
    label=f"Sample Mean = {sample_mean:.2f}"
)

plt.xlabel("Number of Shots")

plt.ylabel("Frequency")

plt.title(
    "Distribution of Shots per Player"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "Figure_1_Shots_Distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 10.1 Figure 2 - Sample Mean vs Benchmark
# ============================================================

plt.figure(figsize=(7, 5))

labels = [
    "Benchmark",
    "Sample Mean"
]

values = [
    2,
    sample_mean
]

plt.bar(
    labels,
    values
)

plt.ylabel("Average Number of Shots")

plt.title(
    "Sample Mean vs Benchmark"
)

# Add value labels
for i, value in enumerate(values):
    plt.text(
        i,
        value + 0.05,
        f"{value:.2f}",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    "Figure_2_Mean_vs_Benchmark.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 11. 95% Confidence Interval
# ============================================================

confidence_level = 0.95

alpha = 1 - confidence_level

degrees_of_freedom = n - 1


# Critical t value
t_critical = st.t.ppf(
    1 - alpha / 2,
    degrees_of_freedom
)


# Standard error
standard_error = (
    sample_sd / np.sqrt(n)
)


# Margin of error
margin_of_error = (
    t_critical * standard_error
)


# Lower and upper bounds
lower_bound = (
    sample_mean - margin_of_error
)

upper_bound = (
    sample_mean + margin_of_error
)


print("\n" + "=" * 60)
print("6. 95% CONFIDENCE INTERVAL")
print("=" * 60)

print(
    "Confidence level:",
    confidence_level
)

print(
    "Degrees of freedom:",
    degrees_of_freedom
)

print(
    "t-critical:",
    round(t_critical, 3)
)

print(
    "Standard error:",
    round(standard_error, 3)
)

print(
    "Margin of error:",
    round(margin_of_error, 3)
)

print(
    "Lower bound:",
    round(lower_bound, 3)
)

print(
    "Upper bound:",
    round(upper_bound, 3)
)

print(
    "95% Confidence Interval:",
    f"[{lower_bound:.3f}, {upper_bound:.3f}]"
)


# ============================================================
# 11.1 Figure 3 - 95% Confidence Interval
# ============================================================

plt.figure(figsize=(8, 4))

plt.errorbar(
    sample_mean,
    0,
    xerr=margin_of_error,
    fmt="o",
    capsize=8,
    markersize=8
)

# Benchmark line
plt.axvline(
    2,
    linestyle="--",
    linewidth=2,
    label="Benchmark = 2"
)

plt.yticks([])

plt.xlabel("Mean Number of Shots")

plt.title(
    "95% Confidence Interval for Population Mean"
)

plt.xlim(
    min(lower_bound, 2) - 0.5,
    max(upper_bound, 2) + 0.5
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "Figure_3_95_Confidence_Interval.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 12. One-Sample t-Test
# ============================================================

# Research Question:
#
# Is the average number of shots greater than 2?
#
# H0: mu = 2
# Ha: mu > 2

test_value = 2

t_statistic, p_value = st.ttest_1samp(
    shots,
    popmean=test_value,
    alternative="greater"
)


print("\n" + "=" * 60)
print("7. ONE-SAMPLE t-TEST")
print("=" * 60)

print(
    "Null hypothesis:",
    "H0: mu = 2"
)

print(
    "Alternative hypothesis:",
    "Ha: mu > 2"
)

print(
    "Significance level:",
    0.05
)

print(
    "Test value:",
    test_value
)

print(
    "Sample mean:",
    round(sample_mean, 3)
)

print(
    "t-statistic:",
    round(t_statistic, 3)
)

print(
    "p-value:",
    round(p_value, 4)
)


# ============================================================
# 13. Hypothesis Test Decision
# ============================================================

significance_level = 0.05


print("\n" + "=" * 60)
print("8. HYPOTHESIS TEST DECISION")
print("=" * 60)


if p_value < significance_level:

    print("p-value < 0.05")

    print("Reject H0.")

    print(
        "There is sufficient statistical evidence "
        "that the population mean number of shots "
        "is greater than 2."
    )

else:

    print("p-value >= 0.05")

    print("Fail to reject H0.")

    print(
        "There is insufficient statistical evidence "
        "that the population mean number of shots "
        "is greater than 2."
    )


# ============================================================
# 13.1 Figure 4 - p-value vs Significance Level
# ============================================================

plt.figure(figsize=(7, 5))

labels = [
    "Significance Level",
    "p-value"
]

values = [
    significance_level,
    p_value
]

plt.bar(
    labels,
    values
)

plt.ylabel("Value")

plt.title(
    "p-value vs Significance Level"
)

# Add value labels
for i, value in enumerate(values):
    plt.text(
        i,
        value + 0.002,
        f"{value:.4f}",
        ha="center"
    )

plt.ylim(
    0,
    max(values) + 0.02
)

plt.tight_layout()

plt.savefig(
    "Figure_4_p_value_vs_Significance_Level.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 14. Final Summary
# ============================================================

print("\n" + "=" * 60)
print("9. FINAL SUMMARY")
print("=" * 60)

print(
    "Population size:",
    N
)

print(
    "Sample size:",
    n
)

print(
    "Sample mean:",
    round(sample_mean, 3)
)

print(
    "Sample standard deviation:",
    round(sample_sd, 3)
)

print(
    "95% Confidence Interval:",
    f"[{lower_bound:.3f}, {upper_bound:.3f}]"
)

print(
    "t-statistic:",
    round(t_statistic, 3)
)

print(
    "p-value:",
    round(p_value, 4)
)

print(
    "Benchmark:",
    test_value
)

print(
    "Significance level:",
    significance_level
)


# ============================================================
# Figure Saving Confirmation
# ============================================================

print("\n" + "=" * 60)
print("FIGURES")
print("=" * 60)

print("All four figures saved successfully.")

print("1. Figure_1_Shots_Distribution.png")
print("2. Figure_2_Mean_vs_Benchmark.png")
print("3. Figure_3_95_Confidence_Interval.png")
print("4. Figure_4_p_value_vs_Significance_Level.png")