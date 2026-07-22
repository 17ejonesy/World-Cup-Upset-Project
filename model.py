import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.proportion import proportions_ztest
import matplotlib.pyplot as plt

DATA = "data/"
df = pd.read_csv(DATA + "wc_combined_2010_2026.csv")
decisive = df[df["decisive"] == 1].copy()

counts = decisive.groupby("era")["upset"].agg(["sum", "count"])
print("Upset counts by era:\n", counts, "\n")

successes = [counts.loc["32-team (2010-2022)", "sum"], counts.loc["48-team (2026)", "sum"]]
nobs = [counts.loc["32-team (2010-2022)", "count"], counts.loc["48-team (2026)", "count"]]
z, p = proportions_ztest(successes, nobs)
print(f"Two-proportion z-test: z = {z:.3f}, p = {p:.4f}")
print("(Not statistically significant at p<0.05 with this sample size)" if p >= 0.05
      else "(Statistically significant difference)")

decisive["is_2026"] = (decisive["era"] == "48-team (2026)").astype(int)
X = decisive[["rank_gap", "is_knockout", "is_2026"]].values
y = decisive["upset"].values

model = LogisticRegression()
model.fit(X, y)
print("\nLogistic regression coefficients:")
for name, coef in zip(["rank_gap", "is_knockout", "is_2026"], model.coef_[0]):
    print(f"  {name}: {coef:.4f}")
print(f"  intercept: {model.intercept_[0]:.4f}")

decisive["gap_bucket"] = pd.cut(decisive["rank_gap"], bins=[0,10,25,50,100,300],
                                 labels=["0-10","10-25","25-50","50-100","100+"])
calib = decisive.groupby("gap_bucket", observed=True)["upset"].agg(["mean","count"])
print("\nActual upset rate by ranking-gap bucket:\n", calib)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

calib_by_era = decisive.groupby(["gap_bucket","era"], observed=True)["upset"].mean().unstack()
calib_by_era.plot(kind="bar", ax=axes[0], color=["#2c7fb8","#e34a33"])
axes[0].set_title("Upset rate by FIFA ranking gap")
axes[0].set_xlabel("Ranking gap (buckets)")
axes[0].set_ylabel("Upset rate")
axes[0].legend(title="", fontsize=8)
axes[0].tick_params(axis='x', rotation=0)

era_rates = decisive.groupby("era")["upset"].mean()
bars = axes[1].bar(era_rates.index, era_rates.values, color=["#2c7fb8","#e34a33"])
axes[1].set_title(f"Overall upset rate by era\n(z={z:.2f}, p={p:.3f})")
axes[1].set_ylabel("Upset rate")
for b in bars:
    axes[1].text(b.get_x()+b.get_width()/2, b.get_height()+0.005, f"{b.get_height():.1%}", ha="center")

plt.tight_layout()
plt.savefig("upset_analysis.png", dpi=150)
print("\nSaved chart to upset_analysis.png")