import pandas as pd
import numpy as np

DATA = "data/"

feat = pd.read_csv(DATA + "wc2026_features.csv")
old = pd.read_csv(DATA + "wc_2010_2022_processed.csv")

print(feat[["home_team_name","away_team_name","home_fifa_rank","away_fifa_rank",
            "is_knockout","home_score","away_score","match_result"]].head())
print("\nis_knockout counts:\n", feat["is_knockout"].value_counts())
print("\nmatch_result values:\n", feat["match_result"].value_counts())

def winner_2026(r):
    if r["match_result"] == "H":
        return r["home_team_name"]
    elif r["match_result"] == "A":
        return r["away_team_name"]
    else:
        return None

NAME_FIX_2026 = {
    "USA": "United States", "South Korea": "South Korea", "IR Iran": "Iran",
    "Korea Republic": "South Korea", "Czech Republic": "Czechia",
    "Ivory Coast": "Ivory Coast", "Cabo Verde": "Cape Verde", "DR Congo": "DR Congo",
}
for c in ["home_team_name", "away_team_name"]:
    feat[c] = feat[c].replace(NAME_FIX_2026)

feat["winner"] = feat.apply(winner_2026, axis=1)
feat["favorite"] = np.where(feat["home_fifa_rank"] < feat["away_fifa_rank"], feat["home_team_name"], feat["away_team_name"])
feat["underdog"] = np.where(feat["home_fifa_rank"] < feat["away_fifa_rank"], feat["away_team_name"], feat["home_team_name"])
feat["rank_gap"] = (feat["home_fifa_rank"] - feat["away_fifa_rank"]).abs()
feat["decisive"] = feat["winner"].notna().astype(int)
feat["upset"] = feat.apply(lambda r: int(r["winner"] == r["underdog"]) if r["decisive"] else 0, axis=1)
feat["stage"] = np.where(feat["is_knockout"] == 1, "Knockout", "Group")
feat["year"] = 2026
feat["era"] = "48-team (2026)"
feat["date"] = feat["date"]

out = feat.rename(columns={
    "home_team_name": "home_team", "away_team_name": "away_team",
    "home_fifa_rank": "home_rank", "away_fifa_rank": "away_rank",
})[["year","date","home_team","away_team","home_score","away_score","stage",
    "is_knockout","home_rank","away_rank","rank_gap","favorite","underdog",
    "winner","decisive","upset","era"]]

combined = pd.concat([old, out], ignore_index=True)
combined.to_csv(DATA + "wc_combined_2010_2026.csv", index=False)

print("\nTotal combined matches:", len(combined))
print("\nAll matches (upset rate diluted by unresolved group draws):")
print(combined.groupby("era")["upset"].agg(["mean","sum","count"]))

decisive = combined[combined["decisive"] == 1]
print("\nDecisive matches only (fair upset-rate comparison):")
print(decisive.groupby("era")["upset"].agg(["mean","sum","count"]))