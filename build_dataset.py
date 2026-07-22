import pandas as pd
import numpy as np

DATA = "data/"

# ---------- 1. Load raw sources ----------
results = pd.read_csv(DATA + "results.csv", parse_dates=["date"])
shootouts = pd.read_csv(DATA + "shootouts.csv", parse_dates=["date"])
rankings = pd.read_csv(DATA + "fifa_rankings.csv", parse_dates=["date"])

# Only real FIFA World Cup finals tournaments, recent eras
wc = results[(results["tournament"] == "FIFA World Cup") &
             (results["date"].dt.year.isin([2010, 2014, 2018, 2022]))].copy()
wc["year"] = wc["date"].dt.year
wc = wc.sort_values("date").reset_index(drop=True)

print("World Cup matches found (2010-2022):", len(wc))
print(wc["year"].value_counts().sort_index())

# ---------- 2. Determine stage (Group vs Knockout) ----------
match_count = {}
stages = []
for _, row in wc.iterrows():
    year = row["year"]
    h, a = row["home_team"], row["away_team"]
    kh = (year, h)
    ka = (year, a)
    match_count[kh] = match_count.get(kh, 0)
    match_count[ka] = match_count.get(ka, 0)
    ch, ca = match_count[kh], match_count[ka]
    stage = "Group" if (ch < 3 and ca < 3) else "Knockout"
    stages.append(stage)
    match_count[kh] += 1
    match_count[ka] += 1

wc["stage"] = stages
print(wc["stage"].value_counts())

# ---------- 3. Attach penalty shootout winner (for knockout draws) ----------
shootouts["date"] = pd.to_datetime(shootouts["date"])
so_lookup = {(r["date"], r["home_team"], r["away_team"]): r["winner"] for _, r in shootouts.iterrows()}
wc["shootout_winner"] = wc.apply(lambda r: so_lookup.get((r["date"], r["home_team"], r["away_team"])), axis=1)

# ---------- 4. Determine match winner (accounting for shootouts) ----------
def get_winner(r):
    if r["home_score"] > r["away_score"]:
        return r["home_team"]
    elif r["away_score"] > r["home_score"]:
        return r["away_team"]
    else:
        if pd.notna(r["shootout_winner"]):
            return r["shootout_winner"]
        return None

wc["winner"] = wc.apply(get_winner, axis=1)

# ---------- 5. Build FIFA ranking lookup ----------
TOURNAMENT_START = {2010: "2010-06-11", 2014: "2014-06-12", 2018: "2018-06-14", 2022: "2022-11-20"}

NAME_FIX = {
    "IR Iran": "Iran", "Korea Republic": "South Korea", "Korea DPR": "North Korea",
    "USA": "United States", "China PR": "China", "Cote d'Ivoire": "Ivory Coast", "Côte d'Ivoire": "Ivory Coast",
    "Czech Republic": "Czechia", "Congo DR": "DR Congo", "Cabo Verde": "Cape Verde",
}
rankings["team"] = rankings["team"].replace(NAME_FIX)

rank_by_year = {}
for year, start in TOURNAMENT_START.items():
    start_date = pd.Timestamp(start)
    snap = rankings[rankings["date"] <= start_date]
    if snap.empty:
        continue
    last_date = snap["date"].max()
    snap = snap[snap["date"] == last_date].copy()
    snap = snap.sort_values("total_points", ascending=False).reset_index(drop=True)
    snap["rank"] = snap.index + 1
    rank_by_year[year] = dict(zip(snap["team"], snap["rank"]))
    print(f"WC{year}: using ranking snapshot {last_date.date()}, {len(snap)} teams ranked")

def lookup_rank(year, team):
    return rank_by_year.get(year, {}).get(team, np.nan)

wc["home_rank"] = wc.apply(lambda r: lookup_rank(r["year"], r["home_team"]), axis=1)
wc["away_rank"] = wc.apply(lambda r: lookup_rank(r["year"], r["away_team"]), axis=1)

missing = wc[wc["home_rank"].isna() | wc["away_rank"].isna()][["year","home_team","away_team"]]
if len(missing):
    print("\nMatches missing a rank (name mismatch likely):")
    print(missing.to_string(index=False))

wc = wc.dropna(subset=["home_rank", "away_rank"]).copy()

# ---------- 6. Build final feature set ----------
wc["rank_gap"] = (wc["home_rank"] - wc["away_rank"]).abs()
wc["favorite"] = np.where(wc["home_rank"] < wc["away_rank"], wc["home_team"], wc["away_team"])
wc["underdog"] = np.where(wc["home_rank"] < wc["away_rank"], wc["away_team"], wc["home_team"])

def is_upset(r):
    if r["winner"] is None:
        return 0
    return int(r["winner"] == r["underdog"])

wc["upset"] = wc.apply(is_upset, axis=1)
wc["decisive"] = wc["winner"].notna().astype(int)
wc["is_knockout"] = (wc["stage"] == "Knockout").astype(int)
wc["era"] = "32-team (2010-2022)"

out_cols = ["year","date","home_team","away_team","home_score","away_score","stage",
            "is_knockout","home_rank","away_rank","rank_gap","favorite","underdog",
            "winner","decisive","upset","era"]
wc_final = wc[out_cols].sort_values("date")
wc_final.to_csv(DATA + "wc_2010_2022_processed.csv", index=False)
print("\nSaved", len(wc_final), "rows to wc_2010_2022_processed.csv")
print("\nUpset rate among decisive matches (2010-2022):",
      round(wc_final.loc[wc_final.decisive==1, "upset"].mean(), 3))