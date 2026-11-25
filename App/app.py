from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import pickle
import numpy as np

DATA_PATH = Path(__file__).parent / "data" / "2025_preds_data.parquet"
MODEL_PATH = Path(__file__).parent / "data" / "final_model.pkl"
CELL_PATH = Path(__file__).parent / "data" / "cell_data.csv"

MIN_PITCHES = 20   # zones with fewer pitches than this are left blank
MAX_RATE = 0.30    

# Where to draw the 5x5 grid
COL_W = 17 / 3
ZONE_BOTTOM = 19.4
ZONE_TOP = 38.5
ROW_H = (ZONE_TOP - ZONE_BOTTOM) / 3

BLUE = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
RED = "#d03b3b"
GRAY = "#e1e0d9"


@st.cache_data
def load_data():
    columns = ["cell", "stand", "p_throws", "player_name", "hp_official_name",
               "inning_topbot", "count", "pitch_cat", "home_team", "inning",
               "outs_when_up", "wrong_call", "pred_prob"]
    df = pd.read_parquet(DATA_PATH, columns=columns)

    # Split into a column (-2 to 2) and a row (1 to 5).
    cell = df["cell"].astype(int)
    df["zone_col"] = cell % 5 - 2
    df["zone_row"] = cell // 5 + 1

    # Cells are measured from the batters side. Flip lefties so every pitch is on the catchers view.
    df.loc[df["stand"] == "L", "zone_col"] = -df["zone_col"]

    df = df.rename(columns={
        "stand": "batter_hand",
        "p_throws": "pitcher_hand",
        "player_name": "pitcher",
        "hp_official_name": "umpire",
        "pitch_cat": "pitch_type",
        "outs_when_up": "outs",
    })
    df["home_away"] = df["inning_topbot"].map({"Bot": "Home", "Top": "Away"})
    df["pitch_type"] = df["pitch_type"].fillna("Unknown")
    return df


def zone_name(col, row, batter_hand):

    if col == 0:
        side = "Center"
    elif (col < 0) == (batter_hand == "R"):
        side = "Inside"
    else:
        side = "Outside"

    if row >= 4:
        height = "High"
    elif row == 3:
        height = "Middle"
    else:
        height = "Low"

    in_zone = abs(col) <= 1 and 2 <= row <= 4
    if in_zone:
        number = (4 - row) * 3 + (col + 1) + 1   # zones 1-9
        return str(number), f"Zone {number} - {height} {side}"
    return f"{height}<br>{side}", f"Off zone - {height} {side}"


def summarize(df, batter_hand):

    grouped = df.groupby(["zone_col", "zone_row"])
    zones = pd.DataFrame({
        "n_pitches": grouped.size(),
        "wrong_rate": grouped["wrong_call"].mean(),
        "model_score": grouped["pred_prob"].mean(),
    })

    # Make sure all 25 zones are there
    all_zones = pd.MultiIndex.from_product([range(-2, 3), range(1, 6)],
                                           names=["zone_col", "zone_row"])
    zones = zones.reindex(all_zones).reset_index()
    zones["n_pitches"] = zones["n_pitches"].fillna(0).astype(int)
    zones["enough_data"] = zones["n_pitches"] >= MIN_PITCHES

    names = [zone_name(c, r, batter_hand) for c, r in zip(zones["zone_col"], zones["zone_row"])]
    zones["short_name"] = [n[0] for n in names]
    zones["long_name"] = [n[1] for n in names]
    return zones


def make_chart(zones, show_challenge, threshold_pct):

    xs = [c * COL_W for c in range(-2, 3)]
    ys = [ZONE_BOTTOM + (r - 2) * ROW_H + ROW_H / 2 for r in range(1, 6)]

    z = []            #
    hover = []
    labels = []
    for row in range(1, 6):
        z_row = []
        hover_row = []
        for col in range(-2, 3):
            zone = zones[(zones["zone_col"] == col) & (zones["zone_row"] == row)].iloc[0]
            x = col * COL_W
            y = ys[row - 1]

            if not zone["enough_data"]:
                z_row.append(None)
                hover_row.append([zone["long_name"], zone["n_pitches"], "Too few pitches", 0])
                labels.append(dict(x=x, y=y, text=f"{zone['n_pitches']} pitches",
                                   showarrow=False, font=dict(size=10, color="#8a8983")))
                continue

            rate = zone["wrong_rate"]
            challenge = round(rate * 100) >= threshold_pct
            decision = "Challenge" if challenge else "Hold"
            hover_row.append([zone["long_name"], zone["n_pitches"], decision, zone["model_score"]])

            if show_challenge:
                z_row.append(1 if challenge else 0)
                text = f"{decision}<br>{rate:.0%}"
                text_color = "white" if challenge else "black"
            else:
                z_row.append(rate)
                text = f"<b>{zone['short_name']}</b><br>{rate:.0%}"
                text_color = "white" if rate > MAX_RATE / 2 else "black"
            labels.append(dict(x=x, y=y, text=text, showarrow=False,
                               font=dict(size=11, color=text_color)))
        z.append(z_row)
        hover.append(hover_row)

    if show_challenge:
        heatmap = go.Heatmap(
            x=xs, y=ys, z=z, zmin=0, zmax=1, xgap=2, ygap=2,
            colorscale=[[0, GRAY], [0.5, GRAY], [0.5, RED], [1, RED]],
            showscale=False, customdata=hover, hoverongaps=False,
            hovertemplate=("%{customdata[0]}<br>Recommendation: %{customdata[2]}"
                           "<br>Pitches: %{customdata[1]:,}<extra></extra>"),
        )
    else:
        heatmap = go.Heatmap(
            x=xs, y=ys, z=z, zmin=0, zmax=MAX_RATE, xgap=2, ygap=2,
            colorscale=BLUE, customdata=hover, hoverongaps=False,
            colorbar=dict(title="Chance<br>wrong", tickformat=".0%", thickness=12),
            hovertemplate=("%{customdata[0]}<br>Chance the call is wrong: %{z:.1%}"
                           "<br>Model score: %{customdata[3]:.1%}"
                           "<br>Pitches: %{customdata[1]:,}<extra></extra>"),
        )

    fig = go.Figure(heatmap)

    # Draw the outline of the strike zone
    fig.add_shape(type="rect", x0=-8.5, x1=8.5, y0=ZONE_BOTTOM, y1=ZONE_TOP,
                  line=dict(color="#5f5e5a", width=3))
    fig.update_layout(
        annotations=labels,
        height=600,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(range=[-20, 20], showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(range=[11, 47], showticklabels=False, showgrid=False, zeroline=False,
                   scaleanchor="x", scaleratio=1),
    )
    return fig


def make_table(zones):
    zones = zones.sort_values(["enough_data", "wrong_rate"], ascending=False)
    table = pd.DataFrame({
        "Location": zones["long_name"],
        "Chance call is wrong": (zones["wrong_rate"] * 100).round(1),
        "Model score": (zones["model_score"] * 100).round(1),
        "Pitches": zones["n_pitches"],
        "Recommendation": zones["recommendation"],
    })
    return table


# UI Page
st.set_page_config(page_title="ABS Challenge Planner", layout="wide")
st.title("ABS Challenge Planner")
mode = st.radio("Data source", ["Historical data", "New prediction"],
                horizontal=True, label_visibility="collapsed")
st.caption("Where is the umpire most likely to miss? Pick a situation to see which pitch "
           "locations are worth a challenge.")

data = load_data()
ALL = "All"


def choices(column):

    return [ALL] + sorted(data[column].dropna().unique())

def render_hist():

    # Sidebar filters
    with st.sidebar:
        st.header("Batter")
        batter_hand = st.radio("Bats", ["R", "L"], horizontal=True)
        home_away = st.selectbox("Batting at", choices("home_away"))

        st.header("Pitcher")
        pitcher_hand = st.selectbox("Throws", choices("pitcher_hand"))
        pitcher = st.selectbox("Pitcher", choices("pitcher"))
        pitch_type = st.selectbox("Pitch type", choices("pitch_type"))

        st.header("Game situation")
        umpire = st.selectbox("Home plate umpire", choices("umpire"))
        home_team = st.selectbox("Ballpark (home team)", choices("home_team"))
        count = st.selectbox("Count (balls-strikes)", choices("count"))
        inning = st.selectbox("Inning", choices("inning"))
        outs = st.selectbox("Outs", choices("outs"))

        st.header("Challenge rule")
        threshold_pct = st.slider("Challenge when the chance the call is wrong is at least",
                                min_value=1, max_value=40, value=10, format="%d%%")

    # Apply the filters
    filtered = data[data["batter_hand"] == batter_hand]
    filters = {"home_away": home_away, "pitcher_hand": pitcher_hand, "pitcher": pitcher,
            "pitch_type": pitch_type, "umpire": umpire, "home_team": home_team,
            "count": count, "inning": inning, "outs": outs}
    for column, value in filters.items():
        if value != ALL:
            filtered = filtered[filtered[column] == value]

    if filtered.empty:
        st.warning("No data for this combination of filters. Try loosening one of them.")
        st.stop()

    zones = summarize(filtered, batter_hand)
    zones["challenge"] = zones["enough_data"] & ((zones["wrong_rate"] * 100).round() >= threshold_pct)
    zones["recommendation"] = zones["challenge"].map({True: "Challenge", False: "Hold"})
    zones.loc[~zones["enough_data"], "recommendation"] = "Too few pitches"

    reliable = zones[zones["enough_data"]]
    if reliable.empty:
        st.warning(f"Every zone has fewer than {MIN_PITCHES} pitches. Try loosening a filter.")
        st.stop()

    # Headline numbers
    best = reliable.sort_values("wrong_rate", ascending=False).iloc[0]
    col1, col2, col3 = st.columns(3)
    col1.metric("Zones worth a challenge", f"{zones['challenge'].sum()} of 25")
    col2.metric("Best spot to challenge", best["long_name"],
                f"{best['wrong_rate']:.0%} chance the call is wrong", delta_color="off")
    col3.metric("Pitches behind this view", f"{zones['n_pitches'].sum():,}")

    # The two charts
    st.caption("Viewed from behind the catcher. Inside/outside are relative to the batter. "
            f"The dark outline is the strike zone. Zones with fewer than {MIN_PITCHES} pitches are left blank.")
    left, right = st.columns(2)
    with left:
        st.subheader("Chance the call is wrong")
        st.plotly_chart(make_chart(zones, False, threshold_pct), width="stretch")
    with right:
        st.subheader(f"Challenge map (at {threshold_pct}% or higher)")
        st.plotly_chart(make_chart(zones, True, threshold_pct), width="stretch")

    # Tables
    st.subheader("Best challenge spots")
    table_settings = {
        "Chance call is wrong": st.column_config.ProgressColumn(
            format="%.0f%%", min_value=0, max_value=MAX_RATE * 100),
        "Model score": st.column_config.NumberColumn(
            format="%.0f%%",
            help="Average model prediction. It runs well above the real miss rate, "
                "so use it to compare zones, not as a true probability."),
    }
    st.dataframe(make_table(reliable).head(5), hide_index=True, width="stretch",
                column_config=table_settings)
    with st.expander("See all 25 zones"):
        st.dataframe(make_table(zones), hide_index=True, width="stretch",
                    column_config=table_settings)
    st.caption(f"Data: {len(data):,} called pitches from the 2025 season.")

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    model["model"].set_params(xgb__device="cpu")
    return model

def load_model_data(c):
    df = pd.read_parquet(DATA_PATH, columns=c)
    return df

def load_cell_data():
    df = pd.read_csv(CELL_PATH)
    return df

def render_predict():
    st.subheader("Predict a new pitch")
    cols = ['release_speed', 'release_spin_rate', 'pitch_cat', 'balls', 'strikes',
       'inning', 'outs_when_up', 'stand', 'p_throws',
       'n_thruorder_pitcher', 'sz_top', 'sz_bot', 'called_strike',
       'pitch_number', 'pitcher_days_since_prev_game', 'hp_official_id','hp_official_name'
       ]
    model = load_model()
    data = load_model_data(cols)
    with st.sidebar:
        with st.form("predict_form"):
            print(data["release_speed"].min())
            stand = st.radio("Batter Side", list(data["stand"].unique()), horizontal=True)
            p_throws = st.radio("Pitcher Side", list(data["p_throws"].unique()), horizontal=True)
            speed = st.slider(
                "Pitch Speed", 
                data["release_speed"].min(), 
                data["release_speed"].max(),
                 data["release_speed"].median()
                 )
            spin = st.slider(
                "Pitch Spin Rate", 
                data["release_spin_rate"].min(),
                data["release_spin_rate"].max(), 
                data["release_spin_rate"].mean()
                )
            p_cat = st.selectbox("Pitch Category", list(data["pitch_cat"].unique()))
            balls = st.selectbox("Balls", list(data["balls"].unique()))
            strikes = st.selectbox("Strikes", list(data["strikes"].unique()))
            inning = st.selectbox("Inning", list(data["inning"].unique()))
            outs = st.selectbox("Outs", list(data["outs_when_up"].unique()))
            n_thruorder_pitcher = st.selectbox(
                "Times Through Order", sorted(data["n_thruorder_pitcher"].dropna().unique())
            )
            sz_top = st.slider(
                "Strike Zone Top (ft)",
                float(data["sz_top"].min()),
                float(data["sz_top"].max()),
                float(data["sz_top"].mean()),
                step=0.1,
            )
            sz_bot = st.slider(
                "Strike Zone Bottom (ft)",
                float(data["sz_bot"].min()),
                float(data["sz_bot"].max()),
                float(data["sz_bot"].mean()),
                step=0.1,
            )
            called_strike = st.selectbox(
                "Called Strike (1=yes/0=No)", data["called_strike"].dropna().unique()
            )
            pitch_number = st.slider(
            "Pitch Number",
            int(data["pitch_number"].min()),
            int(data["pitch_number"].max()),
            int(data["pitch_number"].median()),
            step=1,
                )
            pitcher_days_since_prev_game = st.slider(
                "Days Since Previous Game",
                int(data["pitcher_days_since_prev_game"].min()),
                int(data["pitcher_days_since_prev_game"].max()),
                int(data["pitcher_days_since_prev_game"].median()),
                step=1,
)
            hp_official_name = st.selectbox(
                "Home Plate Umpire", data["hp_official_name"].dropna().unique()
            )

            hp_official_id =  model["ump_map"][hp_official_name]

            

            submitted = st.form_submit_button("Predict")

    def summarize(df, batter_hand):
        grouped = df.groupby(["zone_col", "zone_row"])
        zones = pd.DataFrame({
            "n_pitches": grouped.size(),
            "wrong_rate": grouped["wrong_call"].mean(),
            "model_score": grouped["pred_prob"].mean(),
        })

        # Make sure all 25 zones are there
        all_zones = pd.MultiIndex.from_product([range(-2, 3), range(1, 6)],
                                            names=["zone_col", "zone_row"])
        zones = zones.reindex(all_zones).reset_index()
        zones["n_pitches"] = zones["n_pitches"].fillna(0).astype(int)
        zones["enough_data"] = zones["n_pitches"] >= MIN_PITCHES

        names = [zone_name(c, r, batter_hand) for c, r in zip(zones["zone_col"], zones["zone_row"])]
        zones["short_name"] = [n[0] for n in names]
        zones["long_name"] = [n[1] for n in names]
        return zones  

    def predict_zones(B,user_vals):
        rng = np.random.default_rng(42)
        hand = user_vals['stand']
        flags = ["High", "Low", "Inside", "Outside"]
        combos = load_cell_data().copy()
        combos["share"] = combos["share"] * rng.uniform(0.99, 1.01, size=len(combos))
        rows = []
        filtered_combos = combos[flags]
        

        for record in filtered_combos.to_dict("records"):
            row = {}
            row.update(B["defaults"])
            row.update(user_vals)
            row.update(record)
            rows.append(row)

        X = pd.DataFrame(rows)[list(B["dtypes"])].astype(B["dtypes"])
        combos["prob"] = B["model"].predict_proba(X)[:, 1] * combos["share"]
        cell_prob = combos.groupby("cell")["prob"].sum()
        probs = []
        rows = []
        keys = []

        for row in range (1,6):
            for col in range(-2,3):
                batter_col = -col if hand == "L" else col
                cell = (batter_col + 2) + ( (row - 1) * 5)
                probs.append(cell_prob.loc[cell])
                keys.append((col,row))

        pred_df = pd.DataFrame(keys, columns=["zone_col", "zone_row"]).assign(wrong_call=probs, pred_prob=probs)
        zones = summarize(pred_df, hand)
        zones["enough_data"] = True
        zones["n_pitches"] = 0
        return zones

    # st.write(model["ump_map"])
    # st.dataframe(data)
    # st.dataframe(pd.Series(model["ump_map"], name="hp_official_id"))

    if submitted:
        user_vals = {
        "release_speed": speed,
        "release_spin_rate": spin,
        "pitch_cat": p_cat,
        "balls": balls,
        "strikes": strikes,
        "inning": inning,
        "outs_when_up": outs,
        "stand": stand,
        "p_throws": p_throws,
        "n_thruorder_pitcher": n_thruorder_pitcher,
        "sz_top": sz_top,
        "sz_bot": sz_bot,
        "called_strike": called_strike,
        "pitch_number": pitch_number,
        "pitcher_days_since_prev_game": pitcher_days_since_prev_game,
        "hp_official_id": hp_official_id,
        }
        zones = predict_zones(model, user_vals)
        
        # The two charts
        st.caption("Viewed from behind the catcher. Inside/outside are relative to the batter. "
                f"The dark outline is the strike zone. Zones with fewer than {MIN_PITCHES} pitches are left blank.")
        left, right = st.columns(2)
        with left:
            st.subheader("Chance the call is wrong")
            st.plotly_chart(make_chart(zones, show_challenge=False, threshold_pct=50), width="stretch")
        with right:
            st.subheader(f"Challenge map")
            st.plotly_chart(make_chart(zones, show_challenge=True, threshold_pct=50), width="stretch")
     
        #pressed    = st.form_submit_button("Predict")
        



if mode == "Historical data":
    render_hist()
else:
    render_predict()