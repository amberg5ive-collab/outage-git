import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import hashlib

EXCEL_PATH = "data/raw/Proposed_AOutageIntelligence_Reports.xlsx"
SHEET_NAME = "JPS Reports"

st.set_page_config(page_title="Outage Intelligence", page_icon="💡", layout="wide")

# =========================================================
# PARISH CENTRES + CRITICAL ZONES
# =========================================================
PARISH_COORDS = {
    "Kingston": (17.9712, -76.7936), "St Andrew": (18.0179, -76.8090),
    "St Thomas": (17.9060, -76.3980), "Portland": (18.1740, -76.4500),
    "St Mary": (18.3670, -77.0500), "St Ann": (18.4370, -77.2000),
    "Trelawny": (18.3830, -77.6330), "St James": (18.4760, -77.9130),
    "Hanover": (18.4160, -78.1330), "Westmoreland": (18.2330, -78.1500),
    "St Elizabeth": (18.0500, -77.7830), "Manchester": (18.0330, -77.5000),
    "Clarendon": (17.9660, -77.2830), "St Catherine": (17.9830, -77.0160),
}
CRITICAL_KEYWORDS = ["hospital", "clinic", "health centre", "health center", "nursing home", "police", "fire station", "water pump", "water treatment", "school", "university", "college", "airport", "prison", "courthouse", "army", "military", "emergency", "icu", "maternity", "critical zone"]

def is_critical_zone(rows):
    text = ""
    if "Community" in rows.columns:
        text += " ".join(rows["Community"].astype(str).tolist()) + " "
    if "Description" in rows.columns:
        text += " ".join(rows["Description"].astype(str).tolist()) + " "
    if "Outage Type" in rows.columns:
        text += " ".join(rows["Outage Type"].astype(str).tolist())
    text = text.lower()
    for kw in CRITICAL_KEYWORDS:
        if kw in text:
            return True, kw.title()
    return False, ""

# =========================================================
# THE BRAIN - DBSCAN + CRITICAL ZONES
# =========================================================
@st.cache_data(ttl=30)
def run_brain():
    try:
        df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)
    except Exception as e:
        return None, f"Could not read Excel: {e}. Close Excel and try again."
    if "Data_Check_Status" in df.columns:
        df = df[df["Data_Check_Status"] == "Valid"].copy()
    if "Utility Type" in df.columns:
        df = df[df["Utility Type"].astype(str).str.lower().str.contains("power")]
    df["lat"] = pd.to_numeric(df["Lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["Long"], errors="coerce")
    df["town"] = df["Community"] if "Community" in df.columns else ""
    df["hour"] = pd.to_datetime(df["Time"].astype(str), format="%H:%M", errors="coerce").dt.hour
    df_clean = df.dropna(subset=["lat", "lon", "hour"])
    if len(df_clean) == 0:
        return {"total_reports": len(df), "incidents": [], "noise": len(df), "df": df, "map_df": pd.DataFrame(columns=["lat","lon","name"])}, "No Lat/Long found."
    try:
        from sklearn.cluster import DBSCAN
        df_clean = df_clean.copy()
        df_clean["hours_scaled"] = df_clean["hour"] * 0.005
        brain_model = DBSCAN(eps=0.01, min_samples=2)
        df_clean["incident_id"] = brain_model.fit_predict(df_clean[["lat", "lon", "hours_scaled"]])
    except Exception as e:
        return {"total_reports": len(df), "incidents": [], "noise": len(df), "df": df, "map_df": pd.DataFrame(columns=["lat","lon","name"])}, f"DBSCAN error: {e}"

    incidents = []
    for cluster_id in sorted(df_clean["incident_id"].unique()):
        if cluster_id == -1:
            continue
        rows = df_clean[df_clean["incident_id"] == cluster_id]
        name = rows["town"].mode().iloc[0] if len(rows) else f"Cluster {cluster_id}"
        parish = rows["Parish"].mode().iloc[0] if "Parish" in rows.columns else ""
        count = len(rows)
        critical, critical_word = is_critical_zone(rows)
        if critical:
            priority, status = "P1 - Critical Zone 🏥", "🔴 Dispatch Now - Critical Zone"
        elif count >= 300:
            priority, status = "P1 - Critical", "🔴 Dispatch Now"
        elif count >= 100:
            priority, status = "P2 - High", "🟡 Investigating"
        else:
            priority, status = "P3 - Medium", "🟢 Monitoring"
        incidents.append({"Priority": priority, "Incident": str(name), "Parish": str(parish), "Reports": count, "Utility": "Power", "Provider": "Power Utility", "Status": status, "Lat": float(rows["lat"].mean()), "Long": float(rows["lon"].mean()), "Critical_Reason": critical_word if critical else ""})

    # --- FIX: Single Critical Reports GROUPED by Community + Parish ---
    noise_rows = df_clean[df_clean["incident_id"] == -1]
    grouped_singles = noise_rows.groupby(["town", "Parish"])
    critical_single_count = 0
    for (town_name, parish_name), group in grouped_singles:
        critical, critical_word = is_critical_zone(group)
        if critical:
            critical_single_count += len(group)
            incidents.append({"Priority": "P1 - Critical Zone 🏥", "Incident": str(town_name), "Parish": str(parish_name), "Reports": len(group), "Utility": "Power", "Provider": "Power Utility", "Status": "🔴 Dispatch Now - Critical Zone (Single Report)", "Lat": float(group["lat"].mean()), "Long": float(group["lon"].mean()), "Critical_Reason": critical_word})

    incidents = sorted(incidents, key=lambda x: (0 if "Critical Zone" in x["Priority"] else 1 if "P1" in x["Priority"] else 2 if "P2" in x["Priority"] else 3, -x["Reports"]))
    noise_count = int(len(noise_rows) - critical_single_count)
    result = {"total_reports": len(df), "incidents": incidents, "noise": max(0, noise_count), "df": df, "map_df": pd.DataFrame([{"lat": i["Lat"], "lon": i["Long"], "name": f"{i['Incident']} - {i['Reports']}"} for i in incidents]) if incidents else pd.DataFrame(columns=["lat","lon","name"])}
    return result, None

def ai_reply(user_msg, context="user", brain=None):
    msg = user_msg.lower()
    if context == "user":
        if "no power" in msg or "light" in msg or "power" in msg or "out" in msg or "dark" in msg:
            return "I'm sorry to hear that 💡 I can help you report it quickly. Which Parish and Community are you in?"
        if "how long" in msg or "when" in msg or "fix" in msg:
            return "Once you submit, our system groups your report with other power reports nearby. Hospital / critical zones are always P1, even with 1 report."
        if "thank" in msg:
            return "You're welcome! You're helping your whole community by reporting ❤️"
        return "Thanks! Tell me: 1) Your Community, 2) When did the power go out? If it's near a hospital / clinic, tell me - that makes it Critical."
    else:
        if brain and brain["incidents"]:
            top = brain["incidents"][0]
            if "priorit" in msg or "first" in msg or top["Incident"].lower() in msg:
                return f"Prioritize {top['Incident']} FIRST - {top['Reports']} power reports, {top['Priority']}. Dispatch a crew there."
            summary = ", ".join([f"{i['Incident']} ({i['Reports']})" for i in brain["incidents"][:3]])
            return f"I see {len(brain['incidents'])} Active Incidents, all Power: {summary}. I recommend {top['Incident']} first."
        return "No incidents found yet. Check your Excel has Lat/Long."

brain, brain_error = run_brain()

if "dispatch" not in st.session_state:
    st.session_state.dispatch = {}

page = st.radio("Go to:", ["🧑 User Page - Report an Outage", "🛡️ Admin Page - Dashboard", ], horizontal=True)

if "User Page" in page:
    st.markdown("""
    <style>
        .stApp { background-color: #0A0A0A; }
        .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp label, .stApp span { color: #FFFFFF !important; }
        .stButton > button, .stFormSubmitButton > button { background-color: #FFD400 !important; color: #000000 !important; font-weight: 800 !important; border: none !important; border-radius: 12px !important; padding: 12px 20px !important; font-size: 16px !important; box-shadow: 0 4px 0 #CCAA00; }
        div[data-testid="stRadio"] label { background-color: #1A1A1A !important; color: #FFD400 !important; padding: 10px 18px !important; border-radius: 10px !important; border: 2px solid #FFD400 !important; font-weight: 700 !important; }
        div[data-testid="stRadio"] label[data-checked="true"] { background-color: #FFD400 !important; color: #000000 !important; }
        .stTextInput > div > div > input, .stTextArea > div > div > textarea { background-color: #1A1A1A !important; color: #FFFFFF !important; border: 1px solid #FFD400 !important; border-radius: 10px !important; }
        div[data-testid="stSelectbox"] > div > div { background-color: #1A1A1A !important; border: 1px solid #FFD400 !important; border-radius: 10px !important; }
        div[data-testid="stSelectbox"] > div > div > div { color: #FFFFFF !important; }
        div[data-testid="stSelectbox"] svg { fill: #FFD400 !important; }
        body div[data-baseweb="popover"] *, body div[data-baseweb="menu"] *, body ul[role="listbox"] *, body li[role="option"] * { color: #000000 !important; background-color: #FFFFFF !important; }
        body div[data-baseweb="popover"], body div[data-baseweb="menu"], body ul[role="listbox"] { background-color: #FFFFFF !important; border: 2px solid #FFD400 !important; }
        body div[data-baseweb="option"]:hover *, body li[role="option"]:hover * { background-color: #FFD400 !important; color: #000000 !important; }
        div[data-testid="stDateInput"] > div > div, div[data-testid="stTimeInput"] > div > div { background-color: #1A1A1A !important; border: 1px solid #FFD400 !important; border-radius: 10px !important; }
        div[data-testid="stDateInput"] input, div[data-testid="stTimeInput"] input { color: #FFFFFF !important; background-color: #1A1A1A !important; }
        div[data-testid="stMetric"] { background-color: #1A1A1A !important; border: 2px solid #FFD400 !important; border-radius: 15px !important; padding: 15px !important; }
        div[data-testid="stAlert"] { background-color: #1A1A1A !important; border: 1px solid #FFD400 !important; color: #FFFFFF !important; }
        div[data-testid="stChatMessage"] { background-color: #1A1A1A !important; border: 1px solid #333333 !important; border-radius: 12px !important; }
        hr { border-color: #FFD400 !important; }
    </style>
    <div style="background-color:#FFD400; padding:14px 20px; border-radius:12px; text-align:center; margin-bottom:10px; box-shadow: 10px,10px,12px #000000;">
        <span style="color:#000000; font-size:24px; font-weight:900; text-shadow: 5px, 5px, 6px black;">⚡ OUTAGE INTELLIGENCE &nbsp;|&nbsp; POWER ONLY</span>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
        .stApp { background-color: #F8F9FA; }
        .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp label { color: #1A1A1A !important; }
        .stButton > button { background-color: #1A1A1A !important; color: #FFD400 !important; font-weight: 700 !important; border: none !important; border-radius: 8px !important; padding: 10px 18px !important; }
        div[data-testid="stMetric"] { background-color: #FFFFFF !important; border: 1px solid #E0E0E0 !important; border-left: 5px solid #FFD400 !important; border-radius: 10px !important; padding: 16px !important; }
        div[data-testid="stMetric"] label { color: #666666 !important; font-weight: 600 !important; font-size: 13px !important; text-transform: uppercase; }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #000000 !important; font-size: 30px !important; font-weight: 800 !important; }
        div[data-testid="stDataFrame"] { border: 1px solid #E0E0E0 !important; border-radius: 10px !important; }
    </style>
    <div style="background-color:#FFFFFF; padding:16px 20px; border-radius:10px; border:1px solid #E0E0E0; border-left:6px solid #FFD400; margin-bottom:12px; display:flex; justify-content:space-between; align-items:center;">
        <span style="color:#000000; font-size:22px; font-weight:800;">Outage Intelligence</span>
        <span style="background-color:#FFD400; color:#000000; padding:6px 12px; border-radius:20px; font-size:13px; font-weight:700;">● LIVE DASHBOARD &nbsp;</span>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# =========================================================
# USER PAGE
# =========================================================
if "User Page" in page:
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.subheader("Report a Power Outage in seconds")
        st.caption("Power | Electricity Only")
        st.info("⚡ Utility: Power - Electricity Only | 🏥 Near a hospital / clinic? Tick the box - that makes it Critical.")
        with st.form("report_form"):
            parish = st.selectbox("Parish", ["Kingston", "St Andrew", "St Thomas", "Portland", "St Mary", "St Ann", "Trelawny", "St James", "Hanover", "Westmoreland", "St Elizabeth", "Manchester", "Clarendon", "St Catherine"])
            community = st.text_input("Community / Area", placeholder="e.g. Half Way Tree")
            outage_type = st.selectbox("What happened?", ["No Power", "Flickering Lights", "Wire Down", "Transformer Blast", "Other"])
            c_date, c_time = st.columns(2)
            date = c_date.date_input("Date", datetime.now().date())
            time = c_time.time_input("Time", datetime.now().time())
            description = st.text_area("Describe what you see", placeholder="e.g. Whole street dark since 7pm... Near hospital? Write hospital here.")
            is_critical = st.checkbox("🏥 This is near a Hospital / Clinic / Critical Zone")
            submitted = st.form_submit_button("✅ Submit Report", use_container_width=True)
        if submitted:
            if not community.strip():
                st.error("Please enter your Community!")
            else:
                # FIXED: Same Community = Same Lat/Long every time, so DBSCAN groups them!
                base_lat, base_lon = PARISH_COORDS.get(parish, (18.0, -77.0))
                seed = int(hashlib.md5(community.strip().lower().encode()).hexdigest()[:8], 16)
                offset_lat = ((seed % 100) / 100 - 0.5) * 0.01
                offset_lon = (((seed // 100) % 100) / 100 - 0.5) * 0.01
                lat = base_lat + offset_lat
                lon = base_lon + offset_lon
                final_desc = (description + " - NEAR HOSPITAL / CRITICAL ZONE") if is_critical else description
                new_row = {"Report_ID": f"RPT-USER-{datetime.now().strftime('%Y%m%d%H%M%S')}", "Scenario": "Live User Report", "True_Incident_ID": "", "Date": date.strftime("%Y-%m-%d"), "Time": time.strftime("%H:%M"), "Parish": parish, "Community": community.strip(), "Lat": lat, "Long": lon, "Utility Type": "Power", "Provider": "Power Utility", "Outage Type": outage_type, "Status": "Reported", "Data_Check_Status": "Valid", "Description": final_desc}
                try:
                    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                    with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
                        df.to_excel(writer, sheet_name=SHEET_NAME, index=False)
                    st.cache_data.clear()
                    if is_critical:
                        st.success(f"🚨 CRITICAL ZONE report saved for {community}, {parish}. This will go to P1 immediately!")
                    else:
                        st.success(f"✅ Saved for {community}, {parish}. ID: {new_row['Report_ID']}")
                    st.balloons()
                except Exception as e:
                    st.error(f"Close Excel first, then try again. Error: {e}")
    with col2:
        st.subheader("🤖 AI Assistant")
        if "user_chat" not in st.session_state:
            st.session_state.user_chat = [("assistant", "Hi! How can I help you report your power outage?")]
        for role, msg in st.session_state.user_chat:
            with st.chat_message(role):
                st.write(msg)
        user_q = st.chat_input("Type here... e.g. There's no power on my street")
        if user_q:
            st.session_state.user_chat.append(("user", user_q))
            st.session_state.user_chat.append(("assistant", ai_reply(user_q, "user")))
            st.rerun()

# =========================================================
# ADMIN PAGE - FILTERS + DISPATCH + CHARTS
# =========================================================
else:
    if brain_error:
        st.error(brain_error)
    if brain is None:
        st.warning("Brain could not load. Check Excel path.")
        st.stop()

    incidents_display = []
    for inc in brain["incidents"]:
        d = st.session_state.dispatch.get(inc["Incident"], {})
        new_inc = inc.copy()
        if d.get("status"):
            new_inc["Status"] = d["status"]
        new_inc["Crew"] = d.get("crew", "— Unassigned —") if d.get("crew") else "— Unassigned —"
        new_inc["ETA"] = d.get("eta", "—")
        new_inc["Note"] = d.get("note", "")
        incidents_display.append(new_inc)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active Incidents", len(brain["incidents"]))
    c2.metric("Total Reports", brain["total_reports"])
    c3.metric("Single Reports (P99)", brain["noise"])
    dispatched = sum(1 for x in incidents_display if x["Crew"] != "— Unassigned —")
    c4.metric("Crews Dispatched", f"{dispatched} / {len(incidents_display)}")
    top_incident = brain["incidents"][0] if brain["incidents"] else {"Incident": "None", "Reports": 0}

    st.subheader("📊 Live Overview")
    if brain["incidents"]:
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.caption("Reports by Parish (Top 8)")
            parish_counts = pd.DataFrame(brain["incidents"]).groupby("Parish")["Reports"].sum().sort_values(ascending=False).head(8)
            st.bar_chart(parish_counts)
        with chart_col2:
            st.caption("Incidents by Priority")
            pri_counts = pd.DataFrame(brain["incidents"])["Priority"].value_counts()
            st.bar_chart(pri_counts)
    else:
        st.info("Charts will appear once incidents are found.")

    st.subheader("📍 Live Incidents")
    f1, f2, f3 = st.columns([1, 1, 1.5])
    with f1:
        priority_filter = st.selectbox("Filter Priority", ["All", "P1 - Critical Zone 🏥 Only", "P1 Only", "P2 Only", "P3 Only"])
    with f2:
        parish_list = ["All Parishes"] + sorted(list(set([i["Parish"] for i in incidents_display])))
        parish_filter = st.selectbox("Filter Parish", parish_list)
    with f3:
        search = st.text_input("🔍 Search Community", placeholder="e.g. Half Way Tree, Kingston...")

    filtered = incidents_display.copy()
    if priority_filter == "P1 - Critical Zone 🏥 Only":
        filtered = [x for x in filtered if "Critical Zone" in x["Priority"]]
    elif priority_filter == "P1 Only":
        filtered = [x for x in filtered if x["Priority"].startswith("P1")]
    elif priority_filter == "P2 Only":
        filtered = [x for x in filtered if x["Priority"].startswith("P2")]
    elif priority_filter == "P3 Only":
        filtered = [x for x in filtered if x["Priority"].startswith("P3")]
    if parish_filter != "All Parishes":
        filtered = [x for x in filtered if x["Parish"] == parish_filter]
    if search.strip():
        filtered = [x for x in filtered if search.lower() in x["Incident"].lower() or search.lower() in x["Parish"].lower()]

    st.caption(f"Showing {len(filtered)} of {len(incidents_display)} incidents | Priority = Count + Critical Zone")
    if filtered:
        table_df = pd.DataFrame(filtered)[["Priority", "Incident", "Parish", "Reports", "Status", "Crew", "ETA", "Note"]]
        table_df=table_df.rename(columns={"Incident":"Community/Town", "Reports": "No of Reports"})
        st.dataframe(table_df, use_container_width=True, hide_index=True)
        csv = table_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Incident Report (CSV)", csv, "Outage_Intelligence_Report.csv", "text/csv", use_container_width=True)
    else:
        st.info("No incidents match your filter. Try 'All'.")

    if brain["noise"] > 0:
        st.caption(f"⚪ {brain['noise']} single reports did not cluster (Priority 99 - No Action)")

    st.divider()
    st.subheader("🚑 Dispatch Centre - Assign Crews")
    st.caption("Pick an incident, assign a crew, set status + ETA. This is what a real control room does.")
    if brain["incidents"]:
        incident_names = [f"{i['Priority']} | {i['Incident']} ({i['Parish']}) - {i['Reports']} reports" for i in brain["incidents"]]
        selected = st.selectbox("Select Incident to Dispatch", incident_names)
        selected_incident = brain["incidents"][incident_names.index(selected)]
        inc_key = selected_incident["Incident"]
        current = st.session_state.dispatch.get(inc_key, {})
        crew_options = ["— Unassigned —", "Crew 1 - Kingston", "Crew 2 - St Andrew", "Crew 3 - St Catherine", "Crew 4 - Montego Bay", "Crew 5 - Emergency"]
        d1, d2, d3 = st.columns(3)
        with d1:
            crew = st.selectbox("Assign Crew", crew_options, index=crew_options.index(current.get("crew", "— Unassigned —")) if current.get("crew") in crew_options else 0)
        with d2:
            status = st.selectbox("Update Status", ["🔴 Dispatch Now", "🟡 Investigating", "🚑 Crew Dispatched", "🔧 Crew On Site", "✅ Fixed - Power Restored", "🟢 Monitoring"], index=0)
        with d3:
            eta = st.selectbox("ETA", ["—", "15 mins", "30 mins", "45 mins", "1 hour", "2 hours", "3+ hours"])
        note = st.text_input("Admin Note", value=current.get("note", ""), placeholder="e.g. Transformer blown, need replacement part...")
        if st.button("✅ Save Dispatch", use_container_width=True):
            st.session_state.dispatch[inc_key] = {"crew": crew, "status": status, "eta": eta, "note": note}
            st.success(f"✅ {crew} assigned to {inc_key} - Status: {status} - ETA: {eta}")
            st.rerun()
        if current:
            st.info(f"Current: Crew = {current.get('crew')} | Status = {current.get('status')} | ETA = {current.get('eta')} | Note = {current.get('note')}")
    else:
        st.info("No incidents to dispatch yet.")

    st.divider()
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.subheader("🗺️ Live Map of Jamaica - Power Outages")
        if not brain["map_df"].empty:
            st.map(brain["map_df"], zoom=7)
        else:
            st.info("Map will appear once clusters are found.")
    with col2:
        st.subheader("🤖 AI Assistant - Admin (Live)")
        if "admin_chat" not in st.session_state:
            first_msg = f"Live update: Top is {top_incident['Incident']} with {top_incident['Reports']} reports. Want a dispatch message?" if brain["incidents"] else "No incidents yet."
            st.session_state.admin_chat = [("assistant", first_msg)]
        for role, msg in st.session_state.admin_chat:
            with st.chat_message(role):
                st.write(msg)
        admin_q = st.chat_input("Ask admin AI...", key="admin_input")
        if admin_q:
            st.session_state.admin_chat.append(("user", admin_q))
            st.session_state.admin_chat.append(("assistant", ai_reply(admin_q, "admin", brain)))
            st.rerun()

    st.divider()
    st.caption("Outage Intelligence | LIVE DBSCAN + Critical Zones + Dispatch | Power Only")
    if st.button("🔄 Refresh Brain Now"):
        st.cache_data.clear()
        st.rerun()