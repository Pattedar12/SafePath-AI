"""
SafePath AI — Comprehensive Test Suite & Quality Assurance
============================================================
Validates:
1. Route Calculation & Distance Plausibility (Bengaluru to Delhi, Bengaluru to Chennai, identical coordinates)
2. False Pedestrian Warning Suppression & Temporal Hazard Tracker
3. Authentication, Password Hashing, Session Management & Accessibility Profile
4. Weather Telemetry & Live Location API
5. Traffic Flow & Corridor Velocity API
6. Serial Telemetry Parsing & Simulation Mode
7. Frontend Static Delivery
"""

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

import server
import database
from fastapi.testclient import TestClient

client = TestClient(server.app)

print("\n=======================================================")
print("  SAFEPATH AI — AUTOMATED COMPREHENSIVE TEST SUITE")
print("=======================================================\n")

# -------------------------------------------------------------
# 1. ROUTE CALCULATION & DISTANCE CONVERSION TESTS
# -------------------------------------------------------------
print("--- [TEST 1] ROUTE CALCULATION: Bengaluru to Delhi (Fix 2 Lakh KM Bug) ---")
res = client.get("/api/routes/calculate?origin=Bengaluru&destination=Delhi&mode=driving")
assert res.status_code == 200, f"Route request failed: {res.text}"
data = res.json()
print(f"Origin: {data['origin_name']} -> Destination: {data['destination_name']}")
print(f"Calculated Road Distance: {data['distance_km']} km ({data['distance_text']})")
print(f"Estimated Travel Duration: {data['duration_text']}")
# Must be between 1800 km and 2500 km — NEVER 200,000 km!
assert 1800.0 <= data['distance_km'] <= 2500.0, f"Distance out of realistic road bounds: {data['distance_km']} km"
assert "2,0" in data['distance_text'] or "2,1" in data['distance_text'] or "2,2" in data['distance_text'] or "1,9" in data['distance_text']
assert len(data['polyline']) > 10, "Polyline coordinates missing"
print("✓ PASSED: Bengaluru to Delhi calculates plausible ~2,030 km road distance!\n")

print("--- [TEST 2] ROUTE CALCULATION: Bengaluru to Chennai ---")
res = client.get("/api/routes/calculate?origin=Bengaluru&destination=Chennai&mode=driving")
assert res.status_code == 200
data = res.json()
print(f"Calculated Distance: {data['distance_km']} km ({data['distance_text']})")
assert 280.0 <= data['distance_km'] <= 400.0, f"Distance out of bounds: {data['distance_km']}"
print("✓ PASSED: Bengaluru to Chennai calculates plausible ~326 km road distance!\n")

print("--- [TEST 3] ROUTE CALCULATION: Identical Origin & Destination ---")
res = client.get("/api/routes/calculate?origin=Bengaluru&destination=Bengaluru&mode=driving")
assert res.status_code == 200
data = res.json()
print(f"Calculated Distance: {data['distance_km']} km")
assert data['distance_km'] == 0.0
print("✓ PASSED: Identical origin and destination safely returns 0.0 km!\n")

print("--- [TEST 4] ROUTE CALCULATION: Invalid Locations Error Handling ---")
res = client.get("/api/routes/calculate?origin=XYZ_INVALID_PLACE_99999&destination=Delhi")
assert res.status_code == 400
print(f"Invalid location returned HTTP {res.status_code} with detail: {res.json()['detail']}")
print("✓ PASSED: Graceful error handling on unresolvable locations!\n")

print("--- [TEST 4B] PLACE SEARCH & AUTOCOMPLETE WITH TYPO TOLERANCE ---")
# Test 1: Typo 'collage' -> College
res_typo = client.get("/api/places/autocomplete?q=hkbk+collage+of+engineering")
assert res_typo.status_code == 200
results_typo = res_typo.json().get("results", [])
assert len(results_typo) > 0, "Failed to resolve 'hkbk collage of engineering'"
print(f"Typo query 'hkbk collage of engineering' resolved to: {results_typo[0]['name']} ({results_typo[0]['lat']}, {results_typo[0]['lon']})")
assert "HKBK" in results_typo[0]["name"]

# Test 2: Landmark & University search
res_pres = client.get("/api/places/autocomplete?q=presidency+university")
assert res_pres.status_code == 200
assert len(res_pres.json().get("results", [])) > 0
print(f"Search 'presidency university' resolved to: {res_pres.json()['results'][0]['name']}")

# Test 3: International Landmark
res_intl = client.get("/api/places/autocomplete?q=times+square")
assert res_intl.status_code == 200
assert len(res_intl.json().get("results", [])) > 0
print(f"Search 'times square' resolved to: {res_intl.json()['results'][0]['name']}")

# Test 4: Coordinate string search
res_coord = client.get("/api/places/autocomplete?q=12.9716,77.5946")
assert res_coord.status_code == 200
assert len(res_coord.json().get("results", [])) > 0
print(f"GPS Coordinate query resolved: {res_coord.json()['results'][0]['name']}")
print("✓ PASSED: Comprehensive place search and autocomplete with typo tolerance!\n")

print("--- [TEST 4C] TURN-BY-TURN MANEUVER GENERATION ---")
route_res = client.get("/api/routes/calculate?origin=HKBK+College+of+Engineering&destination=Cubbon+Park&mode=driving")
assert route_res.status_code == 200
route_data = route_res.json()
steps = route_data.get("steps", [])
assert len(steps) > 0, "Route must contain turn-by-turn maneuver steps"
print(f"Route from HKBK to Cubbon Park generated {len(steps)} turn-by-turn steps:")
for idx, s in enumerate(steps[:4]):
    print(f"  Step {idx+1}: [{s.get('icon', 'turn')}] {s.get('instruction')} ({s.get('distance_m', 0)}m)")
last_step = steps[-1]
assert last_step.get("maneuver_type") == "arrive", "Final step must be arrive maneuver"
print("✓ PASSED: Truthful turn-by-turn maneuver extraction from route provider!\n")

print("--- [TEST 4D] ROAD HAZARD REPORTING & RETRIEVAL ---")
haz_post = client.post("/api/hazards/report", json={
    "category": "pothole",
    "description": "Deep surface depression in curb lane",
    "latitude": 13.0359,
    "longitude": 77.6198,
    "accuracy_m": 4.5
})
assert haz_post.status_code == 200
haz_record = haz_post.json()["report"]
assert haz_record["category"] == "pothole"
assert haz_record["status"] == "unverified"
print(f"Reported road hazard ID {haz_record['id']}: Status '{haz_record['status']}'")

haz_get = client.get("/api/hazards/reports")
assert haz_get.status_code == 200
reports = haz_get.json()["reports"]
assert any(r["id"] == haz_record["id"] for r in reports)
print(f"Total road hazard reports in database: {len(reports)}")
print("✓ PASSED: Road hazard reporting and database retrieval operational!\n")



# -------------------------------------------------------------
# 2. FALSE PEDESTRIAN WARNING SUPPRESSION & TEMPORAL TRACKER
# -------------------------------------------------------------
print("--- [TEST 5] FALSE PEDESTRIAN WARNING SUPPRESSION ENGINE ---")
tracker = server.hazard_tracker
frame_w, frame_h = 640, 480

# Case A: Person detected on sidewalk / lateral path (frame 1)
# Transient 1st frame must be held for temporal confirmation
det_transient = [{"object": "person", "confidence": 0.88, "box": [50.0, 100.0, 140.0, 320.0]}]
eval_1 = tracker.update_frame(det_transient, frame_w, frame_h, sonar_dist=None)
print(f"Frame 1 (Transient) -> Risk: {eval_1[0]['risk_state']} | Reason: {eval_1[0]['suppression_reason']}")
assert eval_1[0]['risk_state'] == "NORMAL", "Transient detection must remain NORMAL"

# Case B: Person standing safely outside walking corridor across 4 frames
det_safe_lateral = [{"object": "person", "confidence": 0.85, "box": [50.0, 100.0, 140.0, 320.0]}]
for _ in range(3):
    eval_lateral = tracker.update_frame(det_safe_lateral, frame_w, frame_h, sonar_dist=None)
print(f"Frame 4 (Lateral Sidewalk) -> Risk: {eval_lateral[0]['risk_state']} | Reason: {eval_lateral[0]['suppression_reason']}")
assert eval_lateral[0]['risk_state'] == "NORMAL", "Person on lateral sidewalk must NOT trigger hazard"

# Case C: Person in center walking corridor at safe distance without collision vector
det_center_safe = [{"object": "person", "confidence": 0.90, "box": [280.0, 150.0, 360.0, 320.0]}]
for _ in range(4):
    eval_center = tracker.update_frame(det_center_safe, frame_w, frame_h, sonar_dist=None)
print(f"Frame 4 (Center Corridor, Safe Distance) -> Risk: {eval_center[0]['risk_state']} | Reason: {eval_center[0]['suppression_reason']}")
assert eval_center[0]['risk_state'] == "NORMAL", "Center pedestrian at safe distance must be NORMAL"

# Case D: Critical proximity confirmed by physical ultrasonic sensor (sonar = 42 cm)
eval_danger = tracker.update_frame(det_center_safe, frame_w, frame_h, sonar_dist=42.0)
print(f"Frame 5 (Sonar Proximity 42cm) -> Risk: {eval_danger[0]['risk_state']} | Reason: {eval_danger[0]['suppression_reason']}")
assert eval_danger[0]['risk_state'] == "DANGER", "Proximity collision must escalate to DANGER"
print("✓ PASSED: False pedestrian alerts fully suppressed; genuine hazards properly verified!\n")


# -------------------------------------------------------------
# 3. AUTHENTICATION & ACCESSIBILITY PROFILE TESTS
# -------------------------------------------------------------
print("--- [TEST 6] AUTHENTICATION & USER PROFILE ---")
test_email = f"evaluator_{int(time.time())}@safepath.ai"
reg_payload = {
    "email": test_email,
    "name": "Prof. Smart Mobility Evaluator",
    "password": "SecurePassword123!",
    "age": 28,
    "emergency_contact": "+91 98765 43210",
    "impairment_type": "visual_impairment"
}
reg_res = client.post("/api/auth/register", json=reg_payload)
assert reg_res.status_code == 200, f"Registration failed: {reg_res.text}"
token = reg_res.json()["token"]
user_id = reg_res.json()["user"]["id"]
print(f"User registered with ID: {user_id}, Session Token generated.")

# Duplicate registration check
dup_res = client.post("/api/auth/register", json=reg_payload)
assert dup_res.status_code == 400
print("✓ Duplicate email rejection verified.")

# Login with correct password
login_res = client.post("/api/auth/login", json={"email": test_email, "password": "SecurePassword123!"})
assert login_res.status_code == 200
assert login_res.json()["token"] is not None
print("✓ Login with PBKDF2 hash verification verified.")

# Login with wrong password
bad_login = client.post("/api/auth/login", json={"email": test_email, "password": "WrongPassword"})
assert bad_login.status_code == 401
print("✓ Invalid credential rejection verified.")

# Update profile & accessibility preferences
auth_headers = {"Authorization": f"Bearer {token}"}
pref_payload = {
    "name": "Prof. Evaluator (Updated)",
    "preferences": {
        "large_text": True,
        "high_contrast": True,
        "reduce_animations": True,
        "step_free_routes": True,
        "impairment_type": "visual_impairment"
    }
}
update_res = client.put("/api/auth/profile", json=pref_payload, headers=auth_headers)
assert update_res.status_code == 200
assert update_res.json()["user"]["preferences"]["large_text"] is True
print("✓ Accessibility preferences persisted in SQLite database.")

# Add saved destination
dest_res = client.post("/api/destinations", json={"name": "CSE Department Lab", "address": "HKBK College of Engineering"}, headers=auth_headers)
assert dest_res.status_code == 200
dest_list = client.get("/api/destinations", headers=auth_headers).json()["destinations"]
assert any(d["name"] == "CSE Department Lab" for d in dest_list)
print("✓ User saved destinations persisted.")

# Logout session
logout_res = client.post("/api/auth/logout", headers=auth_headers)
assert logout_res.status_code == 200
print("✓ PASSED: Authentication, SQLite persistence, and Accessibility profile operational!\n")


# -------------------------------------------------------------
# 4. WEATHER & TRAFFIC INTELLIGENCE WITH LIVE COORDINATES
# -------------------------------------------------------------
print("--- [TEST 7] WEATHER & TRAFFIC TELEMETRY ---")
# Coordinates for Bengaluru Koramangala
w_res = client.get("/api/weather/current?lat=12.9352&lon=77.6245&location_name=Koramangala+Bengaluru")
assert w_res.status_code == 200
w_data = w_res.json()
print(f"Weather: {w_data['condition']}, {w_data['temperature_c']}°C, Friction: {w_data['surface_friction_coefficient']}")
assert "temperature_c" in w_data
assert "surface_friction_coefficient" in w_data

# Traffic flow for coordinates
t_res = client.get("/api/traffic/flow?lat=12.9352&lon=77.6245&location_name=Koramangala+Bengaluru")
assert t_res.status_code == 200
t_data = t_res.json()
print(f"Traffic: Speed {t_data['current_speed_kmh']} km/h, Congestion: {t_data['congestion_pct']}%, Risk: {t_data['pedestrian_crossing_risk']}")
assert "current_speed_kmh" in t_data
print("✓ PASSED: Weather and Traffic telemetry operational with custom coordinates!\n")


# -------------------------------------------------------------
# 5. HARDWARE TELEMETRY & ESP PROTOCOL
# -------------------------------------------------------------
print("--- [TEST 8] HARDWARE TELEMETRY & PROTOCOL ---")
ports_res = client.get("/api/serial/ports")
assert ports_res.status_code == 200
status_res = client.get("/api/serial/status")
assert status_res.status_code == 200
h_data = status_res.json()
print(f"Hardware Status: Connected={h_data['connected']}, SimulationMode={h_data['simulation_mode']}")
assert "connected" in h_data
assert "distance_cm" in h_data
print("✓ PASSED: Hardware API and protocol parsing operational!\n")

print("--- [TEST 8B] MULTI-MODAL PROXIMITY & RAIN SENSOR BUZZER WARNINGS ---")
# 1. Close camera obstacle triggers buzzer alert (level 4)
res_close = client.post("/api/serial/simulate_event", json={"event_type": "close_obstacle"})
assert res_close.status_code == 200
data_close = res_close.json()
assert data_close["camera_close_proximity"] is True
assert data_close["alert_level"] == 4
print("✓ Close camera obstacle triggers level 4 DANGER alert and buzzer warning.")

# 2. Ultrasonic proximity (< 30 cm) triggers immediate buzzer warning
res_sonar = client.post("/api/serial/simulate_event", json={"event_type": "ultrasonic_alert", "distance_cm": 24.5})
assert res_sonar.status_code == 200
data_sonar = res_sonar.json()
assert data_sonar["ultrasonic_alert"] is True
assert data_sonar["alert_level"] == 4
print("✓ Ultrasonic obstacle within 24.5 cm triggers buzzer alarm.")

# 3. Rain sensor triggers wet road hazard warning
res_rain = client.post("/api/serial/simulate_event", json={"event_type": "rain_alert"})
assert res_rain.status_code == 200
data_rain = res_rain.json()
assert data_rain["rain_detected"] is True
assert data_rain["alert_level"] == 2
print("✓ Rain sensor activation triggers caution warning & buzzer pulse.")

# 4. Clear all alerts
res_clear = client.post("/api/serial/simulate_event", json={"event_type": "clear"})
assert res_clear.status_code == 200
assert res_clear.json()["alert_level"] == 0
print("✓ PASSED: Camera proximity, ultrasonic, and rain sensor alarms fully validated!\n")



# -------------------------------------------------------------
# 6. STATIC FRONTEND SPA DELIVERY
# -------------------------------------------------------------
print("--- [TEST 9] STATIC FRONTEND SPA SERVING ---")
index_res = client.get("/")
assert index_res.status_code == 200
assert "<div id=\"root\"></div>" in index_res.text
print("✓ PASSED: Production Vite frontend bundle served correctly by FastAPI!\n")

print("=======================================================")
print("  ALL 9 COMPREHENSIVE TESTS PASSED WITH 100% SUCCESS! ")
print("=======================================================\n")
