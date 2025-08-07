import streamlit as st
import pandas as pd
import joblib

# --- Load Model ---
model = joblib.load("G:/GitDemo/Flight_ML_Project/Customer-Satisfaction_1")  # Make sure this is a full pipeline

# --- Page Config ---
st.set_page_config(page_title="Customer Satisfaction", page_icon="🧳", layout="wide")
st.title("🧳 Airline Customer Satisfaction Prediction")
st.markdown("Predict whether a customer is **satisfied** or **neutral/dissatisfied** based on flight experience.")
st.markdown("---")

# --- Input Form ---
with st.form("input_form"):
    st.subheader("👤 Passenger Info & Flight Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        customer_type = st.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
        age = st.slider("Age", 10, 85, 30)
        type_of_travel = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
        travel_class = st.selectbox("Travel Class", ["Eco", "Eco Plus", "Business"])
        flight_distance = st.number_input("Flight Distance (km)", 50, 5000, 500)

    with col2:
        departure_delay = st.number_input("Departure Delay (minutes)", 0, 1000, 0)
        arrival_delay = st.number_input("Arrival Delay (minutes)", 0, 1000, 0)
        gate_location = st.slider("Gate Location", 0, 5, 3)
        baggage_handling = st.slider("Baggage Handling", 0, 5, 3)
        checkin_service = st.slider("Check-in Service", 0, 5, 3)

    with col3:
        departure_convenient = st.slider("Departure/Arrival Time Convenient", 0, 5, 3)
        ease_online_booking = st.slider("Ease of Online Booking", 0, 5, 3)
        online_boarding = st.slider("Online Boarding", 0, 5, 3)
        inflight_wifi = st.slider("Inflight Wifi Service", 0, 5, 3)
        inflight_service = st.slider("Inflight Service", 0, 5, 3)

    st.subheader("✨ Customer Ratings")
    col4, col5, col6 = st.columns(3)

    with col4:
        food_and_drink = st.slider("Food & Drink", 0, 5, 3)
        inflight_entertainment = st.slider("Inflight Entertainment", 0, 5, 3)

    with col5:
        onboard_service = st.slider("On-board Service", 0, 5, 3)
        leg_room_service = st.slider("Leg Room Service", 0, 5, 3)

    with col6:
        seat_comfort = st.slider("Seat Comfort", 0, 5, 3)
        cleanliness = st.slider("Cleanliness", 0, 5, 3)

    submitted = st.form_submit_button("🚀 Predict Satisfaction")

# --- Prediction ---
if submitted:
    input_data = {
        "Gender": gender,
        "Customer Type": customer_type,
        "Age": age,
        "Type of Travel": type_of_travel,
        "Class": travel_class,
        "Flight Distance": flight_distance,
        "Departure Delay in Minutes": departure_delay,
        "Arrival Delay in Minutes": arrival_delay,
        "Gate location": gate_location,
        "Baggage handling": baggage_handling,
        "Checkin service": checkin_service,
        "Departure/Arrival time convenient": departure_convenient,
        "Ease of Online booking": ease_online_booking,
        "Online boarding": online_boarding,
        "Inflight wifi service": inflight_wifi,
        "Inflight service": inflight_service,
        "Food and drink": food_and_drink,
        "Inflight entertainment": inflight_entertainment,
        "On-board service": onboard_service,
        "Leg room service": leg_room_service,
        "Seat comfort": seat_comfort,
        "Cleanliness": cleanliness
    }

    input_df = pd.DataFrame([input_data])

    # Label mapping (ensure it matches your training)
    label_mapping = {
        0: "neutral or dissatisfied",
        1: "satisfied"
    }

    try:
        # Predict
        prediction_numeric = model.predict(input_df)[0]
        prediction_label = label_mapping.get(prediction_numeric, "Unknown")

        # Probability of predicted class
        probability = model.predict_proba(input_df)[0][prediction_numeric] * 100

        st.markdown("### 🧾 Prediction Result")
        if prediction_label == "satisfied":
            st.success(f"✅ Customer is **SATISFIED**  \n🔵 Confidence: `{probability:.2f}%`")
        elif prediction_label == "neutral or dissatisfied":
            st.warning(f"⚠️ Customer is **NEUTRAL or DISSATISFIED**  \n🟠 Confidence: `{probability:.2f}%`")
        else:
            st.error("❌ Prediction label is unknown.")

    except Exception as e:
        st.error(f"❌ Error: {e}")


