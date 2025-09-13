import streamlit as st
import numpy as np
import json
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import uuid
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from supabase import create_client, Client

SUPABASE_URL = "https://pkquxytfpqbuawspslpr.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBrcXV4eXRmcHFidWF3c3BzbHByIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTc1ODUzODUsImV4cCI6MjA3MzE2MTM4NX0.5szbHT2IVEjgnm36nE4yFFhj-uisk9qKE0FX7XlR7XM"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Initialize session state variables ---
if "cart" not in st.session_state:
    st.session_state.cart = []
if "show_cart" not in st.session_state:
    st.session_state.show_cart = False
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "last_suggestions" not in st.session_state:
    st.session_state.last_suggestions = None
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "role" not in st.session_state:
    st.session_state.role = None
if "username" not in st.session_state:
    st.session_state.username = None
if "selected_village" not in st.session_state:
    st.session_state.selected_village = None
# New session states for added features
if "show_field_log" not in st.session_state:
    st.session_state.show_field_log = False
if "show_expert_chat" not in st.session_state:
    st.session_state.show_expert_chat = False
if "show_reminders" not in st.session_state:
    st.session_state.show_reminders = False
if "show_local_resources" not in st.session_state:
    st.session_state.show_local_resources = False
if "filter_organic_only" not in st.session_state:
    st.session_state.filter_organic_only = False
if "price_range" not in st.session_state:
    st.session_state.price_range = [0, 10000]

# --- Helper Functions for Login/Register ---
def login_user(username, password):
    response = supabase.table("users").select("*").eq("username", username).eq("password", password).execute()
    user = response.data[0] if response.data else None
    return user["role"] if user else None

def register_user(username, password, role):
    response = supabase.table("users").select("*").eq("username", username).execute()
    if response.data:
        return False
    supabase.table("users").insert({"username": username, "password": password, "role": role}).execute()
    return True

# --- Order functions ---
def get_orders_for_industry(shop_name):
    response = supabase.table("orders").select("*").eq("shop", shop_name).execute()
    return response.data

def get_orders_for_farmer(farmer_name):
    response = supabase.table("orders").select("*").eq("farmer", farmer_name).execute()
    return response.data

def get_product_details(shop_name, product_name):
    response = supabase.table("products").select("*").eq("product_name", product_name).eq("shop", shop_name).execute()
    return response.data[0] if response.data else None

## --- NEW: Crop-Centric Field Log Functions ---
def save_crop_log_entry(farmer, crop, entry_data):
    data = {
        "farmer": farmer,
        "crop": crop,
        **entry_data
    }
    supabase.table("field_crop_logs").insert(data).execute()

def get_crop_logs(farmer):
    response = supabase.table("field_crop_logs").select("*").eq("farmer", farmer).execute()
    logs = response.data
    # Group logs by crop for UI
    crop_dict = {}
    for log in logs:
        crop = log["crop"]
        crop_dict.setdefault(crop, []).append(log)
    return crop_dict

def delete_crop_log(farmer, crop):
    supabase.table("field_crop_logs").delete().eq("farmer", farmer).eq("crop", crop).execute()

# --- NEW: Expert Chat Functions ---
def save_chat_message(farmer, message, sender_type="farmer", img_path=None):
    data = {
        "farmer": farmer,
        "message": message,
        "sender": sender_type,
        "timestamp": str(datetime.now()),
        "image": img_path
    }
    supabase.table("expert_chats").insert(data).execute()

def get_chat_messages(farmer):
    response = supabase.table("expert_chats").select("*").eq("farmer", farmer).order("timestamp", desc=False).execute()
    return response.data

# --- NEW: Reminders Functions ---
def save_reminder(farmer, reminder_data):
    data = {"farmer": farmer, **reminder_data}
    supabase.table("reminders").insert(data).execute()

def get_reminders(farmer):
    response = supabase.table("reminders").select("*").eq("farmer", farmer).execute()
    return response.data
# --- NEW: Multi-Expert Consultations ---
def load_consultations():
    response = supabase.table("consultations").select("*").execute()
    return {"questions": response.data} if response.data else {"questions": []}

def save_consultations(data):
    # Not needed, handled by add_consultation_question and add_consultation_answer
    pass

def add_consultation_question(farmer, question):
    data = {"id": str(uuid.uuid4())[:8], "farmer": farmer, "question": question, "timestamp": str(datetime.now()), "answers": {}}
    supabase.table("consultations").insert(data).execute()

def add_consultation_answer(question_id, shop_name, answer):
    # Update the answers field for the question
    response = supabase.table("consultations").select("*").eq("id", question_id).execute()
    if response.data:
        q = response.data[0]
        answers = q.get("answers", {})
        answers[shop_name] = {"text": answer, "timestamp": str(datetime.now())}
        supabase.table("consultations").update({"answers": answers}).eq("id", question_id).execute()

# --- NEW: Local Resources Functions ---  
def get_local_dealers(village):
    # Sample local dealer data - you can expand this
    dealers_db = {
        "Palladam": [
            {"name": "Sri Ganesh Agro Store", "contact": "9876543210", "address": "Main Road, Palladam", "products": "Seeds, Fertilizers"},
            {"name": "Farmers Choice", "contact": "9876543211", "address": "Bus Stand Road, Palladam", "products": "Pesticides, Tools"}
        ], 
        "Udumalpet": [
            {"name": "Green Valley Agri Center", "contact": "9876543212", "address": "Market Street, Udumalpet", "products": "Organic Fertilizers"},
            {"name": "Modern Farming Solutions", "contact": "9876543213", "address": "Industrial Area, Udumalpet", "products": "Equipment, Seeds"}
        ], 
        # Add more villages and dealers as needed
    } 
    return dealers_db.get(village, [])

def get_cooperatives(district):
    # Sample cooperative data
    cooperatives_db = {
        "Tiruppur": [
            {"name": "Tiruppur Farmers Cooperative", "contact": "0421-2234567", "services": "Bulk Purchase, Credit"},
            {"name": "Tamil Nadu Agricultural Cooperative", "contact": "0421-2234568", "services": "Insurance, Marketing"}
        ],
        "Erode": [
            {"name": "Erode Agricultural Society", "contact": "0424-2234567", "services": "Seeds, Training"},
            {"name": "Farmers Welfare Association", "contact": "0424-2234568", "services": "Equipment Rental"}
        ],
        "Coimbatore": [
            {"name": "Coimbatore Farmers Union", "contact": "0422-2234567", "services": "Market Linkage"},
            {"name": "Progressive Farmers Group", "contact": "0422-2234568", "services": "Organic Certification"}
        ]
    }
    return cooperatives_db.get(district, [])

# --- Model Loading ---
@st.cache_resource
def load_main_model():
    try:
        model = load_model("final_densenet_model.keras")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_main_model()

# --- Image preprocessing and prediction ---
def preprocess_input(img_array):
    return img_array / 255.0

def simple_leaf_mask(img):
    # Dummy mask for now
    return img

def preprocess_image(img):
    img = simple_leaf_mask(img)
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def predict_disease(img):
    processed_img = preprocess_image(img)
    feedback_map = {}
    try:
        with open("feedback.json") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry.get("image") and entry.get("correct"):
                        feedback_map[entry["image"]] = entry["correct"]
                except Exception:
                    continue
    except Exception:
        pass

    if hasattr(img, "filename") and img.filename in feedback_map:
        return feedback_map[img.filename], 1.0

    if model is None:
        return "Model not loaded", 0.0

    prediction = model.predict(processed_img)
    with open("class_indices.json") as f:
        class_indices = json.load(f)

    index_to_class = {v: k for k, v in class_indices.items()}
    predicted_index = int(np.argmax(prediction))
    predicted_class = index_to_class[predicted_index]
    confidence = float(np.max(prediction))
    return predicted_class, confidence

def normalize_class_name(class_name):
    # normalization rules...
    if class_name.startswith("tomato_Tomato_"):
        return "Tomato__" + class_name.split("tomato_Tomato_")[1]
    if class_name.startswith("tomato_Tomato__"):
        return "Tomato__" + class_name.split("tomato_Tomato__")[1]
    if class_name.startswith("tomato_"):
        return "Tomato__" + class_name.split("tomato_")[1]
    if class_name.startswith("Potato_Potato___"):
        return "Potato__" + class_name.split("Potato_Potato___")[1]
    if class_name.startswith("Potato_Potato__"):
        return "Potato__" + class_name.split("Potato_Potato__")[1]
    if class_name.startswith("Potato_"):
        return "Potato__" + class_name.split("Potato_")[1]
    if class_name.startswith("Pepper_Pepper__bell___"):
        return "Pepper__bell___" + class_name.split("Pepper_Pepper__bell___")[1]
    if class_name.startswith("Pepper_"):
        return "Pepper__bell___" + class_name.split("Pepper_")[1]
    if class_name.startswith("Corn_corn_"):
        return "corn_" + class_name.split("Corn_corn_")[1]
    if class_name.startswith("Chilly_chilly_"):
        return "chilly_" + class_name.split("Chilly_chilly_")[1]
    if class_name.startswith("Cauliflower_cauliflower_"):
        return "cauliflower_" + class_name.split("Cauliflower_cauliflower_")[1]
    return class_name

def get_product_suggestions(disease):
    key = normalize_class_name(disease)
    try:
        with open("products.json") as f:
            products_data = json.load(f)
        return products_data.get(key, [])
    except Exception:
        return []

# --- NEW: Enhanced Product Filtering ---
def filter_products(products, organic_only=False, price_min=0, price_max=10000, min_rating=0):
    filtered = []
    for product in products:
        # Price filter
        if not (price_min <= product.get("price", 0) <= price_max):
            continue
        
        # Rating filter
        if product.get("rating", 0) < min_rating:
            continue
            
        # Organic filter
        if organic_only and "organic" not in product.get("product_name", "").lower():
            continue
            
        filtered.append(product)
    
    return filtered

# --- Multi-language Support (Enhanced) ---
LANGS = ["English", "தமிழ்", "हिन्दी", "తెలుగు", "ಕನ್ನಡ", "മലയാളം"]
TEXT = {
    "English": {
        "title": "🌿 Plant Disease Detection & Fertilizer Recommendation",
        "login": "User Login / Register",
        "cart": "Cart",
        "feedback": "Wrong prediction? Submit correct disease",
        "submit": "Submit Correction",
        "thanks": "Thank you for your feedback!",
        "progress": "Disease Progression Tracking",
        "dashboard": "Open Dashboard",
        "field_log": "Field Log & Analytics",
        "expert_chat": "Expert Consultation",
        "reminders": "Reminders & Alerts",
        "local_resources": "Local Resources",
        "filter_products": "Filter Products"
    },
    "தமிழ்": {
        "title": "🌿 தாவர நோய் கண்டறிதல் & உர பரிந்துரை",
        "login": "பயனர் உள்நுழைவு / பதிவு",
        "cart": "வண்டி",
        "feedback": "தவறான கணிப்பு? சரியான நோயை சமர்ப்பிக்கவும்",
        "submit": "சரிசெய்தல் சமர்ப்பிக்கவும்",
        "thanks": "உங்கள் கருத்துக்கு நன்றி!",
        "progress": "நோய் முன்னேற்ற கண்காணிப்பு",
        "dashboard": "டாஷ்போர்டை திறக்கவும்",
        "field_log": "வயல் பதிவு & பகுப்பாய்வு",
        "expert_chat": "நிபுணர் ஆலோசனை",
        "reminders": "நினைவூட்டல்கள்",
        "local_resources": "உள்ளூர் வளங்கள்",
        "filter_products": "தயாரிப்புகளை வடிகட்டவும்"
    },
    "हिन्दी": {
        "title": "🌿 पौध रोग पहचान और उर्वरक सिफारिश",
        "login": "यूज़र लॉगिन / रजिस्टर",
        "cart": "कार्ट",
        "feedback": "गलत भविष्यवाणी? सही रोग दर्ज करें",
        "submit": "सुधार सबमिट करें",
        "thanks": "आपकी प्रतिक्रिया के लिए धन्यवाद!",
        "progress": "रोग प्रगति ट्रैकिंग",
        "dashboard": "डैशबोर्ड खोलें",
        "field_log": "खेत लॉग और विश्लेषण",
        "expert_chat": "विशेषज्ञ सलाह",
        "reminders": "रिमाइंडर और अलर्ट",
        "local_resources": "स्थानीय संसाधन",
        "filter_products": "उत्पाद फ़िल्टर करें"
    },
    "తెలుగు": {
        "title": "🌿 మొక్కల వ్యాధి గుర్తింపు & ఎరువుల సిఫారసు",
        "login": "వినియోగదారు లాగిన్ / నమోదు",
        "cart": "బండి",
        "feedback": "తప్పు అంచనా? సరైన వ్యాధిని సమర్పించండి",
        "submit": "దిద్దుబాటు సమర్పించండి",
        "thanks": "మీ అభిప్రాయానికి ధన్యవాదాలు!",
        "progress": "వ్యాధి పురోగతి ట్రాకింగ్",
        "dashboard": "డాష్‌బోర్డ్ తెరవండి",
        "field_log": "పొల లాగ్ & విశ్లేషణ",
        "expert_chat": "నిపుణుల సలహా",
        "reminders": "రిమైండర్లు & అలర్ట్లు",
        "local_resources": "స్థానిక వనరులు",
        "filter_products": "ఉత్పత్తులను ఫిల్టర్ చేయండి"
    },
    "ಕನ್ನಡ": {
        "title": "🌿 ಸಸ್ಯ ರೋಗ ಪತ್ತೆ & ರಸಗೊಬ್ಬರ ಶಿಫಾರಸು",
        "login": "ಬಳಕೆದಾರರ ಲಾಗಿನ್ / ನೋಂದಣಿ",
        "cart": "ಕಾರ್ಟ್",
        "feedback": "ತಪ್ಪು ಮುನ್ಸೂಚನೆ? ಸರಿಯಾದ ರೋಗವನ್ನು ಸಲ್ಲಿಸಿ",
        "submit": "ತಿದ್ದುಪಡಿ ಸಲ್ಲಿಸಿ",
        "thanks": "ನಿಮ್ಮ ಪ್ರತಿಕ್ರಿಯೆಗೆ ಧನ್ಯವಾದಗಳು!",
        "progress": "ರೋಗ ಪ್ರಗತಿ ಟ್ರ್ಯಾಕಿಂಗ್",
        "dashboard": "ಡ್ಯಾಶ್‌ಬೋರ್ಡ್ ತೆರೆಯಿರಿ",
        "field_log": "ಹೊಲ ಲಾಗ್ & ವಿಶ್ಲೇಷಣೆ",
        "expert_chat": "ತಜ್ಞರ ಸಲಹೆ",
        "reminders": "ರಿಮೈಂಡರ್‌ಗಳು & ಅಲರ್ಟ್‌ಗಳು",
        "local_resources": "ಸ್ಥಳೀಯ ಸಂಪನ್ಮೂಲಗಳು",
        "filter_products": "ಉತ್ಪನ್ನಗಳನ್ನು ಫಿಲ್ಟರ್ ಮಾಡಿ"
    },
    "മലയാളം": {
        "title": "🌿 സസ്യ രോഗനിർണയം & രാസവള ശുപാർശ",
        "login": "ഉപയോക്താവ് ലോഗിൻ / രജിസ്റ്റർ",
        "cart": "കാർട്ട്",
        "feedback": "തെറ്റായ പ്രവചനം? ശരിയായ രോഗം സമർപ്പിക്കുക",
        "submit": "തിരുത്തൽ സമർപ്പിക്കുക",
        "thanks": "നിങ്ങളുടെ ഫീഡ്‌ബാക്കിന് നന്ദി!",
        "progress": "രോഗ പുരോഗതി ട്രാക്കിംഗ്",
        "dashboard": "ഡാഷ്‌ബോർഡ് തുറക്കുക",
        "field_log": "ഫീൽഡ് ലോഗ് & അനാലിറ്റിക്സ്",
        "expert_chat": "വിദഗ്ദ്ധ കൺസൾട്ടേഷൻ",
        "reminders": "റിമൈൻഡറുകൾ & അലേർട്ടുകൾ",
        "local_resources": "പ്രാദേശിക വിഭവങ്ങൾ",
        "filter_products": "ഉൽപ്പന്നങ്ങൾ ഫിൽട്ടർ ചെയ്യുക"
    }
}

lang = st.sidebar.selectbox("Language", LANGS)

# --- App setup ---
st.set_page_config(layout="wide")
st.markdown("<span style='font-size:1.5em; font-weight:bold;'>CropCure Connect: An AI driven agricultural health management system</span>", unsafe_allow_html=True)

# --- Weather Data ---
def get_weather(lat, lon, api_key):
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        resp = requests.get(url)
        data = resp.json()
        temp = data['main']['temp']
        humidity = data['main']['humidity']
        wind_speed = data['wind']['speed']
        cond = data['weather'][0]['main']
        return temp, humidity, wind_speed, cond
    except Exception:
        return None, None, None, None

VILLAGE_COORDS = {
    "Palladam": (10.9911, 77.2865),
    "Udumalpet": (10.5881, 77.2478),
    "Avinashi": (11.1926, 77.2686),
    "Kangeyam": (11.0066, 77.5607),
    "Dharapuram": (10.7383, 77.5322),
    "Bhavani": (11.4456, 77.6845),
    "Gobichettipalayam": (11.4551, 77.4422),
    "Perundurai": (11.2756, 77.5871),
    "Sathyamangalam": (11.5052, 77.2386),
    "Kodumudi": (11.0772, 77.8726),
    "Mettupalayam": (11.2997, 76.9346),
    "Pollachi": (10.6583, 77.0085),
    "Sulur": (11.0246, 77.1255),
    "Annur": (11.2317, 77.1065),
    "Kinathukadavu": (10.8242, 77.0212)
}

location_data = {
    "Tiruppur": ["Palladam", "Udumalpet", "Avinashi", "Kangeyam", "Dharapuram"],
    "Erode": ["Bhavani", "Gobichettipalayam", "Perundurai", "Sathyamangalam", "Kodumudi"],
    "Coimbatore": ["Mettupalayam", "Pollachi", "Sulur", "Annur", "Kinathukadavu"]
}

api_key = "5c61491035b96d81f96d295570493ef3"  # Replace with valid key
lat, lon = 11.0, 77.0

if st.session_state.get("logged_in") and st.session_state.get("role") == "farmer":
    selected_village = st.session_state.get("selected_village")
    if selected_village and selected_village in VILLAGE_COORDS:
        lat, lon = VILLAGE_COORDS[selected_village]

temp, humidity, wind_speed, cond = get_weather(lat, lon, api_key)

if None not in (temp, humidity, wind_speed, cond):
    col1, col2, col3, col4 = st.columns([1,1,1,2])
    with col4:
        st.markdown(f"##### 🌡️ {temp}°C  💧 {humidity}%  🌬️ {wind_speed} m/s  ☁️ {cond}")
else:
    st.warning("Weather data not available for this location or API key.")

# --- NEW: Feature Navigation Buttons for Farmers ---
if st.session_state.logged_in and st.session_state.role == "farmer":
    # Feature toggle buttons
    st.markdown("### 🔧 Quick Access")
    nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns(5)
    
    with nav_col1:
        if st.button(f"📊 {TEXT[lang]['field_log']}", key="toggle_field_log"):
            st.session_state.show_field_log = not st.session_state.show_field_log
    
    with nav_col2:
        if st.button(f"👨‍🌾 {TEXT[lang]['expert_chat']}", key="toggle_expert_chat"):
            st.session_state.show_expert_chat = not st.session_state.show_expert_chat
    
    with nav_col3:
        if st.button(f"⏰ {TEXT[lang]['reminders']}", key="toggle_reminders"):
            st.session_state.show_reminders = not st.session_state.show_reminders
    
    with nav_col4:
        if st.button(f"🏪 {TEXT[lang]['local_resources']}", key="toggle_local_resources"):
            st.session_state.show_local_resources = not st.session_state.show_local_resources
    
    with nav_col5:
        cart_count = len(st.session_state.cart)
        if st.button(f"🛒 {TEXT[lang]['cart']}: {cart_count}", key="open_cart_btn"):
            st.session_state.show_cart = True

# --- Cart Sidebar for farmers ---
if st.session_state.logged_in and st.session_state.role == "farmer":
    if st.session_state.show_cart:
        with st.sidebar:
            st.header(f"🧺 {TEXT[lang]['cart']}")
            if st.session_state.cart:
                total = sum(item["price"] for item in st.session_state.cart)
                for i, item in enumerate(st.session_state.cart):
                    st.markdown(f"- **{item['product_name']}** from {item['shop']} — ₹{item['price']}")
                st.markdown(f"### 🧾 Total: ₹{total}")
                if st.button("💳 Pay & Confirm Order", key="pay_sidebar"):
                    try:
                        with open("orders.json") as f:
                            orders_data = json.load(f)
                    except Exception:
                        orders_data = {"orders": []}
                    for item in st.session_state.cart:
                        order = {
                            "order_id": str(uuid.uuid4())[:8],
                            "farmer": st.session_state.username,
                            "shop": item["shop"],
                            "product": item["product_name"],
                            "quantity": 1,
                            "status": "pending",
                            "description": "",
                            "timestamp": str(datetime.now())
                        }
                        orders_data["orders"].append(order)
                    with open("orders.json", "w") as f:
                        json.dump(orders_data, f, indent=2)
                    st.success("✅ Order placed successfully!")
                    st.session_state.cart.clear()
                    st.session_state.show_cart = False
                    st.rerun()
            else:
                st.info("Your cart is empty. Add products to proceed.")
            if st.button("Close Cart", key="close_cart_btn"):
                st.session_state.show_cart = False

# --- Login/Register Sidebar ---
st.sidebar.header(TEXT[lang]["login"])
mode = st.sidebar.radio("Mode", ["Login", "Register"])

if mode == "Register":
    reg_role = st.sidebar.selectbox("Role", ["farmer", "industry"])
    reg_username = st.sidebar.text_input("Username", key="reg_username_sidebar")
    reg_password = st.sidebar.text_input("Password", type="password", key="reg_password_sidebar")
    if st.sidebar.button("Register", key="reg_button"):
        success = register_user(reg_username, reg_password, reg_role)
        if success:
            st.success("Registration successful! You can now login.")
            st.rerun()
        else:
            st.error("Username already exists.")

elif mode == "Login":
    username = st.sidebar.text_input("Username", key="login_username_sidebar")
    password = st.sidebar.text_input("Password", type="password", key="login_password_sidebar")
    col_login2, col_logout2 = st.sidebar.columns([1,1])
    if col_login2.button("Login"):
        role = login_user(username, password)
        if role:
            st.session_state.logged_in = True
            st.session_state.role = role
            st.session_state.username = username
            st.success(f"Logged in as {role}!")
            st.rerun()
        else:
            st.error("Invalid credentials.")
    if col_logout2.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.role = None
        st.session_state.username = None
        st.success("Logged out successfully.")
        st.rerun()

# --- Main Application UI ---
if st.session_state.logged_in:
    if st.session_state.role == "farmer":
        
        # NEW FEATURE 1: Expert Consultation & Advisory Chat
        if st.session_state.show_expert_chat:
            st.markdown("---")
            st.subheader(f"👨‍🌾 {TEXT[lang]['expert_chat']}")
            
            # Display chat messages with image support
            chat_messages = get_chat_messages(st.session_state.username)
            if chat_messages:
                st.markdown("### 💬 Chat History")
                for msg in chat_messages[-10:]:
                    text = f"**You:** {msg['message']} _{msg['timestamp'][:19]}_"
                    if msg.get("image"):
                        st.image(msg["image"], caption="Attached Image", use_column_width=True)
                    if msg["sender"] == "farmer":
                        st.markdown(text)
                    else:
                        st.markdown(f"**Expert:** {msg['message']} _{msg['timestamp'][:19]}_")
 
            # Message input with image upload
            new_message = st.text_area("Ask an expert:", key="expert_message")
            chat_image = st.file_uploader("Attach Image for Expert (optional)", type=["jpg", "jpeg", "png"], key="expert_chat_image")
            if st.button("Send Message", key="send_expert_msg"):
                if new_message.strip():
                    img_path = None
                    if chat_image:
                        img_filename = f"expert_chat_{st.session_state.username}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
                        img_path = img_filename
                        with open(img_filename, "wb") as f:
                            f.write(chat_image.read())
                    save_chat_message(st.session_state.username, new_message, "farmer", img_path)
                    st.success("Message sent to expert!")
                    st.rerun()
            
            # FAQ Section
            st.markdown("### ❓ Frequently Asked Questions")
            faq_expander = st.expander("Common Disease Questions")
            with faq_expander:
                st.markdown("""
                **Q: How often should I spray fungicides?**
                A: Generally every 7-14 days during disease season, depending on weather conditions.
                
                **Q: Can I mix different pesticides?**
                A: Only mix compatible chemicals. Always read labels or consult an expert.
                
                **Q: What's the best time to spray?**
                A: Early morning or late evening when temperatures are cooler.
                """)
        
        # NEW FEATURE 3: Manual Field Log & Analytics Dashboard
        if st.session_state.show_field_log:
            st.markdown("---")
            st.subheader(f"📊 {TEXT[lang]['field_log']}")
            
            # --- Crop-centric field log UI ---
            crop_logs = get_crop_logs(st.session_state.username)
            cultivated_crops = list(crop_logs.keys())

            if cultivated_crops:
                st.markdown("### 🪴 Cultivated Crops")
                for crop in cultivated_crops:
                    st.markdown(f"**{crop}**")
                    if st.button(f"Delete crop {crop}", key=f"del_crop_{crop}"):
                        delete_crop_log(st.session_state.username, crop)
                        st.success(f"Crop {crop} deleted.")
                        st.rerun()
                    logs = crop_logs[crop]
                    df = pd.DataFrame(logs)
                    if not df.empty:
                        df['date'] = pd.to_datetime(df['date'])
                        st.markdown(f"#### 📋 {crop} Log Entries")
                        st.dataframe(df[['date', 'activity', 'area', 'cost', 'notes']], use_container_width=True)
                        total_cost = df['cost'].sum()
                        st.info(f"**Total Cost for {crop}: ₹{total_cost}**")
            else:
                st.info("No crops found. Add a new crop log below.")

            # Add new crop log entry form
            with st.expander("➕ Add New Crop Log Entry"):
                log_date = st.date_input("Date", key="log_date_crop")
                crop_type = st.selectbox("Crop", ["Tomato", "Potato", "Pepper", "Corn", "Chilly", "Cauliflower"], key="log_crop_type")
                activity = st.selectbox("Activity", ["Planting", "Spraying", "Fertilizing", "Harvesting", "Disease Observation"], key="log_crop_activity")
                area = st.number_input("Area (acres)", min_value=0.1, key="log_crop_area")
                cost = st.number_input("Cost (₹)", min_value=0, key="log_crop_cost")
                notes = st.text_area("Notes", key="log_crop_notes")
                if st.button("Save Crop Log Entry", key="save_crop_log"):
                    log_entry = {
                        "date": str(log_date),
                        "activity": activity,
                        "area": area,
                        "cost": cost,
                        "notes": notes,
                        "timestamp": str(datetime.now())
                    }
                    save_crop_log_entry(st.session_state.username, crop_type, log_entry)
                    st.success(f"Log entry added for crop {crop_type}!")
                    st.rerun()
        
        # NEW FEATURE 4: Personalized Alerts & Reminders
        if st.session_state.show_reminders:
            st.markdown("---")
            st.subheader(f"⏰ {TEXT[lang]['reminders']}")
            
            # Add new reminder
            with st.expander("➕ Set New Reminder"):
                col1, col2 = st.columns(2)
                with col1:
                    reminder_date = st.date_input("Reminder Date", key="reminder_date")
                    reminder_time = st.time_input("Reminder Time", key="reminder_time")
                    reminder_type = st.selectbox("Type", ["Spraying", "Fertilizing", "Harvesting", "Field Inspection", "Other"], key="reminder_type")
                
                with col2:
                    reminder_title = st.text_input("Title", key="reminder_title")
                    reminder_desc = st.text_area("Description", key="reminder_desc")
                
                if st.button("Set Reminder", key="save_reminder"):
                    reminder_data = {
                        "date": str(reminder_date),
                        "time": str(reminder_time),
                        "type": reminder_type,
                        "title": reminder_title,
                        "description": reminder_desc,
                        "created": str(datetime.now()),
                        "id": str(uuid.uuid4())[:8]
                    }
                    save_reminder(st.session_state.username, reminder_data)
                    st.success("Reminder set successfully!")
                    st.rerun()
            
            # Display upcoming reminders
            reminders = get_reminders(st.session_state.username)
            if reminders:
                today = datetime.now().date()
                upcoming = [r for r in reminders if datetime.strptime(r['date'], '%Y-%m-%d').date() >= today]
                upcoming.sort(key=lambda x: x['date'])
                
                st.markdown("### 🔔 Upcoming Reminders")
                for reminder in upcoming[:5]:  # Show next 5 reminders
                    rem_date = datetime.strptime(reminder['date'], '%Y-%m-%d').date()
                    days_left = (rem_date - today).days
                    
                    if days_left == 0:
                        urgency = "🔴 TODAY"
                    elif days_left == 1:
                        urgency = "🟡 TOMORROW"
                    elif days_left <= 7:
                        urgency = f"🟢 {days_left} days"
                    else:
                        urgency = f"📅 {days_left} days"
                    
                    st.info(f"**{urgency}** - {reminder['type']}: {reminder['title']} at {reminder['time']}")
        
        # NEW FEATURE 8: Local Resource Integration
        if st.session_state.show_local_resources:
            st.markdown("---")
            st.subheader(f"🏪 {TEXT[lang]['local_resources']}")
            
            if st.session_state.selected_village:
                # Local dealers
                st.markdown("### 🛍️ Nearby Input Dealers")
                dealers = get_local_dealers(st.session_state.selected_village)
                if dealers:
                    for dealer in dealers:
                        st.markdown(f"""
                        **{dealer['name']}**  
                        📞 {dealer['contact']}  
                        📍 {dealer['address']}  
                        🛒 Products: {dealer['products']}
                        """)
                        st.markdown("---")
                else:
                    st.info("No local dealers found for your village. Try nearby areas.")
                
                # Cooperatives
                if 'district' in locals():
                    st.markdown("### 🤝 Farmer Cooperatives")
                    cooperatives = get_cooperatives(district)
                    if cooperatives:
                        for coop in cooperatives:
                            st.markdown(f"""
                            **{coop['name']}**  
                            📞 {coop['contact']}  
                            🎯 Services: {coop['services']}
                            """)
                            st.markdown("---")
                
                # Market prices (sample data)
                st.markdown("### 💰 Market Prices")
                sample_prices = {
                    "Tomato": "₹25/kg",
                    "Potato": "₹18/kg", 
                    "Pepper": "₹45/kg",
                    "Corn": "₹22/kg"
                }
                
                price_col1, price_col2 = st.columns(2)
                for i, (crop, price) in enumerate(sample_prices.items()):
                    if i % 2 == 0:
                        price_col1.metric(crop, price)
                    else:
                        price_col2.metric(crop, price)
            else:
                st.warning("Please select your village to see local resources.")

        # Farmer dashboard - existing functionality
        st.subheader("📦 Order Updates from Industry")
        try:
            with open("orders.json") as f:
                all_orders = json.load(f).get("orders", [])
            my_orders = [o for o in all_orders if o["farmer"] == st.session_state.username]
            for o in my_orders:
                status = o.get("status", "")
                info_msg = f"Order {o['order_id']}: {o['product']} from {o['shop']} — Status: **{status}**"
                if status == "delivered":
                    info_msg += " ✅ Delivered!"
                elif status == "out for delivery":
                    if st.button(f"Confirm Delivery (Order {o['order_id']})", key=f"confirm_{o['order_id']}"):
                        for order in all_orders:
                            if order["order_id"] == o["order_id"]:
                                order["status"] = "delivered"
                        with open("orders.json", "w") as f:
                            json.dump({"orders": all_orders}, f, indent=2)
                        st.success(f"Order {o['order_id']} marked as delivered!")
                        st.session_state.show_cart = False
                        st.rerun()
                if desc := o.get("description"):
                    info_msg += f"\n\n**Industry Note:** {desc}"
                st.info(info_msg)
        except Exception:
            pass
                # --- NEW: Multi‑Expert Q&A for Farmers ---
        st.markdown("---")
        st.subheader("🧑‍🌾 Ask All Three Shops")
        farmer_q = st.text_area("Enter your question for all shops:", key="multi_q")
        if st.button("Send Question to All Shops"):
            if farmer_q.strip():
                add_consultation_question(st.session_state.username, farmer_q.strip())
                st.success("Your question has been sent to all shops!")
                st.rerun()
        
        consults = load_consultations()
        my_qs = [q for q in consults["questions"] if q["farmer"] == st.session_state.username]
        if my_qs:
            st.markdown("### 📜 Your Questions & Answers")
            for q in sorted(my_qs, key=lambda x: x["timestamp"], reverse=True):
                st.markdown(f"**Q:** {q['question']}  \n_Asked {q['timestamp'][:19]}_")
                for shop in ["shop1", "shop2", "shop3"]:
                    ans = q["answers"].get(shop)
                    if ans: st.success(f"**{shop}**: {ans['text']} (_{ans['timestamp'][:19]}_)")
                    else: st.info(f"**{shop}**: Pending...")
                st.markdown("---")


        # Location and Crop selection
        st.subheader("📍 Location & Crop Selection")
        district = st.selectbox("Select District", list(location_data.keys()), key="district")
        village = st.selectbox("Select Village", location_data[district], key="village")
        st.session_state.selected_village = village
        crop = st.selectbox("Select Crop", ["cauliflower", "chilly", "corn", "Pepper__bell", "Potato", "Tomato"])

        # Image upload and disease prediction
        st.subheader("📷 Upload Crop Image (try to upload separate image of crop's leaf)")
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Image", width=300)
            if st.button("Predict Disease"):
                if model is None:
                    st.error("Model not loaded. Please check model file.")
                else:
                    disease, confidence = predict_disease(img)
                    st.success(f"✅ Predicted Disease: **{disease}** ({confidence*100:.2f}% confidence)")
                    st.session_state.last_prediction = (disease, confidence)
                    st.session_state.last_suggestions = get_product_suggestions(disease)

        if st.session_state.last_prediction:
            disease, confidence = st.session_state.last_prediction
            st.success(f"✅ Predicted Disease: **{disease}** ({confidence*100:.2f}% confidence)")
            suggestions = st.session_state.last_suggestions

            if suggestions:
                st.subheader(f"🛒 Product Recommendations")
                
                # NEW FEATURE 7: Manual Recommendation Customization
                st.markdown("### 🔧 Filter Products")
                filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
                
                with filter_col1:
                    organic_only = st.checkbox("Organic Only", key="organic_filter")
                
                with filter_col2:
                    min_rating = st.slider("Minimum Rating", 0.0, 5.0, 0.0, 0.5, key="rating_filter")
                
                with filter_col3:
                    price_min = st.number_input("Min Price (₹)", min_value=0, value=0, key="price_min")
                
                with filter_col4:
                    price_max = st.number_input("Max Price (₹)", min_value=0, value=10000, key="price_max")
                
                # Apply filters
                filtered_suggestions = filter_products(suggestions, organic_only, price_min, price_max, min_rating)
                
                # Sort options
                sort_option = st.radio("Sort by", ["Price", "Rating", "Shop Name"])
                if sort_option == "Price":
                    sorted_items = sorted(filtered_suggestions, key=lambda x: x["price"])
                elif sort_option == "Rating":
                    sorted_items = sorted(filtered_suggestions, key=lambda x: x["rating"], reverse=True)
                else:
                    sorted_items = sorted(filtered_suggestions, key=lambda x: x["shop"])
                
                st.markdown(f"**Showing {len(sorted_items)} products (filtered from {len(suggestions)})**")
                
                # Product comparison tool
                if len(sorted_items) >= 2:
                    st.markdown("### ⚖️ Product Comparison")
                    compare_col1, compare_col2 = st.columns(2)
                    
                    with compare_col1:
                        product1_idx = st.selectbox("Product 1", range(len(sorted_items)), 
                                                   format_func=lambda x: sorted_items[x]['product_name'], key="compare1")
                    
                    with compare_col2:
                        product2_idx = st.selectbox("Product 2", range(len(sorted_items)), 
                                                   format_func=lambda x: sorted_items[x]['product_name'], key="compare2")
                    
                    if product1_idx != product2_idx:
                        prod1, prod2 = sorted_items[product1_idx], sorted_items[product2_idx]
                        
                        comparison_data = {
                            "Feature": ["Product Name", "Shop", "Price (₹)", "Rating", "Dosage", "Duration"],
                            "Product 1": [prod1['product_name'], prod1['shop'], prod1['price'], 
                                         prod1['rating'], prod1['dosage'], prod1['duration']],
                            "Product 2": [prod2['product_name'], prod2['shop'], prod2['price'], 
                                         prod2['rating'], prod2['dosage'], prod2['duration']]
                        }
                        
                        comparison_df = pd.DataFrame(comparison_data)
                        st.dataframe(comparison_df, use_container_width=True)

                # Display products
                for i, item in enumerate(sorted_items[:5]):  # Show top 5 after filtering
                    with st.container():
                        prod_col1, prod_col2 = st.columns([3, 1])
                        
                        with prod_col1:
                            organic_badge = "🌱 ORGANIC" if "organic" in item['product_name'].lower() else ""
                            st.markdown(f"""
                            **{item['product_name']}** {organic_badge}  
                            🏪 Shop: {item['shop']}  
                            📦 Dosage: {item['dosage']} / {item['duration']}  
                            💸 Price: ₹{item['price']}  
                            ⭐ Rating: {item['rating']}/5.0  
                            """)
                        
                        with prod_col2:
                            if st.button(f"Add to Cart", key=f"add_{i}"):
                                st.session_state.cart.append(item)
                                st.success(f"Added {item['product_name']} to cart.")
                                st.rerun()
                        
                        st.markdown("---")
            else:
                st.warning("No products available for this disease.")

            # Feedback section
            st.markdown(f"**{TEXT[lang]['feedback']}**")
            correct = st.text_input("Enter correct disease name")
            if st.button(TEXT[lang]['submit']):
                with open("feedback.json", "a") as f:
                    f.write(json.dumps({
                        "image": uploaded_file.name if uploaded_file else None,
                        "predicted": disease,
                        "correct": correct
                    }) + "\n")
                st.success(TEXT[lang]['thanks'])
                st.session_state.last_prediction = (correct, 1.0)

    elif st.session_state.role == "industry":
        # Industry dashboard - existing functionality with enhancements
        st.subheader("🏭 Stock Management")
        try:
            with open("products.json") as f:
                all_products = json.load(f)
        except Exception:
            all_products = {}

        industry_products = []
        for cat, prods in all_products.items():
            for prod in prods:
                if prod.get("shop") == st.session_state.username:
                    prod_copy = prod.copy()
                    prod_copy["category"] = cat
                    industry_products.append(prod_copy)

        st.write("### Your Products")
        for i, prod in enumerate(industry_products):
            organic_badge = "🌱" if "organic" in prod.get('product_name', '').lower() else ""
            st.write(f"**{prod.get('product_name', 'Unnamed')}** {organic_badge} | Dosage: {prod.get('dosage', '-')}, Duration: {prod.get('duration', '-')}, Price: ₹{prod.get('price', '-')}, Rating: {prod.get('rating', '-')}, Category: {prod['category']}")
            
            edit_col, del_col = st.columns([1,1])
            if edit_col.button(f"Edit {prod.get('product_name', 'Unnamed')}", key=f"edit_{i}"):
                new_name = st.text_input("Product Name", value=prod.get('product_name', ''), key=f"edit_name_{i}")
                new_dosage = st.text_input("Dosage", value=prod.get('dosage', ''), key=f"edit_dosage_{i}")
                new_duration = st.text_input("Duration", value=prod.get('duration', ''), key=f"edit_duration_{i}")
                new_price = st.number_input("Price", value=prod.get('price', 0), key=f"edit_price_{i}")
                new_rating = st.number_input("Rating", value=prod.get('rating', 0), key=f"edit_rating_{i}")
                
                if st.button("Save Changes", key=f"save_{i}"):
                    for p in all_products[prod['category']]:
                        if p.get('product_name') == prod.get('product_name') and p.get('shop') == st.session_state.username:
                            p['product_name'] = new_name
                            p['dosage'] = new_dosage
                            p['duration'] = new_duration
                            p['price'] = new_price
                            p['rating'] = new_rating
                    with open("products.json", "w") as f:
                        json.dump(all_products, f, indent=2)
                    st.success("Product updated!")
                    st.rerun()
            
            if del_col.button(f"Delete {prod.get('product_name', 'Unnamed')}", key=f"del_{i}"):
                all_products[prod['category']] = [p for p in all_products[prod['category']] if not (p.get('product_name')==prod.get('product_name') and p.get('shop') == st.session_state.username)]
                with open("products.json", "w") as f:
                    json.dump(all_products, f, indent=2)
                st.success("Product deleted!")
                st.rerun()

        st.write("### Add New Product")
        with st.form("add_product_form", clear_on_submit=True):
            category = st.text_input("Category (match class name)")
            name = st.text_input("Product Name")
            dosage = st.text_input("Dosage")
            duration = st.text_input("Duration")
            price = st.number_input("Price", min_value=0)
            rating = st.number_input("Rating", min_value=0.0, max_value=5.0, step=0.1)
            submitted = st.form_submit_button("Add Product")
            
            if submitted:
                new_prod = {
                    "product_name": name,
                    "dosage": dosage,
                    "duration": duration,
                    "price": price,
                    "rating": rating,
                    "shop": st.session_state.username
                }
                if category in all_products:
                    all_products[category].append(new_prod)
                else:
                    all_products[category] = [new_prod]
                with open("products.json", "w") as f:
                    json.dump(all_products, f, indent=2)
                st.success("Product added!")
                st.rerun()

        st.subheader("📊 Order Management")
        try:
            with open("orders.json") as f:
                orders_data = json.load(f)
                orders = [o for o in orders_data.get("orders", []) if o.get("shop") == st.session_state.username]
        except Exception:
            orders = []

        if orders:
            for order in orders:
                st.markdown(
                    f"**Order ID:** {order['order_id']}  \n"
                    f"**Product:** {order['product']}  \n"
                    f"**Farmer:** {order['farmer']}  \n"
                    f"**Quantity:** {order['quantity']}  \n"
                    f"**Status:** {order['status']}"
                )
                
                # Show action buttons based on order status
                if order["status"] == "pending":
                    desc = st.text_area(
                        f"Description for Farmer (Order {order['order_id']})",
                        key=f"desc_{order['order_id']}"
                    )
                    if st.button(
                        f"Mark as Processing (Order {order['order_id']})",
                        key=f"proc_{order['order_id']}"
                    ):
                        for o in orders_data["orders"]:
                            if o["order_id"] == order["order_id"]:
                                o["status"] = "processing"
                                o["description"] = desc
                        with open("orders.json", "w") as f:
                            json.dump(orders_data, f, indent=2)
                        st.success("Order marked as processing.")
                        st.rerun()
                
                elif order["status"] == "processing":
                    if st.button(
                        f"Mark as Out For Delivery (Order {order['order_id']})",
                        key=f"out_{order['order_id']}"
                    ):
                        for o in orders_data["orders"]:
                            if o["order_id"] == order["order_id"]:
                                o["status"] = "out for delivery"
                        with open("orders.json", "w") as f:
                            json.dump(orders_data, f, indent=2)
                        st.success("Order marked as out for delivery.")
                        st.rerun()
                
                elif order["status"] == "out for delivery":
                    st.info("Waiting for farmer to confirm delivery.")
                
                elif order["status"] == "delivered":
                    st.success("Order delivered!")
        else:
            st.info("No orders to display.")
            
                    # --- NEW: Multi‑Expert Q&A for Industry ---
        st.markdown("---")
        st.subheader("💬 Reply to Farmer Questions")
        consults = load_consultations()
        pending_qs = [q for q in consults["questions"] if st.session_state.username not in q["answers"]]
        if pending_qs:
            for q in pending_qs:
                st.markdown(f"**Farmer:** {q['farmer']}  \n**Question:** {q['question']}  \n_Asked {q['timestamp'][:19]}_")
                ans_text = st.text_area(f"Your Answer (QID {q['id']})", key=f"ans_{q['id']}")
                if st.button(f"Submit Answer to {q['farmer']}", key=f"submit_{q['id']}"):
                    if ans_text.strip():
                        add_consultation_answer(q["id"], st.session_state.username, ans_text.strip())
                        st.success("Answer submitted.")
                        st.rerun()
                st.markdown("---")
        else: st.info("No new questions awaiting your answer.")


else:
    st.info("Please login to use the app.")