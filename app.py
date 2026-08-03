from flask import Flask, render_template, request, send_file , redirect, url_for
import numpy as np
import pandas as pd
import joblib
import random
import os
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model
from PIL import Image
import tempfile
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors

# ==============================
# Flask App
# ==============================
app = Flask(__name__)

# ==============================
# Load Crop Recommendation Model
# ==============================
crop_model_path = "crop_recommendation_model.pkl"
crop_model = joblib.load(crop_model_path)

# ==============================
# Load Yield Prediction Model
# ==============================
yield_model_path = "yield_prediction_model.pkl"
yield_model = joblib.load(yield_model_path)

model_columns_path = "yield_columns.pkl"
model_columns = joblib.load(model_columns_path)



# ==============================
# Recreate EfficientNetB3 Architecture and Load Weights
# ==============================
disease_model_weights = "plant_disease_model_b3_final1.keras"

base_model = EfficientNetB3(weights=None, include_top=False, input_shape=(300, 300, 3))
base_model.trainable = False

inputs = layers.Input(shape=(300, 300, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(29, activation='softmax')(x)
disease_model = Model(inputs, outputs)
disease_model.load_weights(disease_model_weights)

# ==============================
# Classes
# ==============================
class_names = [
    'AppleAppleScab', 'AppleBlackRot', 'AppleCedarAppleRust', 'AppleHealthy',
    'BellPepperBacterialSpot', 'BellPepperHealthy',
    'CherryHealthy', 'CherryPowderyMildew',
    'CornMaizeCercosporaLeafSpot', 'CornMaizeCommonRust', 'CornMaizeHealthy', 'CornMaizeNorthernLeafBlight',
    'GrapeBlackRot', 'GrapeEscaBlackMeasles', 'GrapeHealthy', 'GrapeLeafBlight',
    'PeachBacterialSpot', 'PeachHealthy',
    'PotatoEarlyBlight', 'PotatoHealthy', 'PotatoLateBlight',
    'StrawberryHealthy', 'StrawberryLeafScorch',
    'TomatoBacterialSpot', 'TomatoEarlyBlight', 'TomatoHealthy', 'TomatoLateBlight',
    'TomatoSeptoriaLeafSpot', 'TomatoYellowLeafCurlVirus'
]

# ==============================
# Disease Remedies
# ==============================
disease_advice = {
    'AppleAppleScab': {
        'Low': ['Remove minor infected leaves.', 'Maintain good air circulation.', 'Avoid overhead watering.', 'Keep orchard floor clean.'],
        'Medium': ['Prune infected branches.', 'Apply sulfur-based fungicide.', 'Remove fallen leaves.', 'Monitor disease spread weekly.'],
        'High': ['Apply systemic fungicide immediately.', 'Remove severely infected trees.', 'Quarantine affected areas.', 'Implement integrated pest management.']
    },
    'AppleBlackRot': {
        'Low': ['Pick off small rotted fruits.', 'Clean fallen leaves.', 'Improve tree ventilation.', 'Monitor for further signs.'],
        'Medium': ['Prune infected branches.', 'Spray copper-based fungicide.', 'Dispose of infected fruits safely.', 'Maintain tree health.'],
        'High': ['Remove heavily infected trees.', 'Use systemic fungicides.', 'Restrict movement of infected material.', 'Intensive monitoring of orchard.']
    },
    'AppleCedarAppleRust': {
        'Low': ['Remove minor galls.', 'Improve airflow around trees.', 'Monitor disease progression.', 'Keep grass short around trees.'],
        'Medium': ['Spray fungicide on apple trees.', 'Remove nearby cedar trees if possible.', 'Regularly check for infections.', 'Prune infected branches.'],
        'High': ['Apply systemic fungicide immediately.', 'Remove heavily infected trees or branches.', 'Implement quarantine for infected areas.', 'Frequent monitoring and preventive sprays.']
    },
    'AppleHealthy': {
        'Low': ['No action needed.', 'Keep routine care.', 'Regularly check for pests.', 'Maintain proper irrigation.'],
        'Medium': ['No action needed.', 'Ensure proper fertilization.', 'Monitor for early signs of disease.', 'Maintain plant hygiene.'],
        'High': ['No action needed.', 'Continue regular monitoring.', 'Maintain orchard hygiene.', 'Ensure soil health.']
    },
    'BellPepperBacterialSpot': {
        'Low': ['Remove small infected spots.', 'Avoid overhead watering.', 'Maintain good spacing.', 'Monitor regularly.'],
        'Medium': ['Apply copper-based sprays.', 'Remove affected leaves.', 'Ensure proper nutrition.', 'Rotate crops.'],
        'High': ['Remove infected plants.', 'Use systemic bactericide.', 'Quarantine affected area.', 'Intensive monitoring.']
    },
    'BellPepperHealthy': {
        'Low': ['No action needed.', 'Maintain proper watering.', 'Regular inspection.', 'Good soil management.'],
        'Medium': ['No action needed.', 'Maintain plant hygiene.', 'Monitor for pests.', 'Proper fertilization.'],
        'High': ['No action needed.', 'Continue monitoring.', 'Ensure spacing and ventilation.', 'Maintain soil health.']
    },
    'CherryHealthy': {
        'Low': ['No action needed.', 'Routine care.', 'Monitor for pests.', 'Maintain irrigation.'],
        'Medium': ['No action needed.', 'Ensure proper nutrition.', 'Check for early infection signs.', 'Keep orchard floor clean.'],
        'High': ['No action needed.', 'Continue monitoring.', 'Maintain proper pruning.', 'Protect from extreme weather.']
    },
    'CherryPowderyMildew': {
        'Low': ['Remove minor mildew.', 'Prune lightly.', 'Improve airflow.', 'Avoid wetting leaves.'],
        'Medium': ['Apply sulfur-based fungicides.', 'Remove affected branches.', 'Regularly monitor spread.', 'Maintain proper spacing.'],
        'High': ['Apply systemic fungicides.', 'Remove severely infected plants.', 'Quarantine affected area.', 'Frequent inspection and treatment.']
    },
    'CornMaizeCercosporaLeafSpot': {
        'Low': ['Remove small infected leaves.', 'Monitor regularly.', 'Avoid overhead irrigation.', 'Ensure airflow.'],
        'Medium': ['Apply fungicides.', 'Remove affected leaves.', 'Rotate crops.', 'Maintain soil health.'],
        'High': ['Use systemic fungicides.', 'Remove heavily infected plants.', 'Monitor surrounding fields.', 'Implement crop rotation.']
    },
    'CornMaizeCommonRust': {
        'Low': ['Monitor minor rust spots.', 'Maintain airflow.', 'Avoid wetting leaves.', 'Ensure proper fertilization.'],
        'Medium': ['Apply fungicides.', 'Remove infected leaves.', 'Check crop regularly.', 'Maintain proper spacing.'],
        'High': ['Use systemic fungicides.', 'Remove heavily infected plants.', 'Monitor surrounding fields.', 'Intensive crop protection.']
    },
    'CornMaizeHealthy': {
        'Low': ['No action needed.', 'Maintain regular watering.', 'Routine monitoring.', 'Keep soil healthy.'],
        'Medium': ['No action needed.', 'Proper fertilization.', 'Monitor for pests.', 'Maintain crop hygiene.'],
        'High': ['No action needed.', 'Continue monitoring.', 'Preventive measures for nearby fields.', 'Maintain airflow and spacing.']
    },
    'CornMaizeNorthernLeafBlight': {
        'Low': ['Remove minor lesions.', 'Monitor regularly.', 'Maintain airflow.', 'Avoid overhead irrigation.'],
        'Medium': ['Apply fungicide.', 'Remove infected leaves.', 'Rotate crops.', 'Monitor disease spread.'],
        'High': ['Use systemic fungicides.', 'Remove severely infected plants.', 'Quarantine infected area.', 'Intensive monitoring.']
    },
    'GrapeBlackRot': {
        'Low': ['Remove minor infected fruits.', 'Prune lightly.', 'Maintain airflow.', 'Monitor regularly.'],
        'Medium': ['Apply fungicides.', 'Remove affected leaves and fruits.', 'Maintain proper nutrition.', 'Regular monitoring.'],
        'High': ['Remove infected plants.', 'Apply systemic fungicides.', 'Quarantine infected area.', 'Intensive monitoring.']
    },
    'GrapeEscaBlackMeasles': {
        'Low': ['Prune minor affected wood.', 'Monitor regularly.', 'Maintain vineyard hygiene.', 'Ensure airflow.'],
        'Medium': ['Remove infected branches.', 'Apply fungicides if needed.', 'Monitor disease progression.', 'Ensure plant nutrition.'],
        'High': ['Remove heavily infected wood.', 'Apply systemic fungicides.', 'Quarantine infected area.', 'Frequent monitoring.']
    },
    'GrapeHealthy': {
        'Low': ['No action needed.', 'Regular monitoring.', 'Maintain good soil and water.', 'Routine care.'],
        'Medium': ['No action needed.', 'Monitor for pests.', 'Proper fertilization.', 'Maintain plant hygiene.'],
        'High': ['No action needed.', 'Continue monitoring.', 'Ensure vineyard hygiene.', 'Proper spacing and ventilation.']
    },
    'GrapeLeafBlight': {
        'Low': ['Remove minor lesions.', 'Monitor regularly.', 'Maintain airflow.', 'Proper watering.'],
        'Medium': ['Apply fungicides.', 'Remove affected leaves.', 'Monitor weekly.', 'Maintain plant nutrition.'],
        'High': ['Use systemic fungicides.', 'Remove heavily infected leaves.', 'Quarantine affected area.', 'Intensive monitoring.']
    },
    'PeachBacterialSpot': {
        'Low': ['Remove minor infected leaves.', 'Monitor trees.', 'Maintain airflow.', 'Avoid wetting foliage.'],
        'Medium': ['Apply copper sprays.', 'Remove affected branches.', 'Monitor disease progression.', 'Maintain proper nutrition.'],
        'High': ['Remove infected plants.', 'Use systemic bactericides.', 'Quarantine affected area.', 'Frequent monitoring.']
    },
    'PeachHealthy': {
        'Low': ['No action needed.', 'Routine care.', 'Regular inspection.', 'Maintain soil and water.'],
        'Medium': ['No action needed.', 'Monitor for pests.', 'Proper pruning.', 'Ensure plant nutrition.'],
        'High': ['No action needed.', 'Continue monitoring.', 'Maintain orchard hygiene.', 'Proper irrigation.']
    },
    'PotatoEarlyBlight': {
        'Low': ['Remove minor affected leaves.', 'Monitor weekly.', 'Ensure airflow.', 'Avoid wetting leaves.'],
        'Medium': ['Apply fungicides.', 'Remove infected leaves.', 'Rotate crops.', 'Maintain soil health.'],
        'High': ['Use systemic fungicides.', 'Remove heavily infected plants.', 'Monitor surrounding crops.', 'Intensive care.']
    },
    'PotatoHealthy': {
        'Low': ['No action needed.', 'Regular watering.', 'Monitor plants.', 'Maintain soil fertility.'],
        'Medium': ['No action needed.', 'Monitor for early infection.', 'Proper fertilization.', 'Routine care.'],
        'High': ['No action needed.', 'Continue monitoring.', 'Maintain airflow and spacing.', 'Ensure healthy soil.']
    },
    'PotatoLateBlight': {
        'Low': ['Remove minor infected leaves.', 'Monitor closely.', 'Avoid wetting foliage.', 'Maintain proper spacing.'],
        'Medium': ['Apply fungicides.', 'Remove infected plants.', 'Crop rotation.', 'Maintain soil health.'],
        'High': ['Use systemic fungicides immediately.', 'Destroy heavily infected plants.', 'Quarantine affected area.', 'Frequent monitoring.']
    },
    'StrawberryHealthy': {
        'Low': ['No action needed.', 'Routine watering.', 'Monitor plants.', 'Maintain soil health.'],
        'Medium': ['No action needed.', 'Proper fertilization.', 'Monitor for pests.', 'Keep plant area clean.'],
        'High': ['No action needed.', 'Continue regular monitoring.', 'Maintain spacing and ventilation.', 'Preventive care.']
    },
    'StrawberryLeafScorch': {
        'Low': ['Remove minor affected leaves.', 'Monitor weekly.', 'Maintain airflow.', 'Avoid overhead watering.'],
        'Medium': ['Apply bactericides.', 'Remove infected leaves.', 'Ensure proper nutrition.', 'Monitor spread.'],
        'High': ['Remove heavily infected plants.', 'Use systemic bactericides.', 'Quarantine area.', 'Intensive monitoring.']
    },
    'TomatoBacterialSpot': {
        'Low': ['Remove minor spots.', 'Maintain good spacing.', 'Monitor regularly.', 'Avoid wetting foliage.'],
        'Medium': ['Apply copper-based sprays.', 'Remove affected leaves.', 'Crop rotation.', 'Maintain plant nutrition.'],
        'High': ['Remove infected plants.', 'Use systemic bactericides.', 'Quarantine affected area.', 'Intensive monitoring.']
    },
    'TomatoEarlyBlight': {
        'Low': ['Remove minor lesions.', 'Monitor regularly.', 'Maintain airflow.', 'Avoid wetting leaves.'],
        'Medium': ['Apply fungicides.', 'Remove infected leaves.', 'Rotate crops.', 'Maintain soil health.'],
        'High': ['Use systemic fungicides.', 'Remove heavily infected plants.', 'Monitor nearby plants.', 'Intensive care.']
    },
    'TomatoHealthy': {
        'Low': ['No action needed.', 'Routine care.', 'Monitor for pests.', 'Maintain irrigation and soil.'],
        'Medium': ['No action needed.', 'Proper fertilization.', 'Monitor plants.', 'Maintain hygiene.'],
        'High': ['No action needed.', 'Continue monitoring.', 'Ensure airflow and spacing.', 'Preventive measures.']
    },
    'TomatoLateBlight': {
        'Low': ['Remove minor infected leaves.', 'Monitor closely.', 'Maintain spacing.', 'Avoid wetting foliage.'],
        'Medium': ['Apply fungicides.', 'Remove affected plants.', 'Crop rotation.', 'Maintain soil health.'],
        'High': ['Use systemic fungicides immediately.', 'Destroy heavily infected plants.', 'Quarantine area.', 'Frequent monitoring.']
    },
    'TomatoSeptoriaLeafSpot': {
        'Low': ['Remove minor spots.', 'Monitor weekly.', 'Maintain airflow.', 'Avoid overhead watering.'],
        'Medium': ['Apply fungicides.', 'Remove infected leaves.', 'Rotate crops.', 'Maintain soil health.'],
        'High': ['Use systemic fungicides.', 'Remove heavily infected plants.', 'Quarantine area.', 'Intensive monitoring.']
    },
    'TomatoYellowLeafCurlVirus': {
        'Low': ['Remove mildly affected leaves.', 'Monitor plants regularly.', 'Maintain spacing.', 'Control whiteflies manually.'],
        'Medium': ['Remove affected plants.', 'Use insecticides for whitefly control.', 'Monitor disease spread.', 'Maintain plant nutrition.'],
        'High': ['Destroy infected plants.', 'Apply systemic insecticides for whiteflies.', 'Quarantine infected area.', 'Frequent monitoring.']
    }
}
# ==============================
# Crop Benefits (Updated)
# ==============================
crop_benefits = {
    "rice": [
        "Stable market demand with government support.",
        "Grows well in water-rich fields.",
        "Straw and husk provide extra income."
    ],
    "maize": [
        "Short-duration, high-yield crop.",
        "Multiple uses (food, fodder, industry).",
        "Suitable for both irrigated and rainfed areas."
    ],
    "chickpea": [
        "Needs less water than cereals.",
        "Improves soil fertility for next crops.",
        "Good local and export market price."
    ],
    "kidney beans": [
        "High demand in markets.",
        "Performs well in moderate rainfall areas.",
        "Can be intercropped for added income."
    ],
    "pigeon peas": [
        "Thrives in dryland farming.",
        "Improves soil nitrogen naturally.",
        "Steady dal (pulse) market demand."
    ],
    "moth beans": [
        "Ideal for drought-prone regions.",
        "Low input cost with decent returns.",
        "Protein-rich crop with easy marketability."
    ],
    "mung beans": [
        "Very short crop cycle (60–70 days).",
        "Improves soil fertility.",
        "High demand for sprouts and dal."
    ],
    "black gram": [
        "Profitable even in poor soils.",
        "Adds nitrogen to the soil.",
        "Strong domestic consumption."
    ],
    "lentils": [
        "Grows with residual soil moisture.",
        "Improves soil health.",
        "High export potential with stable prices."
    ],
    "pomegranate": [
        "High-value export fruit.",
        "Drought-tolerant once established.",
        "Long harvesting season gives steady income."
    ],
    "banana": [
        "Yields in 10–12 months.",
        "Year-round market demand.",
        "By-products (leaves, stem) add extra value."
    ],
    "mango": [
        "Long-term orchard with yearly returns.",
        "Premium summer fruit with strong demand.",
        "Processing industries add extra income."
    ],
    "grapes": [
        "Quick returns in 1–2 years.",
        "Used fresh, for raisins, or wine.",
        "High export profit margins."
    ],
    "watermelon": [
        "Harvest in 70–90 days.",
        "Strong summer demand.",
        "Low input, quick cash crop."
    ],
    "muskmelon": [
        "Ready in 2–3 months.",
        "Thrives in sandy soils.",
        "Premium fruit in hot season."
    ],
    "apple": [
        "High-value fruit for hilly areas.",
        "Storage allows delayed selling.",
        "Export-friendly crop."
    ],
    "orange": [
        "High juice and fresh fruit demand.",
        "Multiple harvests possible in some varieties.",
        "Stable pricing and export scope."
    ],
    "papaya": [
        "Starts yielding in 8–10 months.",
        "Produces fruit for 2–3 years.",
        "Demand in food and medicine industries."
    ],
    "coconut": [
        "Lifespan of 60+ years with steady income.",
        "Multiple uses (oil, coir, water).",
        "Consistent industry demand."
    ],
    "cotton": [
        "Always in demand for textiles.",
        "By-products (oil, cake) add profit.",
        "MSP support ensures price stability."
    ],
    "jute": [
        "Eco-friendly fiber with rising demand.",
        "Suited for flood-prone/loamy soils.",
        "Assured buyers and steady market."
    ],
    "coffee": [
        "High export earning crop.",
        "Intercropping with spices gives extra income.",
        "Specialty coffee fetches premium prices."
    ]
}

# Seasonal crop tips based on classes
crop_seasonal_tips = {
    "Apple": {
        "season": "Autumn",
        "tips": [
            "Harvest apples in late autumn for best taste.",
            "Prune branches after harvest to prepare for next season.",
            "Apply nitrogen-rich fertilizers post-harvest."
        ]
    },
    "BellPepper": {
        "season": "Summer",
        "tips": [
            "Plant bell peppers in warm summer soil.",
            "Water in early morning to avoid leaf diseases.",
            "Harvest fruits when fully colored and firm."
        ]
    },
    "Cherry": {
        "season": "Spring",
        "tips": [
            "Cherries ripen in late spring.",
            "Use fungicides to prevent powdery mildew.",
            "Monitor for pests during flowering."
        ]
    },
    "CornMaize": {
        "season": "Monsoon",
        "tips": [
            "Sow maize at the start of monsoon for best yield.",
            "Ensure proper spacing to prevent fungal diseases.",
            "Apply fertilizers during early growth stages."
        ]
    },
    "Grape": {
        "season": "Summer",
        "tips": [
            "Thin grape clusters to improve fruit quality.",
            "Monitor for fungal infections in humid weather.",
            "Harvest in mid-summer when sugar content is high."
        ]
    },
    "Peach": {
        "season": "Summer",
        "tips": [
            "Peaches are harvested in early to mid-summer.",
            "Prune dead branches before flowering.",
            "Apply potassium-rich fertilizers to improve fruit quality."
        ]
    },
    "Potato": {
        "season": "Winter",
        "tips": [
            "Plant potatoes in winter for cool-season crops.",
            "Avoid waterlogging to prevent blight.",
            "Apply phosphorus-rich fertilizers during early growth."
        ]
    },
    "Strawberry": {
        "season": "Spring",
        "tips": [
            "Plant strawberries in early spring.",
            "Protect plants from frost with mulch.",
            "Harvest berries when fully red and firm."
        ]
    },
    "Tomato": {
        "season": "Summer",
        "tips": [
            "Tomatoes grow best in warm summer months.",
            "Prune and stake plants to reduce disease risk.",
            "Harvest fruits when fully ripe and red."
        ]
    }
}

def get_remedy(disease_class, severity_level):
    disease_data = disease_advice.get(disease_class)
    if not disease_data:
        return ["No data available for this disease."]
    remedies = disease_data.get(severity_level)
    if not remedies:
        return ["No remedies available for this severity level."]
    return remedies

# ==============================
# Image Preprocessing
# ==============================
def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((300, 300))
    x = np.array(img, dtype=np.float32)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

import cv2
import numpy as np

def calculate_patch_severity(image_path, disease_class):

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or invalid path.")

    image = cv2.resize(image, (512, 512))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # ---------------------------------
    # 1️⃣ Leaf Mask (Remove Background)
    # ---------------------------------
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    lower_yellow_leaf = np.array([20, 40, 40])
    upper_yellow_leaf = np.array([35, 255, 255])
    yellow_leaf_mask = cv2.inRange(hsv, lower_yellow_leaf, upper_yellow_leaf)

    leaf_mask = cv2.bitwise_or(green_mask, yellow_leaf_mask)

    kernel = np.ones((5, 5), np.uint8)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)

    total_leaf_pixels = np.sum(leaf_mask > 0)

    if total_leaf_pixels == 0:
        return 0.0, None

    # ---------------------------------
    # 2️⃣ Healthy Check (ALL 29 SAFE)
    # ---------------------------------
    if "healthy" in disease_class.lower():
        return 0.0, leaf_mask

    disease_mask = np.zeros_like(leaf_mask)

    # ---------------------------------
    # 3️⃣ GROUP DEFINITIONS
    # ---------------------------------

    brown_spot_group = [
        "AppleAppleScab", "AppleBlackRot",
        "CornMaizeCercosporaLeafSpot",
        "TomatoSeptoriaLeafSpot",
        "PotatoEarlyBlight",
        "StrawberryLeafScorch",
        "GrapeBlackRot"
    ]

    blight_group = [
        "TomatoEarlyBlight", "TomatoLateBlight",
        "PotatoLateBlight",
        "CornMaizeNorthernLeafBlight",
        "GrapeLeafBlight"
    ]

    powdery_group = [
        "CherryPowderyMildew"
    ]

    rust_group = [
        "CornMaizeCommonRust"
    ]

    viral_group = [
        "TomatoYellowLeafCurlVirus"
    ]

    # ---------------------------------
    # 4️⃣ Disease-Specific Detection
    # ---------------------------------

    # 🔴 Brown / Black Spot Diseases
    if disease_class in brown_spot_group:

        brown_mask = cv2.inRange(hsv,
                                 np.array([10, 50, 20]),
                                 np.array([35, 255, 200]))

        black_mask = cv2.inRange(hsv,
                                 np.array([0, 0, 0]),
                                 np.array([180, 255, 60]))

        disease_mask = cv2.bitwise_or(brown_mask, black_mask)

    # 🟡 Blight Diseases (Large Yellow/Brown)
    elif disease_class in blight_group:

        yellow_mask = cv2.inRange(hsv,
                                  np.array([20, 80, 80]),
                                  np.array([35, 255, 255]))

        brown_mask = cv2.inRange(hsv,
                                 np.array([10, 50, 20]),
                                 np.array([25, 255, 200]))

        disease_mask = cv2.bitwise_or(yellow_mask, brown_mask)

    # ⚪ Powdery Mildew (White fungus)
    elif disease_class in powdery_group:

        disease_mask = cv2.inRange(hsv,
                                   np.array([0, 0, 180]),
                                   np.array([180, 70, 255]))

    # 🟠 Rust (Orange spots)
    elif disease_class in rust_group:

        disease_mask = cv2.inRange(hsv,
                                   np.array([5, 100, 100]),
                                   np.array([20, 255, 255]))

    # 🟡 Viral Yellow Leaf Curl
    elif disease_class in viral_group:

        disease_mask = cv2.inRange(hsv,
                                   np.array([20, 80, 80]),
                                   np.array([35, 255, 255]))

    else:
        # Fallback: detect non-green areas inside leaf
        non_green = cv2.bitwise_not(leaf_mask)
        disease_mask = cv2.bitwise_and(non_green, leaf_mask)

    # ---------------------------------
    # 5️⃣ Restrict Detection to Leaf Only
    # ---------------------------------
    disease_mask = cv2.bitwise_and(disease_mask, leaf_mask)

    disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_OPEN, kernel)
    disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_CLOSE, kernel)

    infected_pixels = np.sum(disease_mask > 0)

    severity = (infected_pixels / total_leaf_pixels) * 100

    return round(severity, 2), disease_mask
# ==============================
# Routes
# ==============================

# ---------- Front Page ----------
@app.route("/")
def index():
    return render_template("index.html")  # logo, tagline, about us, next button
# ---------- Seasonal Tips ----------
@app.route("/tips")
def tips():
    seasonal_tips = [
        "Spring: Prepare soil, plant early vegetables, prune trees.",
        "Summer: Ensure adequate irrigation, control pests, harvest early crops.",
        "Monsoon: Protect from waterlogging, apply fungicides, support climbing plants.",
        "Autumn: Harvest fruits, sow winter crops, enrich soil with compost.",
        "Winter: Mulch to protect plants, prune fruit trees, monitor for frost damage."
    ]
    return render_template("tips.html", tips=seasonal_tips)
    
# ---------- Disease Detection ----------
@app.route("/disease", methods=["GET", "POST"])
def disease():
    if request.method == "POST":
        file = request.files["disease_image"]

        # Save uploaded file
        file_path = os.path.join("static", file.filename)
        file.save(file_path)

        # Preprocess & predict
        img_array = preprocess_image(file_path)
        preds = disease_model.predict(img_array)
        class_idx = np.argmax(preds, axis=1)[0]
        disease_class = class_names[class_idx]

        # Get probability (confidence)
        confidence = np.max(preds)

        # Assign severity based on confidence
        # -----------------------------
        # Calculate severity using OpenCV patches
        # -----------------------------
        severity_percent, disease_mask = calculate_patch_severity(file_path, disease_class)

        # Convert percentage to Low/Medium/High
        if severity_percent < 10:
                 severity_level = "Low"
        elif severity_percent < 30:
                 severity_level = "Medium"
        else:
                severity_level = "High"

        severity = severity_level  # pass to template
        # Get remedies (if available)
        remedies = get_remedy(disease_class, severity) if severity in ["Low", "Medium","High"] else []

        # Trigger popup only if not healthy
        show_alert = not disease_class.lower().endswith("healthy")

        # Only overlay if disease exists
        if disease_mask is not None:
        # Read original image
              original = cv2.imread(file_path)
              original = cv2.resize(original, (512, 512))

        # Create colored mask (red)
              red_mask = np.zeros_like(original)
              red_mask[:, :, 2] = disease_mask  # Set red channel

        # Overlay with some transparency
              overlayed = cv2.addWeighted(original, 0.7, red_mask, 0.3, 0)

        # Save overlayed image to static folder to display
              patched_image_filename = f"patched_{file.filename}" 
        # just filename
              cv2.imwrite(os.path.join("static", patched_image_filename), overlayed)

         # Pass this to template
              patched_image_path = patched_image_filename

        else:
              patched_image_path = file.filename  # just show original if healthy
   
        return render_template(
            "disease_result.html",
            disease=disease_class,
            severity=severity,
            confidence=round(float(confidence), 2),
            remedies=remedies,
            image_path=file.filename,
            patched_image_path=patched_image_path,  # 👈 NEW
            show_alert=show_alert
        )

    return render_template("disease.html")


# ---------- Crop Recommendation ----------
@app.route("/crop", methods=["GET", "POST"])
def crop():
    if request.method == "POST":
        features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        try:
            # Convert to floats safely, validate keys exist
            X = np.array([[float(request.form.get(f, 0)) for f in features]])
        except ValueError:
            # Value conversion error fallback
            return render_template("crop.html", error="Invalid input: Please enter valid numeric values.")

        predicted_crop = crop_model.predict(X)[0]

        # Defensive: Use lowercase keys for dictionary lookup
        key = predicted_crop.lower()
        benefits = crop_benefits.get(key, "No benefits info available.")

        return render_template("crop_result.html", crop=predicted_crop, benefits=benefits)

    return render_template("crop.html")


# ---------- Yield Prediction ----------
# ---------- Yield Prediction ----------
@app.route("/yield", methods=["GET", "POST"])
def yield_prediction():
    if request.method == "POST":

        crop = request.form["crop"]
        year = int(request.form["year"])
        season = request.form["season"]
        state = request.form["state"]
        area = float(request.form["area"])
        production = float(request.form["production"])
        rainfall = float(request.form["rainfall"])
        fertilizer = float(request.form["fertilizer"])
        pesticide = float(request.form["pesticide"])

        # Create dataframe with SAME structure as training
        input_df = pd.DataFrame([{
            "Crop": crop,
            "Crop_Year": year,
            "Season": season,
            "State": state,
            "Area": area,
            "Production": production,
            "Annual_Rainfall": rainfall,
            "Fertilizer": fertilizer,
            "Pesticide": pesticide
        }])

        # Apply SAME encoding used during training
        input_encoded = pd.get_dummies(input_df, drop_first=True)

        # Match training columns
        input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

        # Predict
        prediction = yield_model.predict(input_encoded)[0]

        return render_template(
            "yield_result.html",
            crop=crop,
            year=year,
            season=season,
            state=state,
            prediction=round(prediction, 2)
        )

    return render_template("yield.html")



# ---------- Disease PDF ----------
@app.route("/download_disease_pdf", methods=["POST"])
def download_disease_pdf():
    disease = request.form.get("disease")
    severity = request.form.get("severity")
    remedies = request.form.getlist("remedies")
    
    pdf_path = tempfile.mktemp(".pdf")
    c = canvas.Canvas(pdf_path, pagesize=letter)
    
    # ---------- Background color ----------
    c.setFillColor(colors.HexColor("#b2f2bb"))  # mint green background
    c.rect(0, 0, 612, 792, fill=1)  # fill entire page

    # ---------- Logo ----------
    logo_path = os.path.join("static", "logo.jpeg")
 # your logo path
    if os.path.exists(logo_path):
        c.drawImage(logo_path, 50, 720, width=100, height=100, preserveAspectRatio=True)

    # ---------- Website Name ----------
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(170, 770, "FarmSavior")  # change to your website name

    # ---------- Disease Info ----------
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 650, f"Disease Report: {disease}")
    c.setFont("Helvetica", 12)
    c.drawString(50, 620, f"Severity: {severity}")
    c.drawString(50, 600, "Remedies:")
    
    y = 580
    for r in remedies:
        c.drawString(60, y, f"- {r}")
        y -= 20

    c.save()
    return send_file(pdf_path, as_attachment=True, download_name=f"{disease}_report.pdf")

 # ---------- Crop PDF ----------
@app.route("/download_crop_pdf", methods=["POST"])
def download_crop_pdf():
    crop = request.form.get("crop")
    benefits = request.form.getlist("benefits")

    pdf_path = tempfile.mktemp(".pdf")
    c = canvas.Canvas(pdf_path, pagesize=letter)

    # Background
    c.setFillColor(colors.HexColor("#b2f2bb"))
    c.rect(0, 0, 612, 792, fill=1)

    # Logo
    logo_path = os.path.join("static", "logo.jpeg")
    if os.path.exists(logo_path):
        c.drawImage(logo_path, 50, 720, width=100, height=100)

    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(170, 770, "FarmSavior")

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 650, f"Crop Recommendation: {crop}")

    c.setFont("Helvetica", 12)
    y = 620
    for b in benefits:
        c.drawString(60, y, f"- {b}")
        y -= 20

    c.save()
    return send_file(pdf_path, as_attachment=True,download_name=f"{crop}_recommendation.pdf")



# ---------- Yield PDF (SEPARATE ROUTE) ----------
@app.route("/download_yield_pdf", methods=["POST"])
def download_yield_pdf():
    crop = request.form.get("crop")
    state = request.form.get("state")
    season = request.form.get("season")
    year = request.form.get("year")
    prediction = request.form.get("prediction")

    pdf_path = tempfile.mktemp(".pdf")
    c = canvas.Canvas(pdf_path, pagesize=letter)

    c.setFillColor(colors.HexColor("#b2f2bb"))
    c.rect(0, 0, 612, 792, fill=1)

    logo_path = os.path.join("static", "logo.jpeg")
    if os.path.exists(logo_path):
        c.drawImage(logo_path, 50, 720, width=100, height=100)

    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(170, 770, "FarmSavior")

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 650, "Crop Yield Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, 620, f"Crop: {crop}")
    c.drawString(50, 600, f"State: {state}")
    c.drawString(50, 580, f"Season: {season}")
    c.drawString(50, 560, f"Year: {year}")
    c.drawString(50, 530, f"Predicted Yield: {prediction} tons/hectare")

    c.save()
    return send_file(pdf_path, as_attachment=True,
                     download_name=f"{crop}_yield_report.pdf")



# ==============================
# Run App
# ==============================
if __name__ == "__main__":
    app.run(debug=True)
