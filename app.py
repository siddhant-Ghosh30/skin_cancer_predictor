import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="Skin Cancer Classifier",
    page_icon="🧬",
    layout="wide"
)

# Load your trained model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('skin_cancer_final_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Class names with descriptions
class_info = {
    0: {
        "name": "Actinic Keratoses",
        "abbr": "akiec",
        "description": "Precancerous lesions caused by sun exposure",
        "risk": "HIGH",
        "recommendation": "Consult dermatologist - can evolve into squamous cell carcinoma"
    },
    1: {
        "name": "Basal Cell Carcinoma", 
        "abbr": "bcc",
        "description": "Most common skin cancer, slow-growing",
        "risk": "HIGH",
        "recommendation": "Urgent medical consultation required"
    },
    2: {
        "name": "Benign Keratosis",
        "abbr": "bkl", 
        "description": "Non-cancerous skin growths",
        "risk": "LOW",
        "recommendation": "Generally harmless, monitor for changes"
    },
    3: {
        "name": "Dermatofibroma",
        "abbr": "df",
        "description": "Benign fibrous nodule",
        "risk": "LOW", 
        "recommendation": "Harmless, no treatment needed usually"
    },
    4: {
        "name": "Melanoma",
        "abbr": "mel",
        "description": "Most dangerous skin cancer type",
        "risk": "CRITICAL",
        "recommendation": "EMERGENCY - Consult dermatologist immediately"
    },
    5: {
        "name": "Melanocytic Nevi",
        "abbr": "nv",
        "description": "Common moles - benign",
        "risk": "LOW",
        "recommendation": "Generally harmless, monitor ABCDE changes"
    },
    6: {
        "name": "Vascular Lesions",
        "abbr": "vasc",
        "description": "Blood vessel abnormalities",
        "risk": "LOW",
        "recommendation": "Usually benign, consult if concerned"
    }
}


def preprocess_image(image):
    """Convert image to RGB and resize to 224x224"""
    # Convert to RGB if image has 4 channels (RGBA)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    # Resize to match model input
    image_resized = image.resize((224, 224))
    
    # Convert to numpy array and normalize
    image_array = np.array(image_resized) / 255.0
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array, image_resized

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🧬 Skin Cancer Classification AI</h1>', unsafe_allow_html=True)
    
    # Model info sidebar
    with st.sidebar:
        st.header("Model Information")
        st.success("**Accuracy**: 74.69%")
        st.info("**Dataset**: HAM10000 (10,015 images)")
        st.warning("**Note**: AI assistant for educational purposes")
        
        st.header("Risk Levels")
        st.markdown('<div class="risk-critical">CRITICAL - Immediate medical attention</div>', unsafe_allow_html=True)
        st.markdown('<div class="risk-high">HIGH - Consult dermatologist</div>', unsafe_allow_html=True) 
        st.markdown('<div class="risk-low">LOW - Generally harmless</div>', unsafe_allow_html=True)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Skin Image")
        uploaded_file = st.file_uploader(
            "Choose a skin lesion image", 
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of the skin lesion"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Preprocessing with fix for RGBA images
            st.subheader("🛠️ Image Preprocessing")
            image_array, image_resized = preprocess_image(image)
            
            st.success(f"Image processed: {image_resized.size} pixels, {image.mode} mode")
            
            # Analyze button
            if st.button("🔍 Analyze with AI", type="primary", use_container_width=True):
                analyze_image(image_array, image_resized)
    
    with col2:
        if uploaded_file is None:
            st.header("ℹ️ How to Use")
            st.info("""
            1. **Upload** a clear image of skin lesion
            2. **Click Analyze** to get AI prediction
            3. **Review** results and recommendations
            4. **Consult** healthcare professional for diagnosis
            
            **Supported types**: JPG, JPEG, PNG
            **Best results**: Well-lit, focused images
            """)
            
            # Sample images or instructions
            st.header("📋 Skin Cancer Types")
            for class_id, info in class_info.items():
                risk_color = {
                    "CRITICAL": "🔴",
                    "HIGH": "🟠", 
                    "LOW": "🟢"
                }[info["risk"]]
                
                st.write(f"{risk_color} **{info['name']}** ({info['abbr']})")

def analyze_image(image_array, original_image):
    """Analyze the uploaded image and display results"""
    
    with st.spinner("🔄 AI model analyzing..."):
        # Get predictions
        predictions = model.predict(image_array, verbose=0)[0]
        predicted_class = np.argmax(predictions)
        confidence = predictions[predicted_class]
    
    # Display results in default Streamlit style
    st.header("🎯 Analysis Results")
    
    # Main prediction - simple text without colored cards
    info = class_info[predicted_class]
    
    # Use emojis for risk levels instead of colored backgrounds
    risk_emoji = {
        "CRITICAL": "🔴",
        "HIGH": "🟠", 
        "LOW": "🟢"
    }[info["risk"]]
    
    st.subheader(f"Primary Prediction: {info['name']} ({info['abbr']})")
    st.write(f"**Confidence:** {confidence:.2%}")
    st.write(f"**Risk Level:** {risk_emoji} {info['risk']}")
    st.write(f"**Description:** {info['description']}")
    st.write(f"**Recommendation:** {info['recommendation']}")
    
    # Confidence visualization
    st.subheader("📊 Prediction Confidence")
    
    # Create bar chart of all probabilities
    fig, ax = plt.subplots(figsize=(10, 6))
    classes = [class_info[i]["abbr"] for i in range(7)]
    colors = ['red' if class_info[i]["risk"] in ["HIGH", "CRITICAL"] else 'green' for i in range(7)]
    
    bars = ax.bar(classes, predictions, color=colors, alpha=0.7)
    ax.set_ylabel('Confidence')
    ax.set_xlabel('Skin Cancer Types')
    ax.set_title('AI Model Confidence for Each Class')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, predictions):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.2%}', ha='center', va='bottom', fontsize=9)
    
    st.pyplot(fig)
    
    # Detailed breakdown - simple list
    st.subheader("📋 Detailed Breakdown")
    
    for i, prob in enumerate(predictions):
        info = class_info[i]
        risk_icon = {
            "CRITICAL": "🔴",
            "HIGH": "🟠",
            "LOW": "🟢"
        }[info["risk"]]
        
        # Simple row without columns
        st.write(f"{risk_icon} **{info['name']} ({info['abbr']}):** {prob:.2%} - {info['description']}")
    
    # Medical disclaimer
    st.markdown("---")
    st.warning("""
    **⚠️ IMPORTANT MEDICAL DISCLAIMER**
    
    This AI tool is for educational and informational purposes only. It is NOT a substitute for professional medical advice, diagnosis, or treatment. 
    
    - Always seek the advice of qualified healthcare providers
    - Never disregard professional medical advice because of AI predictions
    - Skin cancer diagnosis requires professional medical examination
    - If you have concerns about skin lesions, consult a dermatologist immediately
    """)

if __name__ == "__main__":
    main()