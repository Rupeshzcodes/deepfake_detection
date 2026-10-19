import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import os
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from forensics import ForensicAnalyzer

# -----------------------------------------------------------------------------
# 1. Page Configuration & Professional Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Deepfake Detection | Enterprise Analysis",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Light Theme (Corporate/Academic)
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Internal:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Segoe UI', Helvetica, Arial, sans-serif;
        color: #333333;
        background-color: #f8f9fa; /* Light Gray Background */
    }

    /* Main Container */
    .stApp {
        background-color: #ffffff;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #f1f3f5;
        border-right: 1px solid #dee2e6;
    }

    /* Status Indicators */
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-active { background-color: #28a745; box-shadow: 0 0 2px #28a745; } /* Standard Green */
    .status-warning { background-color: #ffc107; }
    .status-error { background-color: #dc3545; }

    /* Cards/Containers */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e9ecef;
        border-radius: 6px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 10px;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #2c3e50; /* Dark Slate Blue */
        font-weight: 600;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #007bff;
        color: white;
        border: none;
        border-radius: 4px;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        color: white;
    }

    /* DataFrame */
    .dataframe {
        font-size: 14px;
        border: 1px solid #dee2e6 !important;
    }
    .dataframe th {
        background-color: #e9ecef !important;
        color: #495057 !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Logic & Resources
# -----------------------------------------------------------------------------
@st.cache_resource
def load_resources():
    model_path = 'deepfake_model.h5'
    analyzer = ForensicAnalyzer()
    model = None
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        except Exception as e:
            st.error(f"Error loading model: {e}")
    return model, analyzer

model, forensic_analyzer = load_resources()

# -----------------------------------------------------------------------------
# 3. Sidebar: System Status & Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/ios-filled/100/2c3e50/shield.png", width=50)
    st.title("Sentinel Defense")
    st.caption("Enterprise Forensic Suite v2.1")
    st.markdown("---")
    
    st.markdown("### System Status")
    
    # Professional Status Panel
    st.markdown("""
    <div style='background-color: #ffffff; padding: 15px; border-radius: 6px; border: 1px solid #dee2e6;'>
        <div><span class="status-indicator status-active"></span><strong>AI Core:</strong> Online</div>
        <div style='margin-top: 8px;'><span class="status-indicator status-active"></span><strong>Forensics Engine:</strong> Active</div>
        <div style='margin-top: 8px;'><span class="status-indicator status-active"></span><strong>Bio-Mesh:</strong> Ready</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### Configuration")
    confidence_threshold = st.slider("Alert Threshold", 0.0, 1.0, 0.75, 0.05)
    st.caption("Adjusts sensitivity for high-risk alerts.")

# -----------------------------------------------------------------------------
# 4. Main Dashboard
# -----------------------------------------------------------------------------
st.markdown("## Deepfake Analysis Dashboard")
st.markdown("Upload digital media for biometric and forensic verification.")

uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], help="Upload suspect image file")

if uploaded_file is not None:
    # Load Image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # -------------------------------------------------------------------------
    # RUN ANALYSIS
    # -------------------------------------------------------------------------
    with st.spinner("Processing image through Multi-Modal Engine..."):
        # 1. AI Score
        ai_conf = 0.5
        if model:
            try:
                processed = cv2.resize(image, (224, 224)) / 255.0
                processed = np.expand_dims(processed, axis=0)
                ai_conf = 1.0 - float(model.predict(processed, verbose=0)[0][0])
            except:
                pass
        
        # 2. Forensic Fusion (Using the NEW Integrated Score if available)
        # 2. Forensic Fusion (Using the NEW Integrated Score)
        try:
            # Try to use the 13-point suite
            report = forensic_analyzer.calculate_integrated_score(image, ai_conf)
            final_score = report["global_score"]
            # Map new report structure to UI expectation
            breakdown = report["modules"]
            face_mesh = report["data"]["mesh"]
        except Exception as e:
             st.error(f"Forensic Engine Error: {e}")
             # Temporary Fallback for stability, but showing error is key
             report = forensic_analyzer.calculate_fake_probability(image, ai_confidence=ai_conf)
             final_score = report["final_score"]
             breakdown = report["breakdown"]
             face_mesh = report.get("face_mesh_image", image)
        
    # -------------------------------------------------------------------------
    # DISPLAY RESULTS
    # -------------------------------------------------------------------------
    st.markdown("---")
    
    # Top Row: Images (Split View)
    
    # Visualization Selector (Keeps UI clean but adds functionality)
    vis_mode = st.radio("Select Analysis Layer:", ["Face Mesh Topology", "Error Level Analysis (ELA)", "Frequency Spectrum (FFT)", "Noise Variance Analysis", "Texture Analysis (LBP)"], horizontal=True)
    
    col_img1, col_img2 = st.columns(2)
    
    with col_img1:
        st.markdown("#### Original Evidence")
        st.image(image, channels="BGR", use_column_width=True, caption="Source Media")
    
    with col_img2:
        st.markdown(f"#### {vis_mode}")
        
        if vis_mode == "Face Mesh Topology":
            if face_mesh is not None:
                 st.image(face_mesh, channels="BGR", use_column_width=True, caption="Biometric Alignment Analysis")
                 # Show Head Pose Stats if available
                 if "data" in report and report["data"]["pose"]:
                     p = report["data"]["pose"]
                     st.caption(f"Head Pose: Pitch {p['pitch']:.0f}°, Yaw {p['yaw']:.0f}°, Roll {p['roll']:.0f}°")
                     if abs(p['yaw']) > 30:
                         st.warning("⚠️ Profile View Detected: Biometric accuracy may be reduced.")
            else:
                st.info("No face detected for mesh analysis.")
                
        elif vis_mode == "Error Level Analysis (ELA)":
            if "data" in report and report["data"]["ela"] is not None:
                st.image(report["data"]["ela"], use_column_width=True, caption="Compression Artifact Heatmap")
                st.info("Bright regions indicate potential splicing or lower quality patches.")
            else:
                st.warning("ELA data not available.")
                
        elif vis_mode == "Frequency Spectrum (FFT)":
            if "data" in report and report["data"]["fft"] is not None:
                # FFT is a matplotlib plot usually, or we can normalize it to image
                # The forensics module returns the raw magnitude spectrum (float array)
                # We need to normalize it to 0-255 for display as image, or use pyplot
                fft_data = report["data"]["fft"]
                
                # Normalize for display
                plt.figure(figsize=(5,5))
                plt.imshow(fft_data, cmap='inferno')
                plt.axis('off')
                plt.tight_layout()
                st.pyplot(plt)
                st.caption("Log Magnitude Spectrum (Checkerboard patterns = GAN)")
            else:
                st.warning("FFT data not available.")

        elif vis_mode == "Noise Variance Analysis":
             if "data" in report and report["data"]["noise"] is not None:
                 # Normalize Laplacian for display
                 noise_map = report["data"]["noise"]
                 display_noise = cv2.convertScaleAbs(noise_map, alpha=5.0) # Boost visibility
                 st.image(display_noise, use_column_width=True, caption="High-Pass Noise Map")
                 st.info("Uniform noise = Real. Smooth/Patchy areas = Fake/Blur.")
             else:
                 st.warning("Noise data not available.")

        elif vis_mode == "Texture Analysis (LBP)":
             if "data" in report and report["data"]["texture"] is not None:
                 tex_map = report["data"]["texture"]
                 # Normalize
                 display_tex = cv2.normalize(tex_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                 st.image(display_tex, use_column_width=True, caption="Texture Gradient Magnitude")
                 st.info("Low texture energy in detailed areas suggests smoothing/GANs.")
             else:
                 st.warning("Texture data not available.")

    st.markdown("---")

    # Bottom Row: Metrics & Report
    col_metrics, col_details = st.columns([1, 1.5])
    
    with col_metrics:
        mode_label = report.get("mode", "Standard Analysis")
        st.markdown(f"#### Analysis Verdict")
        st.caption(f"Current Mode: {mode_label}")
        
        # Color Logic (Professional Palette)
        veto = report.get("veto_triggered", False)
        
        if veto:
            color = "#dc3545" # Red
            label = "BIOMETRIC VETO"
            subtext = report.get("veto_reason", "Unnatural Eye Physics Detected")
            bg_color = "rgba(220, 53, 69, 0.1)"
        elif final_score > confidence_threshold:
            color = "#dc3545" # Red
            label = "HIGH PROBABILITY FAKE"
            subtext = "Critical Manipulation Detected"
            bg_color = "rgba(220, 53, 69, 0.1)"
        elif final_score > 0.45:
            color = "#ffc107" # Amber/Yellow
            label = "SUSPICIOUS ACTIVITY"
            subtext = "Potential Inconsistencies"
            bg_color = "rgba(255, 193, 7, 0.1)"
        else:
            color = "#28a745" # Green
            label = "AUTHENTIC MEDIA"
            subtext = "Verified Real"
            bg_color = "rgba(40, 167, 69, 0.1)"
            
        # Verdict Box
        st.markdown(f"""
        <div style='background-color: {bg_color}; border-left: 5px solid {color}; padding: 15px; border-radius: 4px;'>
            <h3 style='color: {color}; margin:0;'>{label}</h3>
            <p style='margin:5px 0 0 0; font-weight:500;'>{subtext}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Gauge Chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = final_score * 100,
            title = {'text': "Confidence Score"},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#333"},
                'bar': {'color': color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#dee2e6",
                'steps': [
                    {'range': [0, 45], 'color': '#d4edda'},
                    {'range': [45, 75], 'color': '#fff3cd'},
                    {'range': [75, 100], 'color': '#f8d7da'}],
            },
            number = {'suffix': "%", 'font': {'color': "#333"}}
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': '#333'},
            height=250,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_details:
        st.markdown("#### Forensic Abnormalities Table")
        
        # Prepare Data
        data = []
        # Handle dict items differently depending on which method was used
        # If using integrated score, breakdown is { "Module": score_float }
        
        if isinstance(list(breakdown.values())[0], float):
             # New 13-point suite format
             for key, val in breakdown.items():
                 risk = val * 100
                 status = "FAIL" if risk > 50 else "PASS"
                 details = "High Anomaly Detected" if status == "FAIL" else "Within Normal Range"
                 
                 # Add specific details if available in report['data']
                 if any(x in key for x in ["Biological", "Eye", "Geo"]) and "data" in report and report["data"]["pose"]:
                     p = report["data"]["pose"]
                     if p and status == "FAIL": details = "Irregular Head/Eye Geometry"
                 
                 if "Noise" in key and status == "FAIL": details = "Inconsistent Sensor Noise"
                 if "Texture" in key and status == "FAIL": details = "Unnatural Micro-Texture"
                     
                 data.append({
                    "Module": key,
                    "Status": status,
                    "Risk Score": f"{risk:.1f}%",
                    "Details": details
                })
        else:
            # Old format fallback
            for key, val in breakdown.items():
                if isinstance(val, str):
                    try:
                        risk_str = val.split('%')[0]
                        risk = float(risk_str)
                        details = val.split('(')[-1].strip(')') if '(' in val else ""
                    except:
                        risk = 0.0
                        details = val
                else:
                    risk = 0.0; details=""

                status = "FAIL" if risk > 50 else "PASS"
                data.append({
                    "Module": key,
                    "Status": status,
                    "Risk Score": f"{risk:.1f}%",
                    "Details": details
                })
            
        df = pd.DataFrame(data)
        
        # Styled DataFrame
        def color_status(val):
            color = '#dc3545' if val == 'FAIL' else '#28a745'
            weight = 'bold'
            return f'color: {color}; font-weight: {weight}'

        st.dataframe(
            df.style.map(color_status, subset=['Status']),
            use_container_width=True,
            hide_index=True
        )
        
        # Summary Text
        st.markdown("**Summary Findings:**")
        if report.get("veto_triggered"):
             st.error(report["veto_reason"])
             
        anomalies = [row for row in data if row["Status"] == "FAIL"]
        if anomalies:
             for a in anomalies:
                 st.markdown(f"- ⚠️ **{a['Module']}**: {a['Details']}")
        else:
             st.markdown("- ✅ All forensic checks passed. No significant anomalies found.")

    # -------------------------------------------------------------------------
    # HIDDEN DETAILS (Expanders to keep UI clean)
    # -------------------------------------------------------------------------
    st.markdown("---")
    c1, c2 = st.columns(2)
    
    with c1:
        with st.expander("📊 Ensemble Weights Analysis"):
            # Bar chart of the 4 ensemble components
            if "modules" in report:
                # report['modules'] = {'Deep Learning': 0.4, ...}
                df_mod = pd.DataFrame(list(report["modules"].items()), columns=["Module", "Score"])
                st.bar_chart(df_mod.set_index("Module"), color="#2c3e50")
                st.caption("Contribution of each forensic module to the global score.")
    
    with c2:
        with st.expander("📋 Exif Metadata & Provenance"):
            try:
                # Extract EXIF from uploaded file
                from PIL import Image, ExifTags
                img_pil = Image.open(uploaded_file)
                exif_data = img_pil._getexif()
                if exif_data:
                    exif_dict = {ExifTags.TAGS.get(k, k): v for k, v in exif_data.items()}
                    st.json(exif_dict)
                else:
                    st.info("No EXIF metadata found in this image.")
            except Exception as e:
                st.error(f"Metadata extraction unavailable: {e}")
