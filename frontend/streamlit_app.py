import streamlit as st
import requests
from PIL import Image
import io

st.title("Image Classification Modal Demo")

# Add endpoint configuration at the top
MODAL_ENDPOINT = "https://shukraditya-bose2022--keras-image-endpoint-predict.modal.run"
HEALTH_ENDPOINT = "https://shukraditya-bose2022--keras-image-endpoint-health.modal.run"

# Add a health check section
st.sidebar.header("Backend Status")
if st.sidebar.button("Check Backend Health"):
    try:
        health_response = requests.get(HEALTH_ENDPOINT)
        if health_response.ok:
            health_data = health_response.json()
            if health_data["status"] == "healthy":
                st.sidebar.success("✅ Backend is healthy")
                st.sidebar.json(health_data)
            else:
                st.sidebar.error("❌ Backend is unhealthy")
                st.sidebar.json(health_data)
        else:
            st.sidebar.error(f"Health check failed: {health_response.text}")
    except Exception as e:
        st.sidebar.error(f"Cannot reach backend: {str(e)}")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Show image details
    st.write(f"**Image Details:**")
    st.write(f"- Size: {image.size}")
    st.write(f"- Mode: {image.mode}")
    st.write(f"- Format: {uploaded_file.type}")

    if st.button("Get Prediction"):
        with st.spinner("Making prediction..."):
            try:
                # Prepare the file for upload
                files = {
                    "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                }
                
                # Make request to Modal endpoint
                response = requests.post(MODAL_ENDPOINT, files=files)
                
                if response.ok:
                    result = response.json()
                    
                    # Display results with the new structured response
                    st.success("🎉 Prediction completed!")
                    
                    # Create columns for better layout
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            label="Predicted Class", 
                            value=result["predicted_class"],
                            help="0 or 1 based on the threshold"
                        )
                    
                    with col2:
                        st.metric(
                            label="Probability", 
                            value=f"{result['probability']:.4f}",
                            help="Raw model output probability"
                        )
                    
                    with col3:
                        st.metric(
                            label="Confidence", 
                            value=f"{result['confidence']:.4f}",
                            help="Distance from decision boundary"
                        )
                    
                    # Add a progress bar for probability
                    st.write("**Probability Visualization:**")
                    st.progress(result['probability'])
                    
                    # Show threshold
                    st.info(f"Decision threshold: {result['threshold']}")
                    
                    # Add interpretation
                    class_label = "Positive" if result["predicted_class"] == 1 else "Negative"
                    confidence_level = "High" if result["confidence"] > 0.8 else "Medium" if result["confidence"] > 0.6 else "Low"
                    
                    st.write(f"**Interpretation:** The model predicts this image as **{class_label}** with **{confidence_level}** confidence ({result['confidence']:.2%})")
                    
                    # Show raw response for debugging (optional)
                    with st.expander("Show Raw Response"):
                        st.json(result)
                        
                else:
                    st.error(f"❌ Backend error (Status {response.status_code}): {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to backend. Please check if Modal endpoint is running.")
            except requests.exceptions.Timeout:
                st.error("❌ Request timeout. The backend might be overloaded.")
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")

# Add footer with instructions
st.markdown("---")
st.markdown("""
**Instructions:**
1. Upload an image (PNG, JPG, or JPEG)
2. Click 'Get Prediction' to classify the image
3. View the prediction results with probability and confidence scores

**Model Info:** Binary classifier returning probability scores between 0 and 1
""")
