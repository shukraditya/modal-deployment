# Model Deployment with Streamlit Frontend and Modal Backend

## Streamlit Frontend Features

- **Image Upload**: Upload PNG, JPG, or JPEG images for classification
- **Real-time Prediction**: Get instant binary classification results
- **Health Monitoring**: Check backend status with health endpoint
- **Visual Results**: Display prediction probability, confidence scores, and progress bars
- **Error Handling**: Graceful error messages for connection issues
- **Responsive UI**: Clean interface with metrics and detailed interpretation

## Deploy Modal Backend

1. **Install Modal CLI**:
   ```bash
   pip install modal
   modal token new
   ```

2. **Deploy the backend**:
   ```bash
   cd backend
   modal deploy modal_app.py
   ```

3. **Update endpoints** in `frontend/streamlit_app.py` with your Modal URLs

4. **Run Streamlit**:
   ```bash
   cd frontend
   streamlit run streamlit_app.py
   ```

The backend uses TensorFlow/Keras for binary image classification with GPU acceleration on Modal.


Hosted: 
https://modal-shukra-dep.streamlit.app/ 
