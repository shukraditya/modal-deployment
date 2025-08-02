import modal
from fastapi import UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import io

# Image with required dependencies
image = modal.Image.debian_slim().pip_install(
    "fastapi[all]", "tensorflow", "Pillow", "numpy"
)

app = modal.App("keras-image-endpoint", image=image)
volume = modal.Volume.from_name("model-weights-vol", create_if_missing=True)
MODEL_DIR = "/data"
MODEL_PATH = "/data/data/full_model_task_A.h5"

@app.cls(gpu="any", volumes={MODEL_DIR: volume})
class ModelCtx:
    model = None
    model_loaded = False

    @modal.enter()
    def load(self):
        from tensorflow.keras.models import load_model
        import os
        import traceback
        import time
        
        print("Starting model loading process...")
        
        # Wait for volume to be available with retry logic
        max_retries = 10
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                print(f"Attempt {attempt + 1}/{max_retries}: Checking volume availability...")
                
                # CRITICAL: Reload volume to see latest changes
                volume.reload()
                
                # Wait a moment for volume to be fully mounted
                time.sleep(1)
                
                # Check if volume directory exists and is accessible
                if not os.path.exists(MODEL_DIR):
                    print(f"Volume directory {MODEL_DIR} not found, retrying...")
                    time.sleep(retry_delay)
                    continue
                
                # Check if model file exists
                model_found = False
                actual_model_path = None
                
                # Check primary path
                if os.path.exists(MODEL_PATH):
                    model_found = True
                    actual_model_path = MODEL_PATH
                    print(f"Model found at: {MODEL_PATH}")
                else:
                    print(f"Model file not found at {MODEL_PATH}")
                    # List files in directory for debugging
                    try:
                        files = os.listdir(MODEL_DIR)
                        print(f"Available files in {MODEL_DIR}: {files}")
                        
                        # Check if there's a data subdirectory
                        data_dir_path = os.path.join(MODEL_DIR, "data")
                        if os.path.exists(data_dir_path):
                            print(f"Found data subdirectory, checking contents...")
                            subfiles = os.listdir(data_dir_path)
                            print(f"Files in {data_dir_path}: {subfiles}")
                    except Exception as e:
                        print(f"Could not list directory {MODEL_DIR}: {e}")
                    
                    if attempt < max_retries - 1:
                        print(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        print("Max retries reached, model file not found")
                        self.model = None
                        self.model_loaded = False
                        return
                
                # If we get here, the file exists, try to load it
                print(f"Model file found! Size: {os.path.getsize(actual_model_path)} bytes")
                
                try:
                    self.model = load_model(actual_model_path, compile=False)
                    self.model_loaded = True
                    print(f"Model loaded successfully from {actual_model_path}!")
                    print(f"Model input shape: {self.model.input_shape}")
                    print(f"Model output shape: {self.model.output_shape}")
                    return  # Success, exit the retry loop
                    
                except Exception as e:
                    print(f"Model loading failed on attempt {attempt + 1}: {e}")
                    traceback.print_exc()
                    
                    if attempt < max_retries - 1:
                        print(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        print("Max retries reached, model loading failed")
                        self.model = None
                        self.model_loaded = False
                        return
                        
            except Exception as e:
                print(f"Volume check failed on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    print("Max retries reached, volume unavailable")
                    self.model = None
                    self.model_loaded = False
                    return
        
        # If we get here, all retries failed
        print("All retry attempts failed")
        self.model = None
        self.model_loaded = False

ctx = ModelCtx()

@app.function(gpu="any", volumes={MODEL_DIR: volume})
@modal.fastapi_endpoint(method="GET")
async def health():
    """Health check endpoint to verify model status"""
    import os
    import time
    
    # Also reload volume here for fresh status
    volume.reload()
    
    # Wait a moment for volume to be fully available
    time.sleep(0.5)
    
    model_exists = os.path.exists(MODEL_PATH)
    volume_dir_exists = os.path.exists(MODEL_DIR)
    
    # Get additional debugging info
    volume_info = {
        "volume_dir_exists": volume_dir_exists,
        "model_exists": model_exists,
        "model_path": MODEL_PATH
    }
    
    if volume_dir_exists:
        try:
            files = os.listdir(MODEL_DIR)
            volume_info["files_in_volume"] = files
            volume_info["file_count"] = len(files)
            
            # Check data subdirectory if it exists
            data_dir_path = os.path.join(MODEL_DIR, "data")
            if os.path.exists(data_dir_path):
                subfiles = os.listdir(data_dir_path)
                volume_info["files_in_data_subdir"] = subfiles
                volume_info["data_subdir_exists"] = True
        except Exception as e:
            volume_info["list_error"] = str(e)
    
    if model_exists:
        try:
            volume_info["model_size"] = os.path.getsize(MODEL_PATH)
        except Exception as e:
            volume_info["size_error"] = str(e)
    
    return JSONResponse({
        "status": "healthy" if ctx.model_loaded else "unhealthy",
        "model_loaded": ctx.model_loaded,
        "volume_info": volume_info,
        "timestamp": time.time()
    })

@app.function(volumes={MODEL_DIR: volume})
@modal.fastapi_endpoint(method="GET")
async def list_files():
    """Debug endpoint to list files in the volume"""
    import os
    
    # Reload volume before listing
    volume.reload()
    
    try:
        files = os.listdir(MODEL_DIR)
        file_details = []
        for f in files:
            full_path = os.path.join(MODEL_DIR, f)
            if os.path.isfile(full_path):
                file_details.append({
                    "name": f,
                    "size": os.path.getsize(full_path),
                    "is_model": f == "full_model_task_A.h5"
                })
            else:
                file_details.append({
                    "name": f,
                    "type": "directory"
                })
        
        return JSONResponse({
            "model_dir": MODEL_DIR,
            "files": files,
            "file_details": file_details,
            "model_exists": os.path.exists(MODEL_PATH)
        })
    except Exception as e:
        return JSONResponse({"error": str(e)})

@app.function(gpu="any", volumes={MODEL_DIR: volume})
@modal.fastapi_endpoint(method="POST")
async def predict(file: UploadFile = File(...)):
    import os
    import time
    
    # Check if model is loaded
    if not ctx.model_loaded or ctx.model is None:
        print("Model not loaded, attempting to load it now...")
        
        # Try reloading the model with retry logic
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                volume.reload()
                time.sleep(0.5)  # Wait for volume to be available
                
                if os.path.exists(MODEL_PATH):
                    try:
                        from tensorflow.keras.models import load_model
                        ctx.model = load_model(MODEL_PATH, compile=False)
                        ctx.model_loaded = True
                        print(f"Model loaded successfully on retry attempt {attempt + 1} from {MODEL_PATH}!")
                        break
                    except Exception as e:
                        print(f"Failed to load model on retry attempt {attempt + 1}: {e}")
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                            continue
                else:
                    print(f"Model file not found at {MODEL_PATH} on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
            except Exception as e:
                print(f"Volume reload failed on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
        
        if not ctx.model_loaded:
            raise HTTPException(
                status_code=500, 
                detail="Model not loaded after retry attempts. Please check logs and try again."
            )
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    try:
        # Read and preprocess image
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((370, 370))  # adjust to your model's expected input size
        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, 0)

        # Predict
        pred = ctx.model.predict(arr, verbose=0)
        probability = float(pred[0][0])
        
        # Apply threshold for binary classification
        threshold = 0.5
        predicted_class = 1 if probability >= threshold else 0
        confidence = probability if predicted_class == 1 else 1 - probability
        
        return JSONResponse({
            "predicted_class": predicted_class,
            "probability": probability,
            "confidence": confidence,
            "threshold": threshold
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
