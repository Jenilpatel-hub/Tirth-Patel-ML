import pickle
import numpy as np
import cv2
import os

MODEL_PATH = r"C:\Users\tirth\OneDrive\Desktop\Tirth ML project\trained_lung_colon_model.pkl"

print("=" * 50)
print("STEP 1: Loading model...")
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("✅ Model loaded:", type(model))
except Exception as e:
    print("❌ Model load failed:", e)
    exit()

print("\nSTEP 2: Checking model type...")
print("Model class:", model.__class__.__name__)
print("Model module:", model.__class__.__module__)

# Check if it's a Keras/TF model or sklearn model
try:
    print("Model input shape:", model.input_shape)
    print("✅ It's a Keras model")
    model_type = "keras"
except:
    print("Not a Keras model — likely sklearn/pickle")
    model_type = "sklearn"

print("\nSTEP 3: Creating dummy image input...")
dummy = np.zeros((1, 224, 224, 3), dtype="float32")
print("Dummy input shape:", dummy.shape)
print("Dummy dtype:", dummy.dtype)

print("\nSTEP 4: Running prediction...")
try:
    pred = model.predict(dummy)
    print("✅ Prediction succeeded!")
    print("Output shape:", np.array(pred).shape)
    print("Output:", pred)
except Exception as e:
    print("❌ Prediction failed:", e)

    # Try alternative input shapes
    print("\nTrying different input shapes...")
    for shape in [(1, 224, 224, 1), (1, 150, 150, 3), (1, 64, 64, 3)]:
        try:
            test = np.zeros(shape, dtype="float32")
            out = model.predict(test)
            print(f"✅ Shape {shape} works! Output: {np.array(out).shape}")
            break
        except Exception as e2:
            print(f"❌ Shape {shape} failed: {e2}")

print("\nSTEP 5: Checking expected input shape...")
try:
    print("Input shape:", model.input_shape)
except:
    try:
        print("Layers[0] input:", model.layers[0].input_shape)
    except:
        print("Could not determine input shape")

print("=" * 50)
print("DONE. Share the output above to fix the issue.")