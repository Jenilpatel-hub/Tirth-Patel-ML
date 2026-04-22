from flask import Flask, render_template, request, jsonify
import os
import pickle
import sqlite3
from datetime import datetime
import numpy as np
import cv2
import uuid
import traceback

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
DB_FILE = "database.db"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ---------------- MODEL LOADING ----------------
model = None
INPUT_SIZE = (128, 128)

MODEL_PATH = r"C:\Users\tirth\OneDrive\Desktop\Tirth ML project\trained_lung_colon_model.pkl"

try:
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file not found at: {MODEL_PATH}")
    else:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("[OK] Model loaded. Input shape:", model.input_shape)
except Exception as e:
    print("[ERROR] Model loading failed:", e)
    traceback.print_exc()


# ---------------- RISK LEVEL LOGIC ----------------
def get_risk_level(prediction, confidence):
    """
    Normal  + any confidence  → Low
    Malignant + < 60%         → Moderate
    Malignant + 60–79%        → High
    Malignant + 80%+          → Critical
    """
    if prediction == "Normal":
        return "Low"
    else:
        if confidence < 60:
            return "Moderate"
        elif confidence < 80:
            return "High"
        else:
            return "Critical"


# ---------------- DATABASE ----------------
def init_db():
    try:
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS history(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image TEXT,
            prediction TEXT,
            confidence REAL,
            risk_level TEXT DEFAULT 'Low',
            scan_type TEXT DEFAULT 'ct_scan',
            created_at TEXT
        )
        """)
        # Migrate old DB — add columns if missing
        for col, default in [("scan_type", "'ct_scan'"), ("risk_level", "'Low'")]:
            try:
                cur.execute(f"ALTER TABLE history ADD COLUMN {col} TEXT DEFAULT {default}")
                print(f"[OK] Added column: {col}")
            except:
                pass
        conn.commit()
        conn.close()
        print("[OK] Database ready")
    except Exception as e:
        print("[ERROR] DB init failed:", e)

init_db()


# ---------------- HOME ----------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------------- PREDICT ----------------
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Check the model file path."}), 500

    filepath = None

    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        allowed_extensions = {".jpg", ".jpeg", ".png"}
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in allowed_extensions:
            return jsonify({"error": "Invalid file type. Use PNG, JPG or JPEG."}), 400

        scan_type = request.form.get("scan_type", "ct_scan")
        if scan_type not in {"blood_report", "ct_scan", "xray"}:
            scan_type = "ct_scan"

        filename = str(uuid.uuid4()) + ".jpg"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        print(f"[OK] File saved: {filepath} | Type: {scan_type}")

        img = cv2.imread(filepath)
        if img is None:
            os.remove(filepath)
            return jsonify({"error": "Could not read image file."}), 400

        img = cv2.resize(img, INPUT_SIZE)
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img, verbose=0)
        pred = np.array(pred)
        print(f"[OK] Raw prediction: {pred}")

        if pred.shape[-1] == 1:
            prob = float(pred[0][0])
            if prob > 0.5:
                prediction = "Malignant"
                confidence = round(prob * 100, 2)
            else:
                prediction = "Normal"
                confidence = round((1 - prob) * 100, 2)
        else:
            classes = ["Normal", "Malignant"]
            idx = int(np.argmax(pred[0]))
            prediction = classes[idx]
            confidence = round(float(pred[0][idx]) * 100, 2)

        risk_level = get_risk_level(prediction, confidence)
        print(f"[OK] Result: {prediction} ({confidence}%) → Risk: {risk_level}")

        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO history(image, prediction, confidence, risk_level, scan_type, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (filepath, prediction, confidence, risk_level, scan_type,
              datetime.now().strftime("%d-%m-%Y %H:%M")))
        conn.commit()
        conn.close()

        return jsonify({
            "prediction": prediction,
            "confidence": confidence,
            "risk_level": risk_level,
            "image": filepath,
            "scan_type": scan_type
        })

    except Exception as err:
        traceback.print_exc()
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({"error": "Prediction failed", "details": str(err)}), 500


# ---------------- HISTORY ----------------
@app.route("/history")
def history():
    try:
        scan_type = request.args.get("scan_type", None)
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()

        if scan_type:
            cur.execute("SELECT * FROM history WHERE scan_type=? ORDER BY id DESC", (scan_type,))
        else:
            cur.execute("SELECT * FROM history ORDER BY id DESC")

        rows = cur.fetchall()
        conn.close()

        # columns: id, image, prediction, confidence, risk_level, scan_type, created_at
        data = []
        for row in rows:
            data.append({
                "id":         row[0],
                "image":      row[1],
                "prediction": row[2],
                "confidence": row[3],
                "risk_level": row[4] if len(row) > 4 else "Low",
                "scan_type":  row[5] if len(row) > 5 else "ct_scan",
                "time":       row[6] if len(row) > 6 else ""
            })

        return jsonify(data)

    except Exception as err:
        traceback.print_exc()
        return jsonify([])


# ---------------- DELETE ----------------
@app.route("/delete/<int:id>")
def delete(id):
    try:
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        cur.execute("SELECT image FROM history WHERE id=?", (id,))
        row = cur.fetchone()
        if row and row[0] and os.path.exists(row[0]):
            os.remove(row[0])
        cur.execute("DELETE FROM history WHERE id=?", (id,))
        conn.commit()
        conn.close()
        return jsonify({"success": True})
    except Exception as err:
        traceback.print_exc()
        return jsonify({"error": str(err)}), 500


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)