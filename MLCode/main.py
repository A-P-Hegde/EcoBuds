import cv2
import numpy as np
import glob
import joblib
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# ---------------- Feature Extraction ----------------
def extract_features_from_contour(cnt, hsv, gray):
    area = cv2.contourArea(cnt)
    if area < 50:  # ignore tiny particles/noise
        return None

    perimeter = cv2.arcLength(cnt, True)
    circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter != 0 else 0
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h if h != 0 else 0

    mask = np.zeros(hsv.shape[:2], np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)

    mean_v, std_v = cv2.meanStdDev(hsv[:, :, 2], mask=mask)
    mean_h = cv2.mean(hsv[:, :, 0], mask=mask)[0]

    # Texture: Local Binary Pattern (LBP) histogram
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp[mask == 255], bins=np.arange(0, 11), density=True)

    return [mean_v[0][0], std_v[0][0], mean_h, circularity, aspect_ratio] + hist.tolist()

# ---------------- Build Dataset ----------------
def build_dataset():
    X, y = [], []
    for label, folder in enumerate(["plastics", "organics"]):
        for path in glob.glob(f"{folder}/*.jpg"):
            img = cv2.imread(path)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 3)

            # Threshold on brightness (V channel)
            _, thresh = cv2.threshold(
                hsv[:, :, 2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for cnt in contours:
                feats = extract_features_from_contour(cnt, hsv, gray)
                if feats is not None:
                    X.append(feats)
                    y.append(label)
    return np.array(X), np.array(y)

# ---------------- Train & Save Model ----------------
def train_and_save_model():
    X, y = build_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1, use_label_encoder=False
    )
    clf.fit(X_train, y_train, eval_metric="logloss")

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Cross-validation check
    scores = cross_val_score(clf, X, y, cv=5)
    print("Cross-val accuracy:", scores.mean())

    joblib.dump(clf, "plastic_model.pkl")
    print("✅ Model trained and saved as plastic_model.pkl")

# ---------------- Load Model ----------------
def load_model():
    return joblib.load("plastic_model.pkl")

# ---------------- Concentration Estimation ----------------
def estimate_concentration(
    image_path,
    model,
    sample_volume_liters,
    filter_area_mm2,
    fov_area_mm2,
    num_images_taken,
    particle_area_estimate=200,
    prob_threshold=0.7,
):
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)

    _, thresh = cv2.threshold(
        hsv[:, :, 2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    plastic_count = 0
    total_count = 0

    for cnt in contours:
        feats = extract_features_from_contour(cnt, hsv, gray)
        if feats is None:
            continue
        total_count += 1
        probs = model.predict_proba([feats])[0]
        plastic_prob = probs[0]  # class 0 = plastic
        if plastic_prob > prob_threshold:
            area = cv2.contourArea(cnt)
            est_particles = max(1, int(area / particle_area_estimate))
            plastic_count += est_particles

    ratio = plastic_count / max(1, total_count)

    # Scale up to full filter → per liter
    particles_per_filter = (plastic_count / num_images_taken) * (
        filter_area_mm2 / fov_area_mm2
    )
    particles_per_liter = particles_per_filter / sample_volume_liters

    return {
        "plastic_count_in_image": plastic_count,
        "total_particles_in_image": total_count,
        "plastic_ratio_in_image": ratio,
        "estimated_particles_per_liter": particles_per_liter,
    }

# ---------------- Example Usage ----------------
if __name__ == "__main__":
    # Train only once, then comment this out
    # train_and_save_model()

    model = load_model()
    result = estimate_concentration(
        image_path="test.jpg",
        model=model,
        sample_volume_liters=1.0,
        filter_area_mm2=50000,  # example: full filter size
        fov_area_mm2=2.0,       # microscope field of view area
        num_images_taken=50,    # how many FOVs captured
    )
    print(result)
