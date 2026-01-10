import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "leaf_modell_mobilenetv2_finetuned.h5"
IMG_SIZE = 160

CLASS_NAMES = [
    "Calophyllum inophyllum L",
    "Dendrolobium umbellatum (L.) Berth",
    "Ficus benjamina L",
    "Hibiscus rosa-sinensis L",
    "Ixora chinensis Lam",
    "Macadamia tetraphylla L",
    "Mangifera indica L",
    "Pandanus amaryllifolius Roxb",
    "Pterocarpus santalinus L",
    "Sassafras albidum",
    "Simarouba glauca DC",
    "Syzygium smithii (Poir.) Nied"
]

# -----------------------------
# MODEL LOADING (CACHED)
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# -----------------------------
# ANATOMY DATABASE (STANDARDIZED)
# -----------------------------
anatomy_data = {
    "Calophyllum inophyllum L": {
        "leaf_type": "Evergreen isobilateral leaf with thick cuticle and resin ducts.",
        "vascular_bundle": "Collateral bundle with sclerenchyma caps.",
        "stomata": "Paracytic stomata, mainly abaxial.",
        "special": "Resin ducts and thick waxy cuticle.",
        "ts_image": "images/calophyllum_ts.jpg",
        "stomata_image": "images/paracytic.png"
    },

    "Dendrolobium umbellatum (L.) Berth": {
        "leaf_type": "Isobilateral xeromorphic leaf with resin ducts.",
        "vascular_bundle": "Collateral bundle with strong sclerenchyma.",
        "stomata": "Paracytic stomata.",
        "special": "Coastal adaptation with thick mesophyll.",
        "ts_image": "images/dendrolobium_ts.jpg",
        "stomata_image": "images/paracytic.png"
    },

    "Ficus benjamina L": {
        "leaf_type": "Dorsiventral hypostomatic leaf with latex ducts.",
        "vascular_bundle": "Collateral bundle with latex canals.",
        "stomata": "Paracytic stomata (Moraceae type).",
        "special": "Latex secretion and leathery texture.",
        "ts_image": "images/ficus_ts.jpg",
        "stomata_image": "images/paracytic.png"
    },

    "Hibiscus rosa-sinensis L": {
        "leaf_type": "Dorsiventral leaf with stellate trichomes.",
        "vascular_bundle": "Bicollateral bundle (Malvaceae feature).",
        "stomata": "Anisocytic stomata.",
        "special": "Mucilage cells and stellate hairs.",
        "ts_image": "images/hibiscus_ts.jpg",
        "stomata_image": "images/anisocytic.png"
    },

    "Ixora chinensis Lam": {
        "leaf_type": "Dorsiventral hypostomatic leaf with raphides.",
        "vascular_bundle": "Collateral arc-shaped bundle.",
        "stomata": "Paracytic stomata.",
        "special": "Calcium oxalate raphides.",
        "ts_image": "images/ixora_ts.jpg",
        "stomata_image": "images/paracytic.png"
    },

    "Macadamia tetraphylla L": {
        "leaf_type": "Xeromorphic coriaceous leaf with hypodermis.",
        "vascular_bundle": "Collateral bundle with fiber caps.",
        "stomata": "Sunken paracytic stomata.",
        "special": "Strong lignification.",
        "ts_image": "images/macadamia_ts.jpg",
        "stomata_image": "images/paracytic.png"
    },

    "Mangifera indica L": {
        "leaf_type": "Coriaceous hypostomatic leaf with resin canals.",
        "vascular_bundle": "Collateral bundle with resin ducts.",
        "stomata": "Anomocytic stomata.",
        "special": "Aromatic resin canals.",
        "ts_image": "images/mango_ts.jpg",
        "stomata_image": "images/anomocytic.png"
    },

    "Pandanus amaryllifolius Roxb": {
        "leaf_type": "Isobilateral monocot leaf with air lacunae.",
        "vascular_bundle": "Parallel closed bundles.",
        "stomata": "Paracytic stomata in rows.",
        "special": "Silica bodies and aromatic oils.",
        "ts_image": "images/pandanus_ts.jpg",
        "stomata_image": "images/paracytic.png"
    },

    "Pterocarpus santalinus L": {
        "leaf_type": "Dorsiventral trifoliate leaf.",
        "vascular_bundle": "Bicollateral bundle.",
        "stomata": "Paracytic stomata.",
        "special": "Tannin cells.",
        "ts_image": "images/pterocarpus_ts.jpg",
        "stomata_image": "images/paracytic.png"
    },

    "Sassafras albidum": {
        "leaf_type": "Dorsiventral leaf with oil idioblasts.",
        "vascular_bundle": "Collateral bundle with oil glands.",
        "stomata": "Anomocytic stomata.",
        "special": "Aromatic oil cells.",
        "ts_image": "images/sassafras_ts.jpg",
        "stomata_image": "images/anomocytic.png"
    },

    "Simarouba glauca DC": {
        "leaf_type": "Dorsiventral leaf with resin canals.",
        "vascular_bundle": "Collateral bundle with sclerenchyma.",
        "stomata": "Paracytic stomata.",
        "special": "Bitter oil canals.",
        "ts_image": "images/simarouba_ts.jpg",
        "stomata_image": "images/paracytic.png"
    },

    "Syzygium smithii (Poir.) Nied": {
        "leaf_type": "Xeromorphic leaf with oil cavities.",
        "vascular_bundle": "Collateral crescent-shaped bundle.",
        "stomata": "Paracytic / Anomocytic.",
        "special": "Aromatic oil glands.",
        "ts_image": "images/syzygium_ts.jpg",
        "stomata_image": "images/paracytic.png"
    }
}

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict_species(img_pil):
    img = img_pil.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)[0]
    idx = np.argmax(preds)
    return CLASS_NAMES[idx], preds[idx] * 100

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🌿 Leaf X-Ray AI")
st.caption("AI-Assisted Digital Reconstruction of Leaf Anatomy")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf", use_container_width=True)

    species, confidence = predict_species(image)

    st.subheader(f"Predicted Species: {species}")
    st.write(f"Confidence: **{confidence:.2f}%**")

    if confidence < 70:
        st.warning("Low confidence prediction. Please upload a clearer image.")

    anatomy = anatomy_data[species]

    st.subheader("📘 Anatomical Features")
    st.markdown(f"**Leaf Type:** {anatomy['leaf_type']}")
    st.markdown(f"**Vascular Bundle:** {anatomy['vascular_bundle']}")
    st.markdown(f"**Stomata:** {anatomy['stomata']}")
    st.markdown(f"**Special Features:** {anatomy['special']}")

    st.subheader("🌱 Transverse Section (AI Reference)")
    st.image(anatomy["ts_image"], use_container_width=True)

    st.subheader("🌬️ Stomatal Type")
    st.image(anatomy["stomata_image"], width=200)

st.subheader("🔬 Vascular Bundle Types")
st.image("images/vascular_bundle.jpg", use_container_width=True)
