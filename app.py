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
# ANATOMY DATA
# -----------------------------
anatomy_data =ANATOMY = {
    "Calophyllum inophyllum L": {
        "leaf_type": (
            "Evergreen isobilateral leaf with a thick, shiny, waxy cuticle. "
            "Both adaxial and abaxial sides contain palisade tissue as an adaptation "
            "to intense coastal sunlight. Resin ducts occur in the mesophyll as elongated "
            "secretory canals that aid in defense."
        ),
        "Calophyllum": "Dicot.jpg",

        "vascular_bundle": (
            "A large **Collateral** midrib bundle with xylem on the adaxial side and phloem on the abaxial side. "
            "Thick sclerenchymatous fiber caps reinforce the bundle."
        ),
        "stomata": (
            "Predominantly **Paracytic** stomata with two subsidiary cells parallel to guard cells; mainly abaxial."
        ),
        "stomata_images": "paracytic.png",
        "special": "Resin ducts abundant; thick cuticle and sclerenchyma present",
        "Systematic Position": ("")
    },

    "Dendrolobium umbellatum (L.) Berth": {
        "leaf_type": (
             "The leaf is evergreen and isobilateral, exhibiting a thick, glossy cuticle that reflects strong sunlight in coastal habitats. "
            "Both surfaces possess palisade layers, an adaptation that enhances photosynthetic efficiency."
            "Resin ducts occur in the mesophyll,appearing as elongated cavities filled with viscous secretions that contribute to the plant’s defense."
            "The mesophyll is densely packed, indicating xeromorphic features suited for salt-spray environments."
        ),
        "Dendrolobium": "Dendrobolium umbellatum.png.",

        "vascular_bundle": (
            "Large **Collateral** bundle with xylem on upper side, phloem below. "
            "Strong sclerenchymatous caps and resin canals alongside bundles."
        ),
        "stomata": (
            "Predominantly **Paracytic** stomata; two subsidiary cells parallel to guard cells; abaxial distribution."
        ),
        "stomata_images": "paracytic.png",
        "special": "Compound leaflet structure; terminal leaflet thickened."
    },

    "Ficus benjamina L": {
        "leaf_type": (
            "Epidermis Adaxial (upper) epidermis: Single-layered, compact, covered by a thick, smooth cuticle (reduces water loss). Epidermal cells are larger, rectangular, with straight anticlinal walls. Stomata are absent or extremely rare (hypostomatic leaf). Abaxial (lower) epidermis: Single-layered, thinner cuticle than the upper side. Contains numerous anemocytic stomata (surrounded by subsidiary cells with no fixed arrangement, typical of Moraceae). Stomatal density is moderate Trichomes are usually absent in mature leaves, though young leaves may have sparse cystoliths or hairs.Mesophyll"
             "Well-differentiated into:Palisade parenchyma:  (sometimes 3) layers of elongated, cylindrical, tightly packed chlorenchyma cells rich in chloroplasts. The palisade layer is thicker on the adaxial side (typical sun/shade adaptation).Spongy parenchyma:  layers of loosely arranged, irregularly shaped cells with large intercellular spaces, facilitating gas exchange. Fewer chloroplasts than palisade tissue."
    
        ),
        "Ficus": "Ficus.jpg",

        "vascular_bundle": (
            "The midrib contains a **Collateral bundle** accompanied by latex ducts on both the adaxial and abaxial sides. Xylem elements are highly lignified, supporting efficient transport. Mechanical tissue is abundant, giving the leaf a leathery texture.. "
            "Latex ducts flank vascular tissue."
        ),
        "stomata": "Stomata are **Paracytic**, with parallel subsidiary cells, typical of Moraceae. They are generally confined to the abaxial surface to minimize water loss..",
        "special": "Abundant latex canals; stiff leathery leaf.",
        "stomata_images": "paracytic.png"
    },

    "Hibiscus rosa-sinensis L": {
        "leaf_type": (
            "Epidermis Both adaxial and abaxial epidermis are single-layered.Adaxial epidermis: Cells larger, rectangular to polygonal, walls slightly wavy, covered by a thick, smooth cuticle.Abaxial epidermis: Cells smaller, anticlinal walls more wavy or sinuous. Stomata: Present only on the lower (abaxial) surface → hypostomatic. Stomata are ranunculaceous (anomocytic) – surrounded by 3–6 ordinary epidermal cells with no definite arrangement (typical of Malvaceae).Trichomes:Abundant stellate (star-shaped) hairs on the lower surface and along veins (especially in young leaves).Simple unicellular or multicellular unbranched hairs may also be present.Upper surface usually glabrous or sparingly hairy in mature leaves.Mucilage cells: Large mucilage-containing idioblasts scattered in both epidermises (characteristic of Malvaceae). Mesophyll Clearly differentiated into:Palisade parenchyma: 1–2 (rarely 3) layers of long, cylindrical, compactly arranged cells rich in chloroplasts (adaxial side).Spongy parenchyma: 4–6 layers of loosely arranged, lobed cells with large intercellular spaces; fewer chloroplasts.. "
            "Epidermis contains stellate hairs."
        ),
        "Hibiscus": "Hibiscus .png",
        "vascular_bundle": (
            "The bundles are **bicollateral**, with phloem located on both sides of the xylem — a signature trait of Malvaceae. Sclerenchymatous tissue is extensive around the midrib, providing rigidity."
        ),
        "stomata": "Stomata are **Anisocytic**, distinguished by three subsidiary cells surrounding the guard cells, one of which is smaller. This pattern is a diagnostic feature of the Hibiscus genus.",
        "special": "Mucilage cavities and stellate trichomes present.",
        "stomata_imagee": "Anisocytic stoamta.png"
    },

    "Ixora chinensis Lam": {
        "leaf_type": (
            "Chinese Ixora, Rubiaceae) has a dorsiventral, hypostomatic leaf with a thick-cuticled adaxial epidermis devoid of stomata and an abaxial epidermis bearing typical paracytic (rubiaceous) stomata. The mesophyll consists of 1–3 palisade layers (sometimes with an adaxial hypodermis) and 4–7 spongy layers. Numerous raphide bundles in idioblasts are a hallmark of the family. The midrib contains a single arc-shaped collateral bundle. Trichomes are usually absent in mature leaves."
        ),
        "ixora": "ixora .jpg",
        "vascular_bundle": (
            "Vascular bundles are **Collateral**, with a narrow xylem strand and a relatively broad phloem region. The bundle sheath is distinct and often lignified. Minor veins form a fine reticulum characteristic of Rubiaceae."
        ),
        "stomata": "Stomata are **Paracytic**, a dominant character in Rubiaceae. They may appear slightly sunken, reducing transpiration.",
        "special": "Glossy leaves with xeromorphic features.",
        "stomata_images": "paracytic.png"
    },

    "Macadamia tetraphylla L": {
        "leaf_type": (
            "(Rough-shelled Macadamia, Proteaceae) shows a thick, coriaceous, strongly xeromorphic, hypostomatic leaf with a very thick cuticle on both surfaces. A 1–3-layered adaxial hypodermis and 2–4 palisade layers dominate the compact mesophyll, while scattered branched sclereids give a gritty texture. Stomata are paracytic and sunken on the abaxial side. The midrib houses a large collateral bundle. Mature leaves are glabrous."
        ),
        "Macadamia":"macademia.jpg",
        "vascular_bundle": (
            "Bundles are **Collateral**, surrounded by tough fiber caps forming a protective sheath. The xylem is narrow but heavily lignified. The structure supports efficient water conservation."
        ),
        "stomata": "Stomata are **Paracytic**, often sunken into small depressions that reduce water loss in dry conditions.",
        "special": "Strong xeric adaptation; lignified tissues.",
        "stomata_images": "paracytic.png"
    },

    "Mangifera indica L": {
        "leaf_type": (
            " (Mango, Anacardiaceae) produces glossy, coriaceous, hypostomatic leaves with a very thick adaxial cuticle (10–20 µm) and 1–2 layers of adaxial hypodermis in many cultivars. The mesophyll has 2–4 palisade layers and abundant schizogenous resin canals throughout, along with druses of calcium oxalate. Sunken paracytic stomata occur abaxially. The midrib exhibits a large bicollateral bundle with accessory bundles and conspicuous resin canals — classic Anacardiaceae features."
        ),
        "Mangifera": "mangniferas.png",
        "vascular_bundle": (
            "The vascular bundle is **Collateral**, with a broad zone of xylem and adjacent resin ducts. Sclerenchyma forms a strong cap around the bundle, giving the leaf notable stiffness."
        ),
        "stomata": "Stomata are predominantly **Anomocytic**, with guard cells surrounded by a random arrangement of epidermal cells without defined subsidiary cells.",
        "special": "Aromatic resin canals; tough midrib.",
        "stomata_image": "Anomyctic stomata.png"
    },

    "Pandanus amaryllifolius Roxb": {
        "leaf_type": (
            "Fragrant Pandan, Pandanaceae), being a monocot, has linear, pleated, isobilateral, amphistomatic leaves with heavily silicified epidermises and thick cuticle on both surfaces. Tetracytic stomata are arranged in rows, and the homogeneous chlorenchyma lacks palisade–spongy differentiation. Large longitudinal air lacunae with stellate diaphragms, silica stegmata, raphide idioblasts, and aromatic oil cells (containing 2-acetyl-1-pyrroline) are characteristic. Numerous parallel collateral bundles with double sheaths run through the blade."
        ),
        "Pandanus": "pandanus.jpg",

        "vascular_bundle": (
            "Bundles are closed and **parallel** each with a bundle sheath of sclerenchyma cells. Xylem and phloem lie side by side without cambium. The fibrous margins of the leaf arise from mechanical tissues adjacent to the bundles."
        ),
        "stomata": "Stomata are **Paracytic**, frequently arranged parallel to the long axis of the leaf — a typical monocot condition.",
        "special": "Aromatic oils present; fibrous margins.",
        "stomata_images": "paracytic.png"
    },

    "Pterocarpus santalinus L": {
        "leaf_type": (
            " (Red Sandalwood, Fabaceae) bears trifoliolate, dorsiventral, hypostomatic leaflets with paracytic stomata abaxially. The mesophyll comprises 2–3 palisade layers and 4–6 spongy layers. Prismatic crystals, druses, and tanniniferous cells (reddish-brown) are abundant. The midrib contains an arc-shaped collateral bundle. Mature leaflets are nearly glabrous."
        ),
        "Pterocarpus": "pterocarpusjpg.jpg",
        "vascular_bundle": (
            "Bundles are **Bicollateral**, having phloem both above and below the xylem. Thick sclerenchyma ribs occur along the midrib, supporting structural integrity."
        ),
        "stomata": "**Paracytic** stomata typical of Fabaceae.",
        "special": "Resinous secretory cells.",
        "stomata_images": "paracytic.png"
    },

    "Sassafras albidum": {
        "leaf_type": (
            "(Sassafras, Lauraceae) has dorsiventral, hypostomatic leaves with paracytic stomata on the lower surface. The mesophyll shows 1–2 palisade layers and 4–7 spongy layers. Large spherical oil/mucilage idioblasts containing aromatic compounds (safrole, etc.) are scattered throughout — the most distinctive Lauraceae feature in this species. Calcium oxalate crystals are rare or absent."
        ),
        "Sassafras": "sassafras.jpg",
        "vascular_bundle": (
            "The vascular strand is **Collateral**, with oil glands located near the phloem. The xylem is lignified and arranged in radial clusters."
        ),
        "stomata": "Stomata are **Anomocytic**, with no distinct subsidiary cells, which is characteristic of Lauraceae.",
        "special": "Young leaves have fine trichomes.",
        "stomata_image": "Anomyctic stomata.png"
    },

    "Simarouba glauca DC": {
        "leaf_type": (
            "(Paradise Tree, Simaroubaceae) exhibits dorsiventral, hypostomatic leaves with sunken paracytic stomata abaxially. The mesophyll has 2–4 palisade layers and contains schizogenous resin/oil canals plus abundant prismatic crystals and crystal sand. The midrib shows a large collateral bundle with sclerenchyma caps and numerous resin canals, giving a bitter taste when chewed."
        ),
        "Simarouba": "simrouba.jpg",
        "vascular_bundle": (
            "Bundles are **Collateral**, with a relatively wide phloem zone. Mechanical tissues occur around the bundle, contributing to leaf rigidity."
        ),
        "stomata": "Stomata are **Paracytic**, similar to many members of Simaroubaceae.",
        "special": "Oil-rich secretory structures.",
        "stomata_images": "paracytic.png"
    },

    "Syzygium smithii (Poir.) Nied": {
        "leaf_type": (
            " (Lilly Pilly, Myrtaceae) produces glossy, xeromorphic, hypostomatic leaves with a very thick adaxial cuticle and a single-layered adaxial hypodermis. The mesophyll is compact, with 2–4 tall palisade layers and numerous large schizogenous oil glands appearing as translucent dots. Sunken paracytic stomata lie abaxially. The midrib has a crescent-shaped collateral bundle surrounded by a thick fibrous sheath."
        ),
        "Syzygium": "syzgium.png",
        "vascular_bundle": (
            "Bundles are **collateral**, reinforced by fiber caps on both adaxial and abaxial sides. The xylem is moderately developed, while phloem is compact and efficient."
        ),
        "stomata": "Stomata may be **paracytic or anomocytic**, depending on leaf maturity. The mixed stomatal pattern reflects ecological adaptability.",
        "special": "Aromatic leaves; abundant oil cavities.",
        "stomata_images": "paracytic.png"
    }
}



# -----------------------------
# SAFE IMAGE LOADING
# -----------------------------
def safe_show_image(path, caption):
    try:
        st.image(path, caption=caption, use_container_width=True)
    except:
        pass

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict_species(model, img_pil):
    img = img_pil.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)[0]
    idx = np.argmax(preds)
    species = CLASS_NAMES[idx]
    confidence = preds[idx] * 100
    return species, confidence, preds

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🌿 AI Leaf Species Classifier + Anatomy Extractor")
st.markdown("""
### 🧪🔬 Traditional Cross-Section Preparation vs AI-Assisted Anatomy

In classical plant anatomy, obtaining a good transverse section (T.S.) of a leaf or stem requires high manual skill.  
Students usually cut sections using a razor blade, and most sections become:

- Too thick  
- Uneven or folded  
- Torn or crushed  
- Difficult to stain properly  
- Hard to observe clearly under the microscope  

Achieving a perfect, thin section often takes many attempts, causing frustration and consuming time in practical classes.

### 🤖🌿 How the AI Model Solves This Problem

Your AI-assisted anatomy system overcomes these limitations by:

- Providing **clear, labelled Transverse section  of Leaf or Leaf Structure as Seen in Transverse Section  (AI Generated)** without needing a perfect physical section  
- Allowing even low-quality images to be used for analysis  
- Giving **instant identification** of tissues and systematic positions  
- Reducing dependence on manual microscopic skills  
- Helping beginners learn anatomy faster with error-free visual references  

This approach makes plant anatomy easier, more accurate, and accessible for all students.
            
""")

st.subheader(" Important — Model Scope and Scientific Approach")

st.write("""
- This model has been trained exclusively on images belonging to the following **12 plant species**:  
  1. *Calophyllum inophyllum L.*  
  2. *Dendrolobium umbellatum (L.) Berth.*  
  3. *Ficus benjamina L.*  
  4. *Hibiscus rosa-sinensis L.*  
  5. *Ixora chinensis Lam.*  
  6. *Macadamia tetraphylla L.*  
  7. *Mangifera indica L.*  
  8. *Pandanus amaryllifolius Roxb.*  
  9. *Pterocarpus santalinus L.*  
  10. *Sassafras albidum*  
  11. *Simarouba glauca DC.*  
  12. *Syzygium smithii (Poir.) Nied*  

- The model’s classification based on morphological and anatomical features, such as venation patterns, epidermal structures, leaf symmetry, surface texture, and mesophyll characteristics.
""")



st.write("Upload a leaf image to identify the species anatomical features.")

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Leaf", use_container_width=True)

    model = tf.keras.models.load_model(MODEL_PATH)

    species, conf, full_preds = predict_species(model, image)

    st.subheader(f"Predicted Species: **{species}**")
    st.write(f"**Confidence:** {conf:.2f}%")

    

    anatomy = anatomy_data.get(species, {})
   

    st.subheader("📘 Anatomical Features")
    st.write(f"### 🌿 Leaf Anatomy\n{anatomy.get('leaf_type')}")
    st.write(f"### 🌱 Vascular Bundle\n{anatomy.get('vascular_bundle')}")
    st.write(f"### 🌬️ Stomatal Type\n{anatomy.get('stomata')}")
    st.write(f"### 🔍 Special Notes\n{anatomy.get('special')}")

    st.subheader("🌱 Transverse section of Leaf")
    

    if "Calophyllum" in anatomy:
        safe_show_image(anatomy["Calophyllum"], "T.S Of Leaf")
    if "Dendrolobium" in anatomy:
        safe_show_image(anatomy["Dendrolobium"], "T.S Of Leaf")
    if "Ficus" in anatomy:
        safe_show_image(anatomy["Ficus"], "T.S Of Leaf")
    if "Hibiscus" in anatomy:
        safe_show_image(anatomy["Hibiscus"], "T.S Of Leaf")
    if "ixora" in anatomy:
        safe_show_image(anatomy["ixora"], "T.S Of Leaf")
    if "Macadamia" in anatomy:
        safe_show_image(anatomy["Macadamia"], "T.S Of Leaf")
    if "Mangifera" in anatomy:
        safe_show_image(anatomy["Mangifera"], "T.S Of Leaf")
    if "Pandanus" in anatomy:
        safe_show_image(anatomy["Pandanus"], "T.S Of Leaf")
    if "Pterocarpus" in anatomy:
        safe_show_image(anatomy["Pterocarpus"], "T.S Of Leaf")
    if "Sassafras" in anatomy:
        safe_show_image(anatomy["Sassafras"], "T.S Of Leaf")
    if "Simarouba" in anatomy:
        safe_show_image(anatomy["Simarouba"], "T.S Of Leaf")
    if "Syzygium" in anatomy:
        safe_show_image(anatomy["Syzygium"], "T.S Of Leaf")

    
    st.subheader("🌿🔍 Stomata Photograph")

    if "stomata_images" in anatomy:
        st.image(anatomy["stomata_images"], "paracytic Stomatal Type",width=200)
        st.markdown("### 🌿🔍 **Paracytic Stomatal Type**")

    if "stomata_image" in anatomy:
        st.image(anatomy["stomata_image"], "Anomyctic Stomatal Type",width=200)
        st.markdown("### 🌿🔍 **Anomyctic Stomatal Type**")

    if "stomata_imagee" in anatomy:
        st.image(anatomy["stomata_imagee"], "Anisocytic Stomatal Type",width=200)
        st.markdown("### 🌿🔍 **Anisocytic Stomatal Type**")

    st.subheader("🌱 Vascular  Bundle")
    st.image("e:/GUIESS RECORD/vacular bundles/vascular bundle.jpg", caption="Types of Vascular bundle", use_container_width=True)



