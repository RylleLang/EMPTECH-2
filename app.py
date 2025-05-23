import streamlit as st
from PIL import Image
import torch

# Class names from your datasets
veg_class_names = [
    "beet", "bell_pepper", "cabbage", "carrot", "cucumber", "egg", "eggplant",
    "garlic", "onion", "potato", "tomato", "zucchini"
]
meat_class_names = ["beef", "chicken", "pork"]

# Load your YOLOv11 models (replace with actual YOLOv11 loading code)
@st.cache_resource
def load_model(model_path):
    # Replace with actual YOLOv11 model loading
    # Example: return torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    return None

veg_model = load_model("Kleeos-Project-1/yolov11.pt")
meat_model = load_model("Meat-Detection-2/yolov11.pt")

def detect_ingredients(image, model, class_names):
    # Replace with actual YOLOv11 inference code
    # Example:
    # results = model(image)
    # detected = set([class_names[int(cls)] for cls in results.pred[0][:, -1]])
    # return detected
    return set()

RECIPES = {
    "Scrambled Eggs": {
        "ingredients": {"egg", "salt", "butter"},
        "instructions": "Beat eggs with salt, cook in buttered pan."
    },
    "Pancakes": {
        "ingredients": {"flour", "egg", "milk", "sugar", "butter"},
        "instructions": "Mix ingredients and cook on griddle."
    },
    "Tomato Pasta": {
        "ingredients": {"pasta", "tomato", "salt", "olive oil"},
        "instructions": "Cook pasta, prepare tomato sauce, mix and serve."
    },
    "Adobo": {
        "ingredients": {"chicken", "soy sauce", "vinegar", "garlic", "bay leaves", "pepper"},
        "instructions": "Marinate chicken in soy sauce and vinegar, cook with garlic, bay leaves, and pepper."
    },
    "Sinigang": {
        "ingredients": {"pork", "tamarind", "tomato", "radish", "eggplant", "water spinach"},
        "instructions": "Boil pork with tamarind broth, add vegetables and simmer."
    },
    "Nilaga": {
        "ingredients": {"beef", "potato", "corn", "cabbage", "peppercorn"},
        "instructions": "Boil beef with vegetables and peppercorn until tender."
    },
    "Sisig": {
        "ingredients": {"pork", "onion", "chili", "calamansi", "mayonnaise"},
        "instructions": "Grill pork, chop finely, mix with onion, chili, calamansi, and mayonnaise."
    },
    "Spaghetti": {
        "ingredients": {"spaghetti noodles", "tomato sauce", "ground beef", "hotdog", "cheese"},
        "instructions": "Cook noodles, prepare sauce with ground beef and hotdog, mix and top with cheese."
    },
    "Lumpia": {
        "ingredients": {"spring roll wrappers", "ground pork", "carrot", "onion", "garlic"},
        "instructions": "Mix ground pork with vegetables, wrap in wrappers, and fry."
    },
    "Bistek": {
        "ingredients": {"beef", "soy sauce", "onion", "calamansi", "pepper"},
        "instructions": "Marinate beef in soy sauce and calamansi, cook with onions and pepper."
    },
    "Lechon": {
        "ingredients": {"whole pig", "salt", "pepper", "garlic", "lemongrass"},
        "instructions": "Season pig, stuff with lemongrass, and roast until crispy."
    },
    "Lugaw": {
        "ingredients": {"rice", "chicken broth", "ginger", "garlic", "onion"},
        "instructions": "Cook rice in broth with ginger, garlic, and onion until porridge consistency."
    },
    "Carbonara": {
        "ingredients": {"pasta", "cream", "bacon", "egg yolk", "cheese"},
        "instructions": "Cook pasta, mix with cream, bacon, egg yolk, and cheese."
    },
    "Biko": {
        "ingredients": {"glutinous rice", "coconut milk", "brown sugar"},
        "instructions": "Cook rice with coconut milk and brown sugar until sticky."
    }
}

def suggest_recipes(detected):
    suggestions = []
    for name, recipe in RECIPES.items():
        if recipe["ingredients"].issubset(detected):
            suggestions.append({"name": name, "instructions": recipe["instructions"]})
    return suggestions

st.title("Ingredient Detector and Recipe Suggester")

uploaded_file = st.file_uploader("Upload an image of your ingredients", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run detection
    detected_veg = detect_ingredients(image, veg_model, veg_class_names)
    detected_meat = detect_ingredients(image, meat_model, meat_class_names)
    detected = detected_veg | detected_meat

    st.write("Detected Ingredients:", ", ".join(detected) if detected else "None")

    # Suggest recipes
    suggestions = suggest_recipes(detected)
    if suggestions:
        st.subheader("Suggested Recipes")
        for recipe in suggestions:
            st.markdown(f"**{recipe['name']}**")
            st.write(recipe["instructions"])
    else:
        st.write("No matching recipes found.")

st.info("Note: You must provide your trained YOLOv11 model weights and update the inference code for this app to work.")
