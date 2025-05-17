import uvicorn
from fastapi import FastAPI, File, UploadFile
from typing import Set
from PIL import Image
import io

# Placeholder for YOLOv11 inference (replace with actual model code)
def detect_ingredients_yolov11(image: Image.Image, model_path: str, class_names: list) -> Set[str]:
    # TODO: Load YOLOv11 model and run inference
    # Return a set of detected ingredient names
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

def suggest_recipes(detected: Set[str]):
    suggestions = []
    for name, recipe in RECIPES.items():
        if recipe["ingredients"].issubset(detected):
            suggestions.append({"name": name, "instructions": recipe["instructions"]})
    return suggestions

app = FastAPI()

@app.post("/detect-and-suggest/")
async def detect_and_suggest(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    # Replace with actual model paths and class names
    veg_model_path = "Kleeos-Project-1/yolov11.pt"
    meat_model_path = "Meat-Detection-2/yolov11.pt"
    veg_class_names = []  # Fill with actual class names
    meat_class_names = [] # Fill with actual class names
    ingredients = set()
    ingredients |= detect_ingredients_yolov11(image, veg_model_path, veg_class_names)
    ingredients |= detect_ingredients_yolov11(image, meat_model_path, meat_class_names)
    recipes = suggest_recipes(ingredients)
    return {"detected_ingredients": list(ingredients), "suggested_recipes": recipes}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
