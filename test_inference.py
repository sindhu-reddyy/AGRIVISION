import sys
import os

# Ensure we can import from local modules
sys.path.append(os.getcwd())

try:
    from inference.predict import load_resources, predict_plant_disease
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def test():
    print("Loading resources...")
    try:
        load_resources()
    except Exception as e:
        print(f"Failed to load resources: {e}")
        return

    image_path = "image_000035.JPG"
    if not os.path.exists(image_path):
        print(f"Test image {image_path} not found.")
        return

    question = "What is the disease?"
    print(f"Testing with image: {image_path} and question: '{question}'")

    try:
        result = predict_plant_disease(image_path, question)
        print(f"Raw Result: '{result}'")
        if not result:
            print("Result is empty!")
        else:
            print("Result received successfully.")
    except Exception as e:
        print(f"Prediction failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()
