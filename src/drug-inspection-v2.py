import cv2
import numpy as np
from tensorflow.keras.models import load_model
from transformers import AutoTokenizer, AutoModelForCausalLM  #Generative AI
from rdkit import Chem  #SMILES validation
from rdkit.Chem import Descriptors  #Molecular Property Checks
from tkinter import Tk, filedialog

# Load pre-trained model for drug authenticity prediction
loaded_model = load_model('D:\\drug-inspection-project\\trained_model.h5')

# Load Generative AI Model for SMILES-based drug discovery
tokenizer = AutoTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
gen_model = AutoModelForCausalLM.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k", is_decoder=True)

# Function to load an image using a file dialog
def load_image():
    Tk().withdraw()  # Close the root window
    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    if not file_path:
        raise FileNotFoundError("No image selected.")
    return file_path

# Load and preprocess the drug image for prediction
img_path = load_image()
img = cv2.imread(img_path)

# Check if the image was successfully loaded
if img is None:
    raise FileNotFoundError(f"Image not found at {img_path}. Check the file path.")

# Resize and preprocess the image
img = cv2.resize(img, (128, 128))
img = np.expand_dims(img, axis=0) / 255.0  # Normalize the image

# Prediction using the authenticity model
prediction = loaded_model.predict(img)
predicted_class = np.argmax(prediction, axis=1)[0]
if predicted_class == 1:
    print("Authentic Drug")
else:
    print("Counterfeit Drug")

# Generative AI for New Drug Discovery
import re

def generate_new_drug_smiles(seed_smiles="CCO", num_variants=5, max_length=50):
    """
    Generate new SMILES strings for drug discovery based on a seed.
    :param seed_smiles: Starting molecule SMILES string (default: Ethanol)
    :param num_variants: Number of SMILES variants to generate
    :param max_length: Maximum length of the generated SMILES string
    :return: List of validated SMILES strings representing potential drugs
    """
    # Encode the seed SMILES string and generate new molecules
    inputs = tokenizer(seed_smiles, return_tensors="pt")
    
    # Use sampling for diversity and set num_return_sequences
    outputs = gen_model.generate(
        inputs["input_ids"],
        max_length=max_length,      # Reduce max_length to prevent excessive outputs
        num_return_sequences=num_variants,
        do_sample=True,             # Enable sampling for non-deterministic generation
        top_k=50,                   # Top-k sampling (you can adjust this value)
        top_p=0.95                  # Or use top-p nucleus sampling
    )

    # Decode and filter generated SMILES strings
    new_smiles = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    print("Generated SMILES:", new_smiles)

    # SMILES validation function
    def is_valid_smiles(smi):
        # Check for any invalid characters that aren't part of SMILES format
        if re.search(r"[^A-Za-z0-9@+\-\[\]\(\)=#]", smi):
            return False
        # Check with RDKit for further validation
        try:
            mol = Chem.MolFromSmiles(smi)
            return mol is not None
        except:
            return False

    valid_smiles = []
    for smi in new_smiles:
        if is_valid_smiles(smi):
            try:
                mol = Chem.MolFromSmiles(smi)
                mol_weight = Descriptors.MolWt(mol)
                valid_smiles.append((smi, mol_weight))
                print(f"Valid SMILES: {smi}, Mol Weight: {mol_weight}")
            except Exception as e:
                print(f"Invalid SMILES: {smi}, Error: {e}")

    return valid_smiles

# Call the function
generated_drugs = generate_new_drug_smiles(seed_smiles="CCO")
print("Potential new drugs:", generated_drugs)
