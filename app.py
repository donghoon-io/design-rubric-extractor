import streamlit as st
import dspy
import io
import json
import os
import base64
import pandas as pd
from PIL import Image
from dotenv import load_dotenv

# IMPORTS FOR MIPROv2
from dspy.teleprompt import MIPROv2

load_dotenv()

# ==========================================
# 1. Helper Functions (Image Processing & Data Loading)
# ==========================================

def load_local_data(n_limit):
    """
    Reads preference.json and loads images from ./images folder.
    """
    json_path = "./preference.json"
    images_dir = "./images"
    
    if not os.path.exists(json_path):
        st.error(f"❌ File not found: {json_path}")
        return [], []
    
    if not os.path.exists(images_dir):
        st.error(f"❌ Directory not found: {images_dir}")
        return [], []

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        st.error(f"Error reading JSON: {e}")
        return [], []

    # Limit to N
    data = data[:n_limit]
    
    valid_pairs = []
    annotations = []

    for item in data:
        file_a = item.get('left_file')
        file_b = item.get('right_file')
        pair_id = item.get('pair_id')
        raw_choice = item.get('final_choice', '') 

        if not file_a or not file_b or not pair_id:
            continue

        path_a = os.path.join(images_dir, file_a)
        path_b = os.path.join(images_dir, file_b)

        if not os.path.exists(path_a) or not os.path.exists(path_b):
            continue

        try:
            with open(path_a, 'rb') as f: img_a_bytes = f.read()
            with open(path_b, 'rb') as f: img_b_bytes = f.read()
            
            # Verify they are valid images
            Image.open(io.BytesIO(img_a_bytes)).verify()
            Image.open(io.BytesIO(img_b_bytes)).verify()

            valid_pairs.append({
                'id': pair_id,
                'a': img_a_bytes,
                'b': img_b_bytes,
                'filename_a': file_a,
                'filename_b': file_b
            })

            # Clean up the choice string
            winner = "A"
            if ">>" in raw_choice:
                winner = raw_choice.split(">>")[0].strip().upper()
            elif raw_choice.strip().upper() in ["A", "B"]:
                winner = raw_choice.strip().upper()
            
            annotations.append({
                "pair_id": pair_id, 
                "preference": winner
            })

        except Exception as e:
            print(f"Skipping {pair_id}: {e}")
            continue

    return valid_pairs, annotations

def display_limited_image(image_bytes, caption, height_vh=50):
    if image_bytes is None: return
    b64_img = base64.b64encode(image_bytes).decode()
    html_code = f"""
    <div style="display: flex; flex-direction: column; align-items: center; margin-bottom: 10px;">
        <img src="data:image/png;base64,{b64_img}" 
             style="max-height: {height_vh}vh; max-width: 100%; object-fit: contain; border-radius: 5px;">
        <div style="color: #666; font-size: 0.9em; margin-top: 5px; font-weight: 500;">{caption}</div>
    </div>
    """
    st.markdown(html_code, unsafe_allow_html=True)

def resize_image_for_api(img_bytes, max_dimension=512):
    try:
        img = Image.open(io.BytesIO(img_bytes))
        img.thumbnail((max_dimension, max_dimension))
        buf = io.BytesIO()
        img.save(buf, format="PNG") 
        return buf.getvalue()
    except Exception: return img_bytes

def create_dspy_image_object(img_bytes):
    resized_bytes = resize_image_for_api(img_bytes)
    b64_str = base64.b64encode(resized_bytes).decode('utf-8')
    data_uri = f"data:image/png;base64,{b64_str}"
    return dspy.Image(data_uri)

# ==========================================
# 2. DSPy Logic (MIPROv2 Optimization)
# ==========================================

# --- 2.1 The UPDATED Inference Signature ---
class VisualPreference(dspy.Signature):
    """
    Decide which design better matches the user's taste.
    Return a structured rubric explaining the decision.
    """
    image_a: dspy.Image = dspy.InputField(desc="The first mobile UI design variant.")
    image_b: dspy.Image = dspy.InputField(desc="The second mobile UI design variant.")
    
    # ⬇️ NEW: Explicitly asking for 10 rules
    rubrics: str = dspy.OutputField(
        desc=(
            "A list of EXACTLY 10 concise visual preference rules "
            "that explain why the winner was chosen. "
            "Each rule should be one sentence, numbered 1–10."
        )
    )
    
    winner: str = dspy.OutputField(desc="The design that matches the user's preference. Must be exactly 'A' or 'B'.")

# --- 2.2 The Module ---
class PreferenceModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(VisualPreference)

    def forward(self, image_a, image_b):
        return self.predict(image_a=image_a, image_b=image_b)

# --- 2.3 The Metric ---
def choice_match_metric(example, prediction, trace=None):
    # We only care if the Winner matches. 
    # The rubrics are the *explanation*, not the score target.
    pred_text = prediction.winner.strip().upper()
    
    if "A" in pred_text and "B" not in pred_text: cleaned_pred = "A"
    elif "B" in pred_text and "A" not in pred_text: cleaned_pred = "B"
    else: cleaned_pred = pred_text[0] if pred_text else ""
        
    return cleaned_pred == example.winner.strip().upper()

# --- 2.4 The MIPROv2 Optimizer Function ---
def run_mipro_optimization(api_key, annotations, image_pairs_lookup):
    
    # 1. SETUP MODEL
    lm = dspy.LM(
        model="openai/gpt-4.1-mini", 
        api_key=api_key, 
        api_base="https://openrouter.ai/api/v1", 
        temperature=1.0, 
        cache=False 
    )
    dspy.configure(lm=lm)

    # 2. PREPARE DATA
    trainset = []
    for note in annotations:
        pair = image_pairs_lookup.get(note['pair_id'])
        if pair:
            trainset.append(dspy.Example(
                image_a=create_dspy_image_object(pair['a']),
                image_b=create_dspy_image_object(pair['b']),
                winner=note['preference']
            ).with_inputs('image_a', 'image_b'))

    if not trainset: return None, "No data."
    
    # 3. CONFIGURE MIPROv2 (OPTIMIZED SETTINGS)
    teleprompter = MIPROv2(
        metric=choice_match_metric,
        prompt_model=lm, 
        task_model=lm,   
        num_candidates=3,      # ⬇️ Reduced from 5 for speed
        init_temperature=0.7,  # ⬇️ Reduced slightly for stability
        verbose=True,
        auto=None
    )
    
    st.info(f"🚀 Starting DSPy MIPROv2 Optimizer with {len(trainset)} examples...")

    try:
        # 4. COMPILE (The Optimization Loop)
        optimized_program = teleprompter.compile(
            PreferenceModule(),
            trainset=trainset,
            num_trials=5,             # ⬇️ Crucial: Reduced from 15 to 5
            max_bootstrapped_demos=0, # ⬇️ Crucial: Disable synthetic demo generation
            max_labeled_demos=1,      # Keep 1 real example
            minibatch_size=3,         # ⬇️ Small batch for speed
            requires_permission_to_run=False
        )
        
        # 5. EXTRACT INSIGHTS (BUG FIX APPLIED HERE)
        # We access the signature of the 'predict' module within the compiled program

        # 5. EXTRACT INSIGHTS (VERSION-COMPATIBLE FIX)
        st.success("✅ Optimization Complete! Taste Profile Found.")

        # ChainOfThought → Predict → Signature
        predict_module = optimized_program.predict.predict
        sig = predict_module.signature

        optimized_program.learned_rubric = sig.instructions

        return optimized_program, trainset

    except Exception as e:
        import traceback
        st.error(f"MIPROv2 Optimization failed: {e}")
        st.code(traceback.format_exc())
        return None, str(e)

# ==========================================
# 3. Streamlit Application
# ==========================================

st.set_page_config(page_title="Visual Rubric Optimizer", layout="wide")
st.title("🖼️ Visual Rubric: MIPROv2 Optimizer")

if 'image_pairs' not in st.session_state: st.session_state.image_pairs = []
if 'annotations' not in st.session_state: st.session_state.annotations = []
if 'optimized_prog' not in st.session_state: st.session_state.optimized_prog = None
if 'data_loaded' not in st.session_state: st.session_state.data_loaded = False

with st.sidebar:
    st.header("Settings")
    env_key = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_key = st.text_input("OpenRouter API Key", value=env_key, type="password")
    
    st.divider()
    st.header("Data Loader")
    n_samples = st.number_input("Number of samples (N) to process:", min_value=1, max_value=100, value=10)
    
    if st.button("Run / Load Data"):
        st.session_state.image_pairs = []
        st.session_state.annotations = []
        st.session_state.optimized_prog = None
        
        with st.spinner(f"Loading first {n_samples} items..."):
            pairs, annotations = load_local_data(n_samples)
            if pairs:
                st.session_state.image_pairs = pairs
                st.session_state.annotations = annotations
                st.session_state.data_loaded = True
                st.success(f"✅ Loaded {len(pairs)} pairs!")
            else:
                st.session_state.data_loaded = False
                st.error("Failed to load data.")

tab1, tab2, tab3 = st.tabs(["1. Data Preview", "2. Learn Taste Profile (MIPROv2)", "3. Verify Accuracy & Rubrics"])

with tab1:
    st.header("Data Preview")
    if st.session_state.data_loaded:
        for i, note in enumerate(st.session_state.annotations):
            pair = next((p for p in st.session_state.image_pairs if p['id'] == note['pair_id']), None)
            if pair:
                with st.expander(f"Pair #{i+1} (Winner: {note['preference']})"):
                    col1, col2 = st.columns(2)
                    with col1: display_limited_image(pair['a'], f"A: {pair.get('filename_a','')}", 30)
                    with col2: display_limited_image(pair['b'], f"B: {pair.get('filename_b','')}", 30)

with tab2:
    st.header("Learn Taste Profile (MIPROv2)")
    st.markdown("""
    **Optimization Settings:**
    * `num_candidates`: 3 (Variations of instruction to propose)
    * `num_trials`: 5 (Iterations to refine)
    * `max_bootstrapped_demos`: 0 (Pure Few-Shot optimization)
    """)
    
    if st.button("Run MIPRO Optimization"):
        if not openrouter_key or not st.session_state.data_loaded:
            st.error("Missing Key or Data.")
        else:
            img_lookup = {p['id']: p for p in st.session_state.image_pairs}
            
            with st.spinner("Running MIPROv2 (Lite Mode)..."):
                prog, _ = run_mipro_optimization(openrouter_key, st.session_state.annotations, img_lookup)
                st.session_state.optimized_prog = prog
                
                if prog:
                    st.divider()
                    st.subheader("🧬 The Optimized Instruction (System Prompt)")
                    st.success("This instruction prompts the model to generate the 10 rubrics below:")
                    rubric_text = getattr(prog, 'learned_rubric', "No rubric found")
                    st.code(rubric_text, language="text")

with tab3:
    st.header("Verify Accuracy & View 10 Rubrics")
    if st.session_state.optimized_prog and st.button("Run Inference"):
        results = []
        img_lookup = {p['id']: p for p in st.session_state.image_pairs}
        prog = st.session_state.optimized_prog
        
        # Configure LM for inference
        lm_inference = dspy.LM(
            model="openai/gpt-4.1-mini", 
            api_key=openrouter_key, 
            api_base="https://openrouter.ai/api/v1", 
            temperature=0.0,
            cache=False 
        )
        dspy.configure(lm=lm_inference)

        correct_count = 0
        total_count = 0
        bar = st.progress(0)
        
        for i, note in enumerate(st.session_state.annotations):
            pair = img_lookup.get(note['pair_id'])
            
            try:
                # Use the optimized program
                pred = prog(
                    image_a=create_dspy_image_object(pair['a']), 
                    image_b=create_dspy_image_object(pair['b'])
                )
                
                raw_winner = pred.winner.strip().upper()
                if "A" in raw_winner and "B" not in raw_winner: ai_choice = "A"
                elif "B" in raw_winner and "A" not in raw_winner: ai_choice = "B"
                else: ai_choice = raw_winner[0] if raw_winner else "?"
                
                match = "✅" if ai_choice == note['preference'] else "❌"
                if match == "✅": correct_count += 1
                total_count += 1
                    
                # Store the Generated Rubrics
                results.append({
                    "Pair ID": note['pair_id'],
                    "Human": note['preference'], 
                    "AI": ai_choice, 
                    "Match": match, 
                    "Generated Rubrics (10 Rules)": pred.rubrics
                })
            except Exception as e:
                st.error(f"Error on item {i}: {e}")
            
            bar.progress((i+1)/len(st.session_state.annotations))
        
        if total_count > 0:
            accuracy = (correct_count / total_count) * 100
            st.metric("Inference Accuracy", f"{accuracy:.1f}%")
        
        # Display DataFrame with Rubrics column
        st.dataframe(pd.DataFrame(results))
        
        # Detail view for first item
        if results:
            st.divider()
            st.subheader("Example Output (Rubrics for first item)")
            st.markdown(results[0]["Generated Rubrics (10 Rules)"])