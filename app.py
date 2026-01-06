import streamlit as st
import dspy
import io
import json
import re
import os
from dotenv import load_dotenv
import base64
import pandas as pd
from PIL import Image
from dspy.teleprompt import COPRO

load_dotenv()

# ==========================================
# 1. Helper Functions (Image Processing & Data Loading)
# ==========================================

def load_local_data(n_limit):
    """
    Reads preference.json and loads images from ./images folder.
    Returns formatted pairs and annotations automatically.
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
            
            Image.open(io.BytesIO(img_a_bytes)).verify()
            Image.open(io.BytesIO(img_b_bytes)).verify()

            valid_pairs.append({
                'id': pair_id,
                'a': img_a_bytes,
                'b': img_b_bytes,
                'filename_a': file_a,
                'filename_b': file_b
            })

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
# 2. DSPy Logic
# ==========================================

# --- 2.1 The Inference Signature (Base) ---
class VisualPreference(dspy.Signature):
    """
    This is a placeholder signature. 
    The actual instructions will be injected dynamically based on the User's Taste Profile.
    """
    image_a: dspy.Image = dspy.InputField(desc="The first mobile UI design variant.")
    image_b: dspy.Image = dspy.InputField(desc="The second mobile UI design variant.")
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
    pred_text = prediction.winner.upper()
    match = re.search(r'\b(A|B)\b', pred_text)
    if match:
        found = match.group(1)
        return found == example.winner.strip().upper()
    return False

# --- 2.4 The "Taste Decoder" (Strict 7 Items + Likert 7) ---
class RubricWriter(dspy.Signature):
    """
    You are a decoder of personal preference on mobile UI design. Your job is to reverse-engineer the user's *idiosyncratic* aesthetic preferences of the mobile UI design.
    
    IMPORTANT: 
    - The user's choice might NOT follow standard mobile UI best practices.
    - Do NOT write generic rules like "Good readability" or "Clean layout".
    - Instead, identify the specific "Vibe" or "Trait" that the winner has and the loser lacks.
    
    FORMATTING RULES:
    - Extract EXACTLY 10 distinct 'Preference Criteria'.
    - For each criterion, assign an **Importance Weight (1-7)** (1=Slight Preference, 7=Non-negotiable Requirement).
    - Format: "1. [Specific Mobile UI Design Preference Trait] (Weight: [1-7]): [Description of the preference]"
    - Do NOT mention "Image A" or "Image B". Make them universal rules of this user's preference.
    - When writing rubrics, focus on mobile UI elements and color.
    """
    image_a: dspy.Image = dspy.InputField(desc="Design A")
    image_b: dspy.Image = dspy.InputField(desc="Design B")
    winner: str = dspy.InputField(desc="The user's choice (A or B)")
    
    analysis: str = dspy.OutputField(desc="Analysis of the specific stylistic difference.")
    clean_rubric: str = dspy.OutputField(desc="The 10-item Design Preference Profile with Weights (1-7).")

def generate_readable_rubric(trainset, original_program):
    ex = trainset[0]
    writer = dspy.Predict(RubricWriter)
    pred = writer(
        image_a=ex.image_a, 
        image_b=ex.image_b, 
        winner=f"Image {ex.winner}"
    )
    
    raw_rubric = pred.clean_rubric
    
    formatted_instruction = f"""
    You are simulating a specific user's mobile UI design preference and returning the output that user would prefer. Use their 'Preference Profile' below.

    ### USER PREFERENCE PROFILE (WEIGHTED):
    {raw_rubric}

    ### SCORING PROTOCOL:
    Step 1: For every item in the profile, rate A and B from 1 to 7 based on how much they satisfy this specific rubric.
    Step 2: Multiply the rating by the (Weight).
    Step 3: Show the calculation: "Item X: A=(Score*Weight)=Total, B=(Score*Weight)=Total"
    Step 4: Sum all totals.
    Step 5: Output the Winner (A or B) based on the highest sum.
    """
    
    original_program.learned_rubric = formatted_instruction
    return original_program

# --- 2.5 Main Optimization Loop ---
def run_optimization(api_key, annotations, image_pairs_lookup):
    
    # --- MODEL CONFIGURATION ---
    model_id = "openrouter/google/gemini-2.5-flash"
    
    lm = dspy.LM(
        model=model_id, 
        api_key=api_key, 
        api_base="https://openrouter.ai/api/v1", 
        temperature=0.7,
        cache=False 
    )
    
    # FIX: Use Context Manager instead of global configure to avoid Threading Error
    # dspy.configure(lm=lm) <-- REMOVED

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
    
    st.info("Phase 1: Analyzing preferences...")
    
    # FIX: Wrap execution in context
    with dspy.context(lm=lm):
        try:
            teleprompter = COPRO(
                metric=choice_match_metric,
                breadth=5, depth=2, init_temperature=1.0, prompt_model=lm
            )
            optimized_program = teleprompter.compile(
                PreferenceModule(),
                trainset=trainset,
                eval_kwargs={"num_threads": 4, "display_progress": True}
            )
        except Exception as e:
            print(f"Fallback due to error: {e}")
            optimized_program = PreferenceModule()

        st.info("Phase 2: Decoding User Taste Profile (Likert 1-7)...")
        # generate_readable_rubric calls dspy.Predict, which needs the context
        optimized_program = generate_readable_rubric(trainset, optimized_program)
        
    st.success("✅ Taste Profile Decoded!")

    return optimized_program, trainset

# ==========================================
# 3. Streamlit Application
# ==========================================

st.set_page_config(page_title="Visual Rubric Optimizer", layout="wide")
st.title("🖼️ Visual Rubric: Personal Taste Decoder")

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

tab1, tab2, tab3 = st.tabs(["1. Data Preview", "2. Learn Taste Profile", "3. Run Scoring Inference"])

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
    st.header("Learn Taste Profile")
    if st.button("Generate Profile"):
        if not openrouter_key or not st.session_state.data_loaded:
            st.error("Missing Key or Data.")
        else:
            img_lookup = {p['id']: p for p in st.session_state.image_pairs}
            with st.spinner("Decoding aesthetic biases..."):
                prog, _ = run_optimization(openrouter_key, st.session_state.annotations, img_lookup)
                st.session_state.optimized_prog = prog
                
                if prog:
                    st.divider()
                    st.subheader("📝 Learned Taste Profile")
                    rubric_text = getattr(prog, 'learned_rubric', "")
                    display_text = rubric_text.split("### SCORING PROTOCOL")[0] if "###" in rubric_text else rubric_text
                    st.info(display_text)

with tab3:
    st.header("Weighted Inference Scoring")
    if st.session_state.optimized_prog and st.button("Calculate Scores"):
        results = []
        img_lookup = {p['id']: p for p in st.session_state.image_pairs}
        prog = st.session_state.optimized_prog
        rubric_to_use = getattr(prog, 'learned_rubric', None)
        
        if not rubric_to_use:
            st.error("No rubric found. Please run Tab 2 first.")
        else:
            # FIX: Re-initialize LM locally for this thread
            lm_inference = dspy.LM(
                model="openrouter/google/gemini-2.5-flash", 
                api_key=openrouter_key, 
                api_base="https://openrouter.ai/api/v1", 
                temperature=0.7,
                cache=False 
            )

            # Dynamic Class to bypass Cache and fix AttributeError
            class DynamicPreference(dspy.Signature):
                __doc__ = rubric_to_use
                image_a: dspy.Image = dspy.InputField(desc="The first UI design variant.")
                image_b: dspy.Image = dspy.InputField(desc="The second UI design variant.")
                winner: str = dspy.OutputField(desc="The design that matches the user's taste. Must be exactly 'A' or 'B'.")

            fresh_scorer = dspy.ChainOfThought(DynamicPreference)

            correct_count = 0
            total_count = 0
            bar = st.progress(0)
            
            # FIX: Wrap loop in context
            with dspy.context(lm=lm_inference):
                for i, note in enumerate(st.session_state.annotations):
                    pair = img_lookup.get(note['pair_id'])
                    
                    try:
                        pred = fresh_scorer(
                            image_a=create_dspy_image_object(pair['a']), 
                            image_b=create_dspy_image_object(pair['b'])
                        )
                        
                        raw_winner = pred.winner.strip().upper()
                        # Simple cleanup
                        if "A" in raw_winner and "B" not in raw_winner: ai_choice = "A"
                        elif "B" in raw_winner and "A" not in raw_winner: ai_choice = "B"
                        else: ai_choice = raw_winner[0] if raw_winner else "?"
                        
                        # Logic for Hit Rate
                        if ai_choice == note['preference']:
                            match = "✅"
                            correct_count += 1
                        else:
                            match = "❌"
                        total_count += 1
                            
                        reasoning_snippet = pred.reasoning[:600] + "..." if hasattr(pred, 'reasoning') else "No reasoning"
                        
                        results.append({
                            "Human": note['preference'], 
                            "AI": ai_choice, 
                            "Match": match, 
                            "Detailed Calculation": reasoning_snippet
                        })
                    except Exception as e:
                        st.error(f"Error on item {i}: {e}")
                    
                    bar.progress((i+1)/len(st.session_state.annotations))
            
            # Display Hit Rate
            if total_count > 0:
                accuracy = (correct_count / total_count) * 100
                st.metric("Inference Accuracy (Hit Rate)", f"{accuracy:.1f}%")
            
            st.dataframe(pd.DataFrame(results))