import streamlit as st
import dspy
import zipfile
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
# 1. Helper Functions (Image Processing)
# ==========================================

def parse_zip_file(uploaded_file):
    image_pairs = {}
    with zipfile.ZipFile(uploaded_file, 'r') as z:
        for filename in z.namelist():
            if filename.endswith('/') or '__MACOSX' in filename or os.path.basename(filename).startswith('._'):
                continue
            if not filename.lower().endswith('.png'):
                continue
            match = re.search(r'(\d+)_([ab])\.png', filename.lower())
            if match:
                idx = int(match.group(1))
                variant = match.group(2)
                if idx not in image_pairs: image_pairs[idx] = {'a': None, 'b': None}
                try:
                    with z.open(filename) as f:
                        img_bytes = f.read()
                        try:
                            Image.open(io.BytesIO(img_bytes)).verify()
                            image_pairs[idx][variant] = img_bytes
                        except Exception: continue
                except Exception: continue

    valid_pairs = []
    for idx in sorted(image_pairs.keys()):
        pair = image_pairs[idx]
        if pair['a'] is not None and pair['b'] is not None:
            valid_pairs.append({'id': idx, 'a': pair['a'], 'b': pair['b']})
    return valid_pairs

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

# --- 2.1 The Signature ---
class VisualPreference(dspy.Signature):
    """
    You are a UI Design Expert.
    Evaluate the two designs (A and B) based on design principles inferred from previous choices.
    Determine the winner and explain your reasoning.
    """
    image_a: dspy.Image = dspy.InputField(desc="The first UI design variant.")
    image_b: dspy.Image = dspy.InputField(desc="The second UI design variant.")
    winner: str = dspy.OutputField(desc="The superior design. Must be exactly 'A' or 'B'.")

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

# --- 2.4 The "Rubric Extractor" (Strict 7 Items + Likert) ---
class RubricWriter(dspy.Signature):
    """
    You are a Design Lead of a mobile UI design.
    1. First, analyze the provided comparison to understand why the winner was chosen.
    2. Then, abstract this into a "Design Quality Checklist" (System Prompt).
    
    FORMATTING RULES:
    - The checklist must contain EXACTLY 7 distinct criteria.
    - For each criterion, assign a Significance Score (1-5) indicating how heavily this factor influenced the decision.
    - Format each line as: "1. [Criterion Title] (Significance: [1-5]/5): [General Rule]"
    - The output must be specific rules, yet it should not any reference any text or too specific details of the instance.
    - Avoid rules that are trivial or obvious; make them extremely specific, serendipitous, intriguing, and genuinely unexpected.
    - You are encouraged to mention or reference mobile UI design element to describe the rules.
    - Do NOT mention "Image A" or "Image B".
    """
    image_a: dspy.Image = dspy.InputField(desc="Design A")
    image_b: dspy.Image = dspy.InputField(desc="Design B")
    winner: str = dspy.InputField(desc="The user's choice (A or B)")
    
    analysis: str = dspy.OutputField(desc="Specific reasoning (A vs B comparison).")
    clean_rubric: str = dspy.OutputField(desc="The 7-item checklist with significance scores.")

def force_rubric_extraction(trainset, original_program):
    ex = trainset[0]
    writer = dspy.Predict(RubricWriter)
    pred = writer(
        image_a=ex.image_a, 
        image_b=ex.image_b, 
        winner=f"Image {ex.winner}"
    )
    
    new_rubric = pred.clean_rubric
    
    # Save to property for UI and Inference
    original_program.learned_rubric = new_rubric
    
    return original_program, new_rubric

# --- 2.5 Main Optimization Loop ---
def run_optimization(api_key, annotations, image_pairs_lookup):
    lm = dspy.LM(model="openai/gpt-4o-mini", api_key=api_key, api_base="https://openrouter.ai/api/v1", temperature=0.7)
    dspy.configure(lm=lm)

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
    
    # 1. Try COPRO first
    st.info("Attempting to learn rubric via COPRO...")
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
        st.warning(f"COPRO failed ({e}). Falling back to manual extraction.")
        optimized_program = PreferenceModule()

    # 2. Get Instructions
    current_instr = ""
    try:
        current_instr = optimized_program.predict.extended_signature.instructions
    except:
        try:
            current_instr = optimized_program.predict.signature.instructions
        except: pass

    is_generic = "inferred from previous choices" in current_instr or "1. (Rule" in current_instr or not current_instr
    
    # 3. If generic, Force Extraction
    if is_generic:
        st.warning("⚠️ Optimization yielded a generic prompt. Forcing explicit Rubric Extraction (Visual Mode)...")
        optimized_program, new_rubric = force_rubric_extraction(trainset, optimized_program)
        st.success("✅ Rubric Extracted and Applied!")
    else:
        optimized_program.learned_rubric = current_instr

    return optimized_program, trainset

# ==========================================
# 3. Streamlit Application
# ==========================================

st.set_page_config(page_title="Visual Rubric Optimizer", layout="wide")
st.title("🖼️ Visual Rubric Optimization (Strict Inference)")

with st.sidebar:
    st.header("Settings")
    
    # Try to fetch the key from the .env file
    env_key = os.getenv("OPENROUTER_API_KEY", "")
    
    # Pre-fill the input box with the env key if it exists
    openrouter_key = st.text_input("OpenRouter API Key", value=env_key, type="password")

if 'image_pairs' not in st.session_state: st.session_state.image_pairs = []
if 'annotations' not in st.session_state: st.session_state.annotations = []
if 'optimized_prog' not in st.session_state: st.session_state.optimized_prog = None
if 'current_index' not in st.session_state: st.session_state.current_index = 0

tab1, tab2, tab3, tab4 = st.tabs(["1. Upload", "2. Annotate", "3. Learn Rubric", "4. Inference"])

with tab1:
    uploaded_zip = st.file_uploader("Upload ZIP", type="zip")
    if uploaded_zip and st.button("Process ZIP"):
        pairs = parse_zip_file(uploaded_zip)
        if pairs:
            st.session_state.image_pairs = pairs
            st.session_state.current_index = 0
            st.session_state.annotations = []
            st.session_state.optimized_prog = None
            st.success(f"Loaded {len(pairs)} pairs!")

with tab2:
    st.header("Annotate Preferences")
    pairs = st.session_state.image_pairs
    idx = st.session_state.current_index
    if not pairs: st.warning("Upload images first.")
    elif idx < len(pairs):
        current_pair = pairs[idx]
        col1, col2 = st.columns(2)
        with col1: display_limited_image(current_pair['a'], "A")
        with col2: display_limited_image(current_pair['b'], "B")
        with st.form(f"form_{idx}"):
            choice = st.radio("Better Design?", ["A", "B"], horizontal=True)
            if st.form_submit_button("Next"):
                st.session_state.annotations.append({"pair_id": current_pair['id'], "preference": choice})
                st.session_state.current_index += 1
                st.rerun()
    else:
        st.success("Done!")
        if st.button("Reset"):
            st.session_state.current_index = 0
            st.session_state.annotations = []
            st.rerun()

with tab3:
    st.header("Learn Rubric")
    if st.button("Generate Rubric"):
        if not openrouter_key or len(st.session_state.annotations) < 2:
            st.error("Need Key & Data (2+).")
        else:
            img_lookup = {p['id']: p for p in st.session_state.image_pairs}
            with st.spinner("Analyzing..."):
                prog, _ = run_optimization(openrouter_key, st.session_state.annotations, img_lookup)
                st.session_state.optimized_prog = prog
                
                if prog:
                    st.divider()
                    st.subheader("📝 Learned Rubric")
                    rubric_text = getattr(prog, 'learned_rubric', "No rubric text found.")
                    st.info(rubric_text)

with tab4:
    st.header("Test Rubric (Strict)")
    if st.session_state.optimized_prog and st.button("Run Inference"):
        results = []
        img_lookup = {p['id']: p for p in st.session_state.image_pairs}
        prog = st.session_state.optimized_prog
        
        # --- CRITICAL FIX: FORCE RUBRIC INJECTION ---
        rubric_to_use = getattr(prog, 'learned_rubric', None)
        if rubric_to_use:
            st.caption(f"ℹ️ Forcing inference using the learned rubric...")
            # We explicitly overwrite the instructions on the predictor before running
            try:
                if hasattr(prog.predict, 'extended_signature'):
                    prog.predict.extended_signature.instructions = rubric_to_use
                elif hasattr(prog.predict, 'signature'):
                    prog.predict.signature.instructions = rubric_to_use
            except Exception as e:
                st.warning(f"Could not inject rubric: {e}")
        # --------------------------------------------
        
        bar = st.progress(0)
        for i, note in enumerate(st.session_state.annotations):
            pair = img_lookup.get(note['pair_id'])
            
            # Run inference (now guaranteed to use the text rubric)
            pred = prog(image_a=create_dspy_image_object(pair['a']), image_b=create_dspy_image_object(pair['b']))
            
            ai_choice = pred.winner.strip().upper().replace("IMAGE", "").replace(" ", "")[0]
            match = "✅" if ai_choice == note['preference'] else "❌"
            results.append({"Human": note['preference'], "AI": ai_choice, "Match": match, "Reasoning": pred.reasoning})
            bar.progress((i+1)/len(st.session_state.annotations))
        
        st.dataframe(pd.DataFrame(results))