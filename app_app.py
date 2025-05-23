import streamlit as st
import pandas as pd
import numpy as np
import re
import ast
from collections import Counter, defaultdict

# Azure SDKs
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from openai import AzureOpenAI # Use AzureOpenAI

# Scikit-learn
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier

# --- Configuration and Setup ---

def get_azure_config():
    """Fetches Azure OpenAI credentials from Key Vault or Streamlit secrets."""
    try:
        key_vault_name = st.secrets.get("AZURE_KEY_VAULT_NAME")
        
        if key_vault_name:
            KVUri = f"https://{key_vault_name}.vault.azure.net"
            credential = DefaultAzureCredential()
            client = SecretClient(vault_url=KVUri, credential=credential)
            azure_openai_key = client.get_secret("AzureOpenAIKey").value
            azure_openai_endpoint = client.get_secret("AzureOpenAIEndpoint").value
            azure_openai_deployment_name = client.get_secret("AzureOpenAIDeploymentName").value
        else:
            azure_openai_key = st.secrets["AZURE_OPENAI_KEY"]
            azure_openai_endpoint = st.secrets["AZURE_OPENAI_ENDPOINT"]
            azure_openai_deployment_name = st.secrets["AZURE_OPENAI_DEPLOYMENT_NAME"]
        
        st.session_state.secrets_loaded = True
        return azure_openai_key, azure_openai_endpoint, azure_openai_deployment_name

    except Exception as e:
        st.error(f"Could not load Azure credentials: {e}")
        st.info(
            "Important: Please ensure your Azure OpenAI credentials (KEY, ENDPOINT, DEPLOYMENT_NAME) "
            "are correctly set in Streamlit secrets or environment variables. "
            "If using Key Vault, check its configuration and access permissions."
        )
        st.session_state.secrets_loaded = False
        return None, None, None

if 'app_initialized' not in st.session_state:
    AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT_NAME = get_azure_config()
    if st.session_state.get('secrets_loaded', False):
        st.session_state.aoai_client = AzureOpenAI(
            api_key=AZURE_OPENAI_KEY,
            api_version="2023-12-01-preview", 
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )
        st.session_state.aoai_deployment_name = AZURE_OPENAI_DEPLOYMENT_NAME
    else:
        st.session_state.aoai_client = None
        st.session_state.aoai_deployment_name = None
    st.session_state.app_initialized = True

symptom_explanations = {
    "itching": "(skin irritation that makes you want to scratch)", "skin_rash": "(red, bumpy, scaly, or itchy skin)",
    "nodal_skin_eruptions": "(bumps or sores on the skin, often near lymph nodes)", "continuous_sneezing": "(sneezing many times in a row or very often)",
    "shivering": "(shaking slightly and uncontrollably, often from cold or fear)", "chills": "(feeling cold and shivery, sometimes with a fever)",
    "joint_pain": "(soreness or aching in your joints like knees, elbows, or shoulders)", "stomach_pain": "(pain in your belly area)",
    "acidity": "(heartburn or a burning feeling in your chest/stomach)", "ulcers_on_tongue": "(small, painful sores on your tongue)",
    "muscle_wasting": "(muscles getting smaller or weaker)", "vomiting": "(throwing up)",
    "burning_micturition": "(pain or burning feeling when you pee)", "spotting_urination": "(passing only small amounts of urine, or drops, when you feel the urge to pee)",
    "fatigue": "(feeling very tired or exhausted)", "weight_gain": "(gaining body weight)",
    "anxiety": "(feeling very worried, nervous, or uneasy)", "cold_hands_and_feets": "(hands and feet feeling unusually cold)",
    "mood_swings": "(quick changes in your mood or emotions)", "weight_loss": "(losing body weight without trying)",
    "restlessness": "(feeling unable to relax or stay still)", "lethargy": "(lack of energy and enthusiasm; feeling sluggish)",
    "patches_in_throat": "(white or red spots or patches in your throat)", "irregular_sugar_level": "(blood sugar levels that go too high or too low)",
    "cough": "(forcing air out of your lungs with a sudden, sharp sound)", "high_fever": "(body temperature much higher than normal)",
    "sunken_eyes": "(eyes appearing deep-set or hollow)", "breathlessness": "(difficulty breathing or feeling like you can't get enough air)",
    "sweating": "(perspiring more than usual)", "dehydration": "(not having enough water in your body)",
    "indigestion": "(discomfort in your stomach, like bloating or pain after eating)", "headache": "(pain in your head)",
    "yellowish_skin": "(skin turning yellow, which could be jaundice)", "dark_urine": "(urine that is darker than usual, like brown or deep yellow)",
    "nausea": "(feeling like you might throw up)", "loss_of_appetite": "(not feeling hungry or wanting to eat)",
    "pain_behind_the_eyes": "(aching or pain felt behind your eyeballs)", "back_pain": "(pain in your back)",
    "constipation": "(difficulty having a bowel movement or pooping)", "abdominal_pain": "(pain in your belly area)",
    "diarrhoea": "(loose, watery bowel movements or pooping frequently)", "mild_fever": "(body temperature slightly higher than normal)",
    "yellow_urine": "(urine that is bright or deep yellow)", "yellowing_of_eyes": "(the white part of your eyes turning yellow)",
    "acute_liver_failure": "(sudden and serious problem with your liver)", "fluid_overload": "(too much fluid in your body)",
    "swelling_of_stomach": "(belly area looking larger or puffier)", "swelled_lymph_nodes": "(lymph nodes, often in neck, armpits, or groin, feeling swollen or tender)",
    "malaise": "(a general feeling of discomfort, illness, or unease)", "blurred_and_distorted_vision": "(vision not clear, things look fuzzy or out of shape)",
    "phlegm": "(thick mucus coughed up from your lungs or throat)", "throat_irritation": "(throat feeling scratchy, sore, or dry)",
    "redness_of_eyes": "(eyes looking red or bloodshot)", "sinus_pressure": "(feeling of pressure or fullness in your sinuses, around your nose and forehead)",
    "runny_nose": "(mucus dripping from your nose)", "congestion": "(nose feeling blocked or stuffy)",
    "chest_pain": "(pain or discomfort in your chest)", "weakness_in_limbs": "(arms or legs feeling weak)",
    "fast_heart_rate": "(heart beating faster than usual)", "pain_during_bowel_movements": "(hurting when you poop)",
    "pain_in_anal_region": "(pain around your bottom or anus)", "bloody_stool": "(blood in your poop)",
    "irritation_in_anus": "(itching or discomfort around your anus)", "neck_pain": "(pain in your neck)",
    "dizziness": "(feeling lightheaded, unsteady, or like the room is spinning)", "cramps": "(sudden, tight, and painful muscle contractions)",
    "bruising": "(getting bruises easily or for no clear reason)", "obesity": "(having an excessive amount of body fat)",
    "swollen_legs": "(legs looking larger or puffier)", "swollen_blood_vessels": "(blood vessels appearing larger or bulging, like varicose veins)",
    "puffy_face_and_eyes": "(face and area around eyes looking swollen or puffy)", "enlarged_thyroid": "(thyroid gland in your neck feeling larger than normal)",
    "brittle_nails": "(nails that break or crack easily)", "swollen_extremeties": "(arms or legs looking swollen or puffy)",
    "excessive_hunger": "(feeling hungry much more than usual)", "extra_marital_contacts": "(this symptom is sensitive and refers to sexual history; usually discussed directly with a doctor)",
    "drying_and_tingling_lips": "(lips feeling dry and having a prickly or stinging sensation)", "slurred_speech": "(difficulty speaking clearly, words may sound jumbled)",
    "knee_pain": "(pain in your knee)", "hip_joint_pain": "(pain in your hip joint)",
    "muscle_weakness": "(muscles not as strong as usual)", "stiff_neck": "(difficulty moving your neck, or neck feeling tight)",
    "swelling_joints": "(joints looking larger or puffier)", "movement_stiffness": "(difficulty starting to move, or feeling stiff when moving)",
    "spinning_movements": "(feeling like you or the room is spinning, a type of dizziness called vertigo)", "loss_of_balance": "(feeling unsteady or having trouble keeping your balance)",
    "unsteadiness": "(feeling wobbly or not stable on your feet)", "weakness_of_one_body_side": "(one side of your body feeling weaker than the other)",
    "loss_of_smell": "(not being able to smell things)", "bladder_discomfort": "(feeling of unease, pressure, or pain in your bladder area)",
    "foul_smell_of_urine": "(urine having a bad or unusual odor)", "continuous_feel_of_urine": "(feeling like you constantly need to pee, even if you just went)",
    "passage_of_gases": "(passing gas or farting)", "internal_itching": "(itching sensation felt inside the body, often in the genital or anal area)",
    "toxic_look_(typhos)": "(appearing very ill, pale, and weak - a term doctors might use)", "depression": "(feeling very sad, hopeless, or losing interest in things for a long time)",
    "irritability": "(getting easily annoyed or agitated)", "muscle_pain": "(aching or pain in your muscles)",
    "altered_sensorium": "(confusion, not thinking clearly, or not being fully aware of surroundings)", "red_spots_over_body": "(red spots appearing on your skin)",
    "belly_pain": "(pain in your stomach or abdomen area)", "abnormal_menstruation": "(periods that are unusual in timing, flow, or duration)",
    "dischromic_patches": "(patches of skin that are a different color, lighter or darker)", "watering_from_eyes": "(eyes producing more tears than usual)",
    "increased_appetite": "(feeling hungrier than usual)", "polyuria": "(peeing much more often or in larger amounts than usual)",
    "family_history": "(if others in your family have had similar health issues)", "mucoid_sputum": "(coughing up thick, slimy mucus)",
    "rusty_sputum": "(coughing up mucus that is rust-colored, often indicating blood)", "lack_of_concentration": "(difficulty focusing or paying attention)",
    "visual_disturbances": "(any problems with your eyesight, like blurriness or seeing spots)", "receiving_blood_transfusion": "(having had blood given to you from someone else)",
    "receiving_unsterile_injections": "(getting shots with needles that weren't clean)", "coma": "(a state of deep unconsciousness)",
    "stomach_bleeding": "(bleeding from your stomach, which might show up in vomit or stool)", "distention_of_abdomen": "(belly feeling bloated, swollen, or stretched out)",
    "history_of_alcohol_consumption": "(if you drink alcohol and how much)", "fluid_overload.1": "(too much fluid in the body, similar to 'fluid_overload')",
    "blood_in_sputum": "(coughing up blood or bloody mucus)", "prominent_veins_on_calf": "(veins on your lower leg looking large and sticking out)",
    "palpitations": "(feeling your heart beat very fast, hard, or irregularly)", "painful_walking": "(hurting when you walk)",
    "pus_filled_pimples": "(pimples that have pus in them)", "blackheads": "(small, dark bumps on the skin, often on the face)",
    "scurring": "(scarring or marks left on the skin after sores heal)", "skin_peeling": "(skin flaking or coming off in layers)",
    "silver_like_dusting": "(fine, silvery scales on the skin, often seen in psoriasis)", "small_dents_in_nails": "(tiny pits or depressions on your fingernails or toenails)",
    "inflammatory_nails": "(nails that are red, swollen, or painful around the edges)", "blister": "(a small bubble on the skin filled with clear fluid)",
    "red_sore_around_nose": "(a red, painful spot or sore near your nose)", "yellow_crust_ooze": "(sores that weep a yellowish fluid which then dries into a crust)"
}
symptom_explanations = {k.lower().replace(' ', '_'): v for k, v in symptom_explanations.items()}

@st.cache_resource
def load_data_and_train_model():
    try:
        df = pd.read_csv('Training.csv')
    except FileNotFoundError:
        st.error("Oops! We couldn't find the necessary data file (Training.csv). Please make sure it's available.")
        return None, None, None, None, None, None, None, None
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    binary_cols = [col for col in numerical_cols if df[col].nunique() == 2 and col != 'prognosis']
    cols_to_keep = binary_cols + ['prognosis']
    df_processed = df[cols_to_keep].copy()
    for col in binary_cols:
        if df_processed[col].isnull().any():
            df_processed[col] = df_processed[col].fillna(0)
    if 'prognosis' not in df_processed.columns:
        st.error("The 'prognosis' column is missing. We can't proceed.")
        return None, None, None, None, None, None, None, None
    df_processed.dropna(subset=['prognosis'], inplace=True)
    if df_processed.empty or len(df_processed['prognosis'].unique()) < 2:
        st.error("The dataset is empty or lacks variety. We need more data.")
        return None, None, None, None, None, None, None, None
    y = df_processed['prognosis']
    lb = LabelBinarizer()
    Y = lb.fit_transform(y)
    disease_names = lb.classes_
    X = df_processed[binary_cols]
    if X.empty or len(Y) == 0:
        st.error("Issue preparing symptom data (X) or outcomes (Y).")
        return None, None, None, None, None, None, None, None
    base_rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    ovr = OneVsRestClassifier(base_rf, n_jobs=-1)
    ovr.fit(X, Y)
    importances = pd.DataFrame(
        {disease: clf.feature_importances_ for disease, clf in zip(disease_names, ovr.estimators_)},
        index=binary_cols
    ).T
    thresh = importances.quantile(0.90, axis=1)
    key_features = {disease: importances.columns[importances.loc[disease] > thresh[disease]]
                    for disease in disease_names}
    disease_to_feats = {disease: list(feats) for disease, feats in key_features.items()}
    symptom_to_diseases = defaultdict(list)
    for disease, feats in disease_to_feats.items():
        for feat in feats:
            symptom_to_diseases[feat].append(disease)
    return (df_processed, binary_cols, lb, ovr, disease_names, 
            disease_to_feats, dict(symptom_to_diseases), X.columns.tolist())

def llm_call_azure(messages, client, deployment_name):
    if not client or not deployment_name:
        st.warning("AI assistant unavailable.")
        return "Error: AI assistant not initialized."
    try:
        completion = client.chat.completions.create(model=deployment_name, messages=messages)
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"Issue with AI assistant: {e}")
        return f"Error in LLM call: {e}"

def extract_initial_symptoms(text, all_known_symptoms):
    pattern = r'\[.*?\]'
    match = re.search(pattern, text)
    extracted_symptoms = []
    if match:
        try:
            potential_symptoms = ast.literal_eval(match.group())
            if isinstance(potential_symptoms, list):
                extracted_symptoms = [s for s in potential_symptoms if s in all_known_symptoms]
        except: pass
    if not extracted_symptoms:
        text_lower = text.lower()
        for symptom in all_known_symptoms:
            symptom_pattern = r'\b' + re.escape(symptom.lower().replace('_', ' ')) + r'\b'
            if re.search(symptom_pattern, text_lower): extracted_symptoms.append(symptom)
            elif '_' not in symptom and re.search(r'\b' + re.escape(symptom.lower()) + r'\b', text_lower):
                 extracted_symptoms.append(symptom)
    return list(set(extracted_symptoms))

def get_initial_diseases(initial_symptoms, symptom_to_diseases_map, top_n=5):
    counts = Counter()
    for symptom in initial_symptoms:
        for disease in symptom_to_diseases_map.get(symptom, []): counts[disease] += 1
    if not counts: return []
    most_common = counts.most_common()
    if not most_common: return []
    relevant_diseases = []
    min_symptom_match = 1
    candidates = [d for d, c in most_common if c >= min_symptom_match]
    if len(candidates) > top_n: relevant_diseases = [d for d, c in most_common[:top_n]]
    else: relevant_diseases = candidates
    relevant_diseases.sort(key=lambda d: (-counts[d], d))
    return relevant_diseases

st.set_page_config(page_title="AI Assistant For Medical Diagnosis", layout="wide")
st.title("🩺 AI Assistant For Medical Diagnosis")
st.caption("I'm here to help you understand your symptoms. Please remember, this is not a medical diagnosis.")

if not st.session_state.get('aoai_client'):
    st.error("AI assistant unavailable. Check setup or try later.")
    st.stop()

with st.spinner("Getting things ready..."):
    load_attempt = load_data_and_train_model()

if load_attempt is None:
    st.error("Problem loading data/model. Cannot continue.")
    st.stop()
else:
    (df_processed, binary_cols, lb, ovr_model, disease_names, 
     disease_to_feats, symptom_to_diseases, original_feature_order) = load_attempt

if 'step' not in st.session_state: st.session_state.step = "initial_input"
if 'initial_symptoms' not in st.session_state: st.session_state.initial_symptoms = []
if 'symptom_dict' not in st.session_state:
    st.session_state.symptom_dict = {col: (df_processed[col].mode().iloc[0] if not df_processed[col].mode().empty else 0) 
                                     for col in binary_cols}
if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'symptoms_to_prompt_chunks' not in st.session_state: st.session_state.symptoms_to_prompt_chunks = []
if 'current_chunk_index' not in st.session_state: st.session_state.current_chunk_index = 0
# To store checkbox states for the current chunk
if 'current_chunk_selections' not in st.session_state: st.session_state.current_chunk_selections = {}


def set_step(step_name): st.session_state.step = step_name

if st.session_state.step == "initial_input":
    with st.container():
        st.header("👋 Welcome! How can I help you today?")
        st.markdown("Please tell me about what you're experiencing. The more details you share, the better I can understand.")
        user_description = st.text_area("E.g., 'I've had a fever and bad cough for two days, and I'm very tired.'", 
                                        height=150, key="user_desc_input")
        if st.button("Analyze My Symptoms", key="analyze_symptoms_btn", type="primary"):
            if not user_description.strip(): st.warning("Please describe your symptoms.")
            else:
                with st.spinner("Thinking about your symptoms..."):
                    prompt_content = f"""You are an AI agent. Given a user's condition, identify symptoms from the list.
                                       Return ONLY a Python-style list of strings. E.g.: ['fever', 'headache']. If none, return [].
                                       User's condition: "{user_description}"
                                       Symptom list: {binary_cols}"""
                    messages = [{"role": "user", "content": prompt_content}]
                    llm_response = llm_call_azure(messages, st.session_state.aoai_client, st.session_state.aoai_deployment_name)
                    if "Error" in llm_response: st.error(f"Sorry, I had trouble. Try rephrasing? ({llm_response})")
                    else:
                        st.session_state.initial_symptoms = extract_initial_symptoms(llm_response, binary_cols)
                        if not st.session_state.initial_symptoms:
                            st.info("Couldn't pinpoint specific symptoms. Don't worry, we'll ask some questions next.")
                        for symptom in st.session_state.initial_symptoms:
                            if symptom in st.session_state.symptom_dict: st.session_state.symptom_dict[symptom] = 1
                        set_step("clarify_symptoms")
                        st.rerun()

if st.session_state.step == "clarify_symptoms":
    with st.container():
        st.header("💬 Let's Clarify Your Symptoms")
        if st.session_state.initial_symptoms:
            st.markdown("Based on your description, I noted:")
            for s_code in st.session_state.initial_symptoms:
                s_display = s_code.replace('_', ' ').capitalize()
                explanation = symptom_explanations.get(s_code.lower(), "")
                st.markdown(f"- *{s_display}* {explanation}")
            st.markdown("Now, a few more questions to get a clearer picture.")
            st.divider()

        if not st.session_state.symptoms_to_prompt_chunks:
            initial_diseases = get_initial_diseases(st.session_state.initial_symptoms, symptom_to_diseases)
            all_symptoms_to_check = set()
            if initial_diseases:
                st.write(f"I'll ask about symptoms related to: {', '.join([d.replace('_', ' ').title() for d in initial_diseases[:3]])}{'...' if len(initial_diseases) > 3 else ''}.")
                for disease in initial_diseases:
                    for feat in disease_to_feats.get(disease, []): all_symptoms_to_check.add(feat)
            else:
                st.info("Since your description was general, I'll ask about some common symptoms.")
                common_general_symptoms = ["fever", "headache", "fatigue", "cough", "nausea", "body_ache", "chills", "sore_throat"] # Reduced to 8 for consistency if this path is taken
                all_symptoms_to_check = set(s for s in common_general_symptoms if s in binary_cols)

            symptoms_to_prompt_further = sorted([s for s in list(all_symptoms_to_check) 
                                                 if s not in st.session_state.initial_symptoms and s in binary_cols])
            
            CHUNK_SIZE = 8 # MODIFIED CHUNK SIZE
            st.session_state.symptoms_to_prompt_chunks = [symptoms_to_prompt_further[i:i+CHUNK_SIZE]
                                                         for i in range(0, len(symptoms_to_prompt_further), CHUNK_SIZE)]
            st.session_state.current_chunk_index = 0
            st.session_state.chat_history = [] 
            st.session_state.current_chunk_selections = {} # Reset for new clarification phase

        for msg_data in st.session_state.chat_history:
            with st.chat_message(msg_data["role"]):
                st.markdown(msg_data["content"], unsafe_allow_html=True if msg_data["role"] == "assistant" else False)

        if st.session_state.current_chunk_index < len(st.session_state.symptoms_to_prompt_chunks):
            current_chunk = st.session_state.symptoms_to_prompt_chunks[st.session_state.current_chunk_index]
            
            if current_chunk:
                # Display question to user only once per chunk
                question_header = "**Are you experiencing any of the following?**<br><small>Please check all that apply, or 'None of these'.</small><br>"
                if not st.session_state.chat_history or not st.session_state.chat_history[-1]["content"].startswith("**Are you experiencing"):
                     st.session_state.chat_history.append({"role": "assistant", "content": question_header})
                     with st.chat_message("assistant"):
                        st.markdown(question_header, unsafe_allow_html=True)
                
                # Create a form for the checkboxes and button
                form_key = f"symptom_form_{st.session_state.current_chunk_index}"
                with st.form(key=form_key):
                    st.session_state.current_chunk_selections = {} # Reset for this chunk display
                    for i, symptom_code in enumerate(current_chunk):
                        symptom_display_name = symptom_code.replace('_', ' ').capitalize()
                        explanation = symptom_explanations.get(symptom_code.lower(), "")
                        checkbox_label = f"**{symptom_display_name}** {explanation}"
                        # Use symptom_code as key for selection state, default to False
                        st.session_state.current_chunk_selections[symptom_code] = st.checkbox(
                            checkbox_label, 
                            key=f"chk_{symptom_code}_{st.session_state.current_chunk_index}"
                        )
                    
                    none_applies = st.checkbox(
                        "**None of these symptoms apply to me**", 
                        key=f"none_chk_{st.session_state.current_chunk_index}"
                    )
                    
                    submitted = st.form_submit_button("Confirm Selections and Continue")

                if submitted:
                    selected_in_form = []
                    if none_applies:
                        for symptom_code in current_chunk:
                            st.session_state.symptom_dict[symptom_code] = 0
                        selected_in_form.append("None of these")
                    else:
                        for symptom_code in current_chunk:
                            # Checkbox state directly from st.session_state.current_chunk_selections
                            if st.session_state.current_chunk_selections.get(symptom_code, False):
                                st.session_state.symptom_dict[symptom_code] = 1
                                selected_in_form.append(symptom_code.replace('_', ' ').capitalize())
                            else: # Ensure it's set to 0 if not selected and not 'none'
                                st.session_state.symptom_dict[symptom_code] = 0
                    
                    user_reply_text = "I selected: " + (", ".join(selected_in_form) if selected_in_form else "Nothing selected (problem here if not 'None')")
                    if not selected_in_form and not none_applies: # Edge case: user submits without selecting anything AND not "none"
                        st.warning("Please make a selection or choose 'None of these'.")
                        # Do not advance, allow user to correct
                    else:
                        st.session_state.chat_history.append({"role": "user", "content": user_reply_text})
                        st.session_state.current_chunk_index += 1
                        st.session_state.current_chunk_selections = {} # Clear selections for next chunk
                        st.rerun()
            else: 
                st.session_state.current_chunk_index += 1
                st.rerun()
        else: 
            st.success("Great, thank you for answering those questions!")
            st.balloons()
            if st.button("See My Symptom Summary", type="primary"):
                set_step("show_results")
                st.rerun()

if st.session_state.step == "show_results":
    with st.container():
        st.header("📝 Here's a Summary Based on Your Symptoms")
        with st.spinner("Analyzing and preparing your summary..."):
            input_data = [st.session_state.symptom_dict.get(s, 0) for s in original_feature_order]
            input_df = pd.DataFrame([input_data], columns=original_feature_order)
            probs = ovr_model.predict_proba(input_df)[0]
            disease_probs = dict(zip(disease_names, probs))
            sorted_diseases_probs = sorted(disease_probs.items(), key=lambda x: -x[1])
            active_s_codes = [s_code for s_code, flag in st.session_state.symptom_dict.items() 
                               if flag == 1 and s_code in binary_cols]
            active_s_display = []
            for s_code in active_s_codes:
                s_display = s_code.replace('_', ' ').capitalize()
                explanation = symptom_explanations.get(s_code.lower(), "")
                active_s_display.append(f"- {s_display} {explanation}")
            
            col1, col2 = st.columns([1,2])
            with col1:
                st.subheader("Your Confirmed Symptoms:")
                if active_s_display:
                    for item in active_s_display: st.markdown(item)
                else: st.markdown("No specific symptoms confirmed from lists.")
            with col2:
                st.subheader("Potential Insights:")
                if not any(p > 0.01 for _, p in sorted_diseases_probs[:5]):
                     st.info("Model didn't strongly point to specific conditions.")
                else:
                    st.markdown("Model suggests these conditions might be worth considering. **Not diagnoses.**")
                    for disease, prob in sorted_diseases_probs[:5]: 
                        if prob > 0.01: st.write(f"- **{disease.replace('_', ' ').title()}**: {prob:.0%} likelihood")
            st.divider()

            if st.session_state.aoai_client:
                st.subheader("💬 A Little More About What This Might Mean:")
                with st.spinner("Asking AI assistant for explanation..."):
                    top_cond_llm = [f"{d.replace('_', ' ').title()} ({p:.0%} like.)" 
                                      for d, p in sorted_diseases_probs[:3] if p > 0.05] 
                    active_s_text_llm = [s_code.replace('_', ' ').capitalize() for s_code in active_s_codes]
                    interpretation_prompt = f"""You are a friendly, empathetic AI health assistant.
                        User symptoms: {', '.join(active_s_text_llm) if active_s_text_llm else "None confirmed"}
                        Potential conditions (NOT diagnoses, just patterns): {', '.join(top_cond_llm) if top_cond_llm else "None strongly indicated"}
                        Please:
                        1. Reassure the user. Explain 'potential conditions' simply.
                        2. If conditions listed, discuss gently. If low likelihood/none, acknowledge.
                        3. General, safe advice (monitor, rest, hydrate).
                        4. Suggest seeing a doctor if severe, worsening, persistent, or worried.
                        5. **End with clear disclaimer**: "Important Reminder: I am an AI assistant, for educational purposes only. Not a medical diagnosis. Always consult a qualified healthcare professional for health concerns."
                        Keep it easy to read, calming."""
                    messages = [{"role": "system", "content": "You are a helpful and empathetic AI assistant."},
                                {"role": "user", "content": interpretation_prompt}]
                    interpretation = llm_call_azure(messages, st.session_state.aoai_client, st.session_state.aoai_deployment_name)
                    if "Error" not in interpretation: st.markdown(interpretation)
                    else: st.warning("Couldn't get detailed AI explanation. See general advice below.")
            else:
                st.markdown("""**Important Reminder:** Not a medical diagnosis. Consult a healthcare professional.
                              Monitor symptoms, rest, hydrate. Seek medical advice if symptoms worsen/persist.""")
        st.divider()
        if st.button("Start Over", key="start_over_btn"):
            keys_to_clear = ['step', 'initial_symptoms', 'symptom_dict', 'chat_history', 
                             'symptoms_to_prompt_chunks', 'current_chunk_index', 'user_desc_input', 'current_chunk_selections']
            for key in keys_to_clear:
                if key in st.session_state: del st.session_state[key]
            if 'binary_cols' in locals() and 'df_processed' in locals():
                st.session_state.symptom_dict = {col: (df_processed[col].mode().iloc[0] if not df_processed[col].mode().empty else 0) 
                                                 for col in binary_cols}
            st.rerun()

st.markdown("---")
st.markdown("<small>Disclaimer: AI Symptom Guide for informational purposes only. Not medical advice. Consult a physician for health concerns.</small>", unsafe_allow_html=True)