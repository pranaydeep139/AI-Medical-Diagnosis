import streamlit as st
import pandas as pd
import numpy as np
import re
import ast
import json
import os
import joblib # For loading the .pkl model
from collections import Counter, defaultdict

# Azure SDKs
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from openai import AzureOpenAI
from azure.storage.blob import BlobServiceClient


ARTIFACTS_PATH = "app_artifacts"  # Path to the folder with JSON artifacts and model.pkl

print("🟢 Starting blob artifacts download process...")

# 1. Authenticate via DefaultAzureCredential
print("🔑 Acquiring DefaultAzureCredential...")
credential = DefaultAzureCredential()
print("✅ Credential acquired.")

# 2. Create BlobServiceClient via AAD
account_url = f"https://{st.secrets['AZURE_STORAGE_ACCOUNT_NAME']}.blob.core.windows.net"
print(f"🌐 Connecting to BlobServiceClient at {account_url} using AAD...")
service_client = BlobServiceClient(account_url=account_url, credential=credential)
print("✅ AAD-based BlobServiceClient connected.")

# 3. Fallback via connection string
print("🔄 Overriding with connection string auth (if provided)...")
conn_str = st.secrets.get("AZURE_STORAGE_CONNECTION_STRING", None)
if conn_str:
    service_client = BlobServiceClient.from_connection_string(conn_str)
    print("✅ BlobServiceClient initialized from connection string.")
else:
    print("⚠️ No AZURE_STORAGE_CONNECTION_STRING found; continuing with AAD client.")

# 4. Get ContainerClient
container_name = "plsworkman"
print(f"📦 Getting container client for '{container_name}'...")
container_client = service_client.get_container_client(container_name)
print("✅ Container client ready.")

# 5. Ensure local artifacts folder exists
print(f"🗂️ Ensuring local artifacts path exists at '{ARTIFACTS_PATH}'...")
os.makedirs(ARTIFACTS_PATH, exist_ok=True)
print("✅ Artifacts directory is present.")

# 6. Download each blob
artifact_files = [
    "model.pkl",
    "model_binary_cols.json",
    "model_disease_names.json",
    "model_disease_to_feats.json",
    "model_symptom_to_diseases.json",
    "symptom_defaults.json"
]

for blob_name in artifact_files:
    download_path = os.path.join(ARTIFACTS_PATH, blob_name)
    if os.path.isfile(download_path):  # check if file exists :contentReference[oaicite:2]{index=2}
        print(f"⚡ '{blob_name}' already exists at '{download_path}'; skipping download.")
        continue

    print(f"⬇️  Downloading blob '{blob_name}'...")
    blob_client = container_client.get_blob_client(blob_name)
    try:
        with open(download_path, "wb") as download_file:
            download_stream = blob_client.download_blob()  # :contentReference[oaicite:3]{index=3}
            download_file.write(download_stream.readall())
        print(f"✅ Successfully downloaded '{blob_name}' to '{download_path}'.")
    except Exception as e:
        print(f"❌ Failed to download '{blob_name}': {e}")

print("🎉 All done with blob downloads!")

def get_azure_config():
    """Fetches Azure OpenAI credentials from Key Vault or Streamlit secrets."""
    config = {}
    try:
        key_vault_name = st.secrets.get("AZURE_KEY_VAULT_NAME")
        if key_vault_name:
            KVUri = f"https://{key_vault_name}.vault.azure.net"
            credential = DefaultAzureCredential()
            client = SecretClient(vault_url=KVUri, credential=credential)
            config["azure_openai_key"] = client.get_secret("AzureOpenAIKey").value
            config["azure_openai_endpoint"] = client.get_secret("AzureOpenAIEndpoint").value
            config["azure_openai_deployment_name"] = client.get_secret("AzureOpenAIDeploymentName").value
        else:
            config["azure_openai_key"] = st.secrets["AZURE_OPENAI_KEY"]
            config["azure_openai_endpoint"] = st.secrets["AZURE_OPENAI_ENDPOINT"]
            config["azure_openai_deployment_name"] = st.secrets["AZURE_OPENAI_DEPLOYMENT_NAME"]
        
        st.session_state.secrets_loaded = True
        return config

    except Exception as e:
        st.error(f"Could not load Azure OpenAI credentials: {e}")
        st.info(
            "Important: Please ensure your Azure OpenAI credentials (KEY, ENDPOINT, DEPLOYMENT_NAME) "
            "are correctly set in Streamlit secrets or environment variables. "
            "If using Key Vault, check its configuration and access permissions."
        )
        st.session_state.secrets_loaded = False
        return None

if 'app_initialized' not in st.session_state:
    azure_config = get_azure_config()
    if st.session_state.get('secrets_loaded', False) and azure_config:
        st.session_state.aoai_client = AzureOpenAI(
            api_key=azure_config["azure_openai_key"],
            api_version="2023-12-01-preview", 
            azure_endpoint=azure_config["azure_openai_endpoint"]
        )
        st.session_state.aoai_deployment_name = azure_config["azure_openai_deployment_name"]
    else:
        st.session_state.aoai_client = None
        st.session_state.aoai_deployment_name = None
    st.session_state.app_initialized = True

# --- Paste your full symptom_explanations dictionary here ---
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
if 'fluid_overload.1' in symptom_explanations: # Handle specific case if necessary
    symptom_explanations['fluid_overload_1'] = symptom_explanations.pop('fluid_overload.1')

@st.cache_resource
def load_application_resources():
    """Loads pre-computed artifacts and the model needed for the app."""
    try:
        model_path = os.path.join(ARTIFACTS_PATH, 'model.pkl')
        ml_model = joblib.load(model_path)

        with open(os.path.join(ARTIFACTS_PATH, 'model_binary_cols.json'), 'r') as f:
            binary_cols = json.load(f)
        with open(os.path.join(ARTIFACTS_PATH, 'model_disease_names.json'), 'r') as f:
            disease_names = json.load(f)
        with open(os.path.join(ARTIFACTS_PATH, 'model_disease_to_feats.json'), 'r') as f:
            disease_to_feats = json.load(f)
        with open(os.path.join(ARTIFACTS_PATH, 'model_symptom_to_diseases.json'), 'r') as f:
            symptom_to_diseases_raw = json.load(f)
            symptom_to_diseases = defaultdict(list, symptom_to_diseases_raw)
        with open(os.path.join(ARTIFACTS_PATH, 'symptom_defaults.json'), 'r') as f:
            symptom_defaults = json.load(f)

        original_feature_order = list(binary_cols) 

        for symptom_code in binary_cols:
            if symptom_code.lower() not in symptom_explanations:
                symptom_explanations[symptom_code.lower()] = "(No detailed explanation available)"
        
        # Return model as the first element for clarity
        return (ml_model, binary_cols, disease_names, disease_to_feats, 
                symptom_to_diseases, original_feature_order, symptom_defaults)

    except FileNotFoundError as e:
        st.error(f"Oops! Critical application resource file not found: {e.filename}. "
                 f"Please ensure all artifact files (including model.pkl) from '{ARTIFACTS_PATH}/' are available.")
        return None
    except Exception as e:
        st.error(f"Error loading application resources: {e}")
        return None

def llm_call_azure(messages, client, deployment_name):
    if not client or not deployment_name:
        st.warning("AI assistant (OpenAI) unavailable: Not initialized.")
        return "Error: AI assistant not initialized."
    try:
        completion = client.chat.completions.create(model=deployment_name, messages=messages)
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"Issue with AI assistant (OpenAI): {e}")
        return f"Error in LLM call: {e}"

def extract_initial_symptoms(text, all_known_symptoms):
    pattern = r'\[.*?\]'
    match = re.search(pattern, text)
    extracted_symptoms = []
    if match:
        try:
            potential_symptoms_raw = ast.literal_eval(match.group())
            if isinstance(potential_symptoms_raw, list):
                extracted_symptoms = [
                    str(s).strip().lower().replace(' ', '_') for s in potential_symptoms_raw
                    if str(s).strip().lower().replace(' ', '_') in all_known_symptoms
                ]
        except (ValueError, SyntaxError):
            pass 
    if not extracted_symptoms:
        text_lower = text.lower()
        for symptom in all_known_symptoms:
            symptom_text_form = symptom.replace('_', ' ')
            symptom_pattern = r'\b' + re.escape(symptom_text_form) + r'\b'
            if re.search(symptom_pattern, text_lower):
                extracted_symptoms.append(symptom)
    return list(set(extracted_symptoms))

def get_initial_diseases(initial_symptoms, symptom_to_diseases_map, top_n=5):
    counts = Counter()
    for symptom in initial_symptoms:
        for disease in symptom_to_diseases_map.get(symptom, []):
            counts[disease] += 1
    if not counts: return []
    most_common = counts.most_common()
    if not most_common: return []
    relevant_diseases = []
    min_symptom_match = 1
    candidates = [d for d, c in most_common if c >= min_symptom_match]
    if len(candidates) > top_n:
        relevant_diseases = [d for d, c in most_common[:top_n]]
    else:
        relevant_diseases = candidates
    relevant_diseases.sort(key=lambda d: (-counts[d], d))
    return relevant_diseases

st.set_page_config(page_title="AI Assistant For Medical Diagnosis", layout="wide")
st.title("🩺 AI Assistant For Medical Diagnosis")
st.caption("I'm here to help you understand your symptoms. Please remember, this is not a medical diagnosis.")

resources_loaded = False
if 'app_resources' not in st.session_state:
    with st.spinner("Getting application resources ready..."):
        load_attempt = load_application_resources()
        if load_attempt:
            st.session_state.app_resources = load_attempt
            resources_loaded = True
        else:
            st.error("Fatal: Could not load application resources. The app cannot continue.")
            st.stop()
else:
    resources_loaded = True

if not resources_loaded:
    st.error("Application resources not available.")
    st.stop()

# Unpack resources, model is now the first element
(ml_model, binary_cols, disease_names_from_model, disease_to_feats, 
 symptom_to_diseases_map, original_feature_order, symptom_defaults) = st.session_state.app_resources

if 'step' not in st.session_state: st.session_state.step = "initial_input"
if 'initial_symptoms' not in st.session_state: st.session_state.initial_symptoms = []
if 'symptom_dict' not in st.session_state:
    st.session_state.symptom_dict = symptom_defaults.copy()
if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'symptoms_to_prompt_chunks' not in st.session_state: st.session_state.symptoms_to_prompt_chunks = []
if 'current_chunk_index' not in st.session_state: st.session_state.current_chunk_index = 0
if 'current_chunk_selections' not in st.session_state: st.session_state.current_chunk_selections = {}

def set_step(step_name): st.session_state.step = step_name

# --- UI Flow ---
if st.session_state.step == "initial_input":
    # ... (initial_input step code remains largely the same as previous version) ...
    if not st.session_state.get('aoai_client'):
        st.error("AI assistant (OpenAI) is unavailable. Please check configuration or try again later.")
    
    with st.container():
        st.header("👋 Welcome! How can I help you today?")
        st.markdown("Please tell me about what you're experiencing. The more details you share, the better I can understand.")
        user_description = st.text_area("E.g., 'I've had a fever and bad cough for two days, and I'm very tired.'", 
                                        height=150, key="user_desc_input")
        if st.button("Analyze My Symptoms", key="analyze_symptoms_btn", type="primary"):
            if not user_description.strip(): st.warning("Please describe your symptoms.")
            elif not st.session_state.get('aoai_client'):
                 st.error("Cannot analyze: AI assistant (OpenAI) is unavailable.")
            else:
                with st.spinner("Thinking about your symptoms..."):
                    prompt_content = f"""You are an AI agent. Given a user's condition, identify symptoms from the list.
                                       Return ONLY a Python-style list of strings. E.g.: ['fever', 'headache']. If none, return [].
                                       Ensure symptom names in the list exactly match those in the provided list (case-sensitive, use underscores).
                                       User's condition: "{user_description}"
                                       Symptom list: {original_feature_order}"""
                    messages = [{"role": "user", "content": prompt_content}]
                    llm_response = llm_call_azure(messages, st.session_state.aoai_client, st.session_state.aoai_deployment_name)
                    
                    if "Error" in llm_response: 
                        st.error(f"Sorry, I had trouble understanding the symptoms from your description. Could you try rephrasing? (Details: {llm_response})")
                    else:
                        st.session_state.initial_symptoms = extract_initial_symptoms(llm_response, original_feature_order)
                        if not st.session_state.initial_symptoms:
                            st.info("I couldn't pinpoint specific symptoms from your description. Don't worry, we'll ask some targeted questions next to help clarify.")
                        else:
                            st.success(f"Okay, I've noted these symptoms: {', '.join([s.replace('_', ' ') for s in st.session_state.initial_symptoms])}.")
                        
                        for symptom_code in st.session_state.initial_symptoms:
                            if symptom_code in st.session_state.symptom_dict:
                                st.session_state.symptom_dict[symptom_code] = 1
                        
                        set_step("clarify_symptoms")
                        st.rerun()

if st.session_state.step == "clarify_symptoms":
    # ... (clarify_symptoms step code remains largely the same as previous version) ...
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
            initial_diseases = get_initial_diseases(st.session_state.initial_symptoms, symptom_to_diseases_map)
            all_symptoms_to_check = set()
            if initial_diseases:
                display_diseases = [d.replace('_', ' ').title() for d in initial_diseases[:3]]
                st.write(f"I'll ask about symptoms commonly related to conditions like: {', '.join(display_diseases)}{'...' if len(initial_diseases) > 3 else ''}.")
                for disease in initial_diseases:
                    for feat in disease_to_feats.get(disease, []):
                        all_symptoms_to_check.add(feat)
            else:
                st.info("Since your initial description was general or I couldn't extract specific symptoms, I'll ask about some common ones.")
                common_general_symptoms_options = ["itching", "skin_rash", "cough", "fever", "headache", "fatigue", "nausea", "chills", "joint_pain", "stomach_pain"]
                common_general_symptoms = [s for s in common_general_symptoms_options if s in original_feature_order][:8]
                all_symptoms_to_check = set(common_general_symptoms)

            symptoms_to_prompt_further = sorted([s for s in list(all_symptoms_to_check) 
                                                 if s not in st.session_state.initial_symptoms and s in original_feature_order])
            
            CHUNK_SIZE = 8
            st.session_state.symptoms_to_prompt_chunks = [symptoms_to_prompt_further[i:i+CHUNK_SIZE]
                                                         for i in range(0, len(symptoms_to_prompt_further), CHUNK_SIZE)]
            st.session_state.current_chunk_index = 0
            st.session_state.chat_history = [] 
            st.session_state.current_chunk_selections = {}

        for msg_data in st.session_state.chat_history:
            with st.chat_message(msg_data["role"]):
                st.markdown(msg_data["content"], unsafe_allow_html=True if msg_data["role"] == "assistant" else False)
        
        if st.session_state.current_chunk_index < len(st.session_state.symptoms_to_prompt_chunks):
            current_chunk = st.session_state.symptoms_to_prompt_chunks[st.session_state.current_chunk_index]
            
            if current_chunk:
                question_header = "**Are you experiencing any of the following?**<br><small>Please check all that apply, or 'None of these'.</small><br>"
                if not st.session_state.chat_history or not st.session_state.chat_history[-1]["content"].startswith("**Are you experiencing"):
                     st.session_state.chat_history.append({"role": "assistant", "content": question_header})
                     with st.chat_message("assistant"):
                        st.markdown(question_header, unsafe_allow_html=True)
                
                form_key = f"symptom_form_{st.session_state.current_chunk_index}"
                with st.form(key=form_key):
                    _form_selections = {} 
                    for symptom_code in current_chunk:
                        symptom_display_name = symptom_code.replace('_', ' ').capitalize()
                        explanation = symptom_explanations.get(symptom_code.lower(), "")
                        checkbox_label = f"**{symptom_display_name}** {explanation}"
                        _form_selections[symptom_code] = st.checkbox(
                            checkbox_label, 
                            key=f"chk_{symptom_code}_{st.session_state.current_chunk_index}"
                        )
                    
                    none_applies = st.checkbox(
                        "**None of these symptoms apply to me**", 
                        key=f"none_chk_{st.session_state.current_chunk_index}"
                    )
                    
                    submitted = st.form_submit_button("Confirm Selections and Continue")

                if submitted:
                    user_selection_summary = []
                    if none_applies:
                        for symptom_code in current_chunk:
                            st.session_state.symptom_dict[symptom_code] = 0
                        user_selection_summary.append("None of these")
                    else:
                        for symptom_code in current_chunk:
                            if _form_selections.get(symptom_code, False):
                                st.session_state.symptom_dict[symptom_code] = 1
                                user_selection_summary.append(symptom_code.replace('_', ' ').capitalize())
                            else:
                                st.session_state.symptom_dict[symptom_code] = 0
                    
                    if not user_selection_summary and not none_applies:
                        st.warning("Please make a selection for the symptoms listed, or choose 'None of these symptoms apply to me'.")
                    else:
                        user_reply_content = "I selected: " + (", ".join(user_selection_summary) if user_selection_summary else "None of these (as indicated by checkbox)")
                        st.session_state.chat_history.append({"role": "user", "content": user_reply_content})
                        st.session_state.current_chunk_index += 1
                        st.rerun()
            else:
                st.session_state.current_chunk_index += 1
                st.rerun()
        else: 
            st.success("Great, thank you for answering those questions!")
            st.balloons()
            if st.button("See My Symptom Summary and Potential Insights", type="primary"):
                set_step("show_results")
                st.rerun()

if st.session_state.step == "show_results":
    with st.container():
        st.header("📝 Here's a Summary Based on Your Symptoms")
        
        active_s_codes = [s_code for s_code, flag in st.session_state.symptom_dict.items() 
                           if flag == 1 and s_code in original_feature_order]
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
            else: st.markdown("No specific symptoms were confirmed during our chat.")

        with col2:
            st.subheader("Potential Insights (from AI Model):")
            with st.spinner("Analyzing your symptoms with our model..."):
                # Prepare input for the local model
                input_data_list = [st.session_state.symptom_dict.get(s, 0) for s in original_feature_order]
                input_df = pd.DataFrame([input_data_list], columns=original_feature_order)
                
                try:
                    # Use the loaded ml_model directly
                    probabilities_array = ml_model.predict_proba(input_df)[0] # Get the first (and only) row of probabilities
                    
                    # disease_names_from_model was loaded from model_disease_names.json, should match model output
                    disease_probs = dict(zip(disease_names_from_model, probabilities_array))
                    sorted_diseases_probs = sorted(disease_probs.items(), key=lambda x: -x[1]) # Sort by prob desc

                    if not any(p > 0.01 for _, p in sorted_diseases_probs[:5]):
                         st.info("Based on the information provided, the model did not strongly point to any specific conditions from its knowledge base.")
                    else:
                        st.markdown("Our model suggests the following conditions might be worth considering with a healthcare professional. **Please remember, these are not diagnoses.**")
                        for disease, prob in sorted_diseases_probs[:5]: 
                            if prob > 0.01:
                                st.write(f"- **{disease.replace('_', ' ').title()}**: Model indicates a {prob:.0%} likelihood based on symptom patterns.")
                except Exception as e:
                    st.error(f"Error during model prediction: {e}")
                    sorted_diseases_probs = [] # Ensure it's defined for LLM part

        st.divider()

        if st.session_state.aoai_client:
            st.subheader("💬 A Little More About What This Might Mean (from AI Assistant):")
            with st.spinner("Asking AI assistant for a general explanation..."):
                top_cond_llm_parts = []
                if 'sorted_diseases_probs' in locals() and sorted_diseases_probs: 
                    for d, p in sorted_diseases_probs[:3]: 
                        if p > 0.05: 
                            top_cond_llm_parts.append(f"{d.replace('_', ' ').title()} (model indicated {p:.0%} likelihood)")
                
                active_s_text_llm = [s_code.replace('_', ' ').capitalize() for s_code in active_s_codes]
                
                interpretation_prompt = f"""You are a friendly, empathetic AI health assistant.
                    The user has reported the following symptoms: {', '.join(active_s_text_llm) if active_s_text_llm else "None specifically confirmed"}.
                    A separate pattern-matching model has suggested potential considerations (NOT diagnoses): {', '.join(top_cond_llm_parts) if top_cond_llm_parts else "None were strongly indicated by the model at this time"}.
                    
                    Please provide a general, reassuring, and easy-to-understand explanation.
                    1. Briefly acknowledge the user's reported symptoms.
                    2. Reassure the user. Explain that the 'potential considerations' from the model are based on symptom patterns and are not diagnoses.
                    3. If specific conditions were listed by the model (in top_cond_llm_parts), briefly and gently mention them as possibilities to discuss with a doctor if symptoms align or cause concern. Do not overstate their certainty. If none were strongly indicated, acknowledge that and emphasize general well-being.
                    4. Provide general, safe advice for managing mild symptoms (e.g., monitor symptoms, rest, stay hydrated).
                    5. Strongly emphasize the importance of consulting a doctor for an accurate diagnosis and treatment plan, especially if symptoms are severe, worsening, persistent, or if the user is worried.
                    6. Conclude with a clear and prominent disclaimer: "Important Reminder: I am an AI assistant designed for informational purposes only and cannot provide medical advice or a diagnosis. The information shared should not replace consultation with a qualified healthcare professional. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition."
                    
                    Keep your tone calm, supportive, and easy to read. Avoid medical jargon where possible or explain it simply."""
                
                messages = [{"role": "system", "content": "You are a helpful and empathetic AI assistant focused on health information."},
                            {"role": "user", "content": interpretation_prompt}]
                interpretation = llm_call_azure(messages, st.session_state.aoai_client, st.session_state.aoai_deployment_name)
                
                if "Error" not in interpretation: 
                    st.markdown(interpretation)
                else: 
                    st.warning("Could not get a detailed explanation from the AI assistant at this time. Please see the general advice below.")
        else:
            st.markdown("""**Important Reminder:** This tool is for informational purposes only and does not provide a medical diagnosis. 
                          It's always best to consult with a qualified healthcare professional for any health concerns or before making any decisions related to your health.
                          If you are feeling unwell, please monitor your symptoms, rest, and stay hydrated. If your symptoms are severe, worsen, or persist, or if you are worried, seek medical advice promptly.""")
        st.divider()
        if st.button("Start Over", key="start_over_btn"):
            keys_to_clear = ['step', 'initial_symptoms', 'symptom_dict', 'chat_history', 
                             'symptoms_to_prompt_chunks', 'current_chunk_index', 'user_desc_input', 
                             'current_chunk_selections']
            for key in keys_to_clear:
                if key in st.session_state: del st.session_state[key]
            
            if 'app_resources' in st.session_state:
                _symptom_defaults = st.session_state.app_resources[6] # symptom_defaults is now 7th item (index 6)
                st.session_state.symptom_dict = _symptom_defaults.copy()
            elif 'original_feature_order' in globals():
                st.session_state.symptom_dict = {col: 0 for col in original_feature_order}
            st.rerun()

st.markdown("---")
st.markdown("<small>Disclaimer: AI Symptom Guide for informational purposes only. Not medical advice. Consult a physician for health concerns.</small>", unsafe_allow_html=True)