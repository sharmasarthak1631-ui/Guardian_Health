import streamlit as st
import pandas as pd
import re
import json
from pypdf import PdfReader
import google.generativeai as genai

# --- CONFIGURATION & SETUP ---
st.set_page_config(page_title="Medical Note Analyzer & Policy Checker", layout="wide")

# Set up your API Key
API_KEY: str = "AIzaSyD5vCEN59Eh79ROciqOcNc0GzcdefBiWjw"
if API_KEY != "AIzaSyD5vCEN59Eh79ROciqOcNc0GzcdefBiWjw":
    genai.configure(api_key=API_KEY)

# --- THE AGENT CONSTITUTION ---
COMPLIANCE_CONSTITUTION = """
Act as a Healthcare Compliance Agent. Your rules are:
- Accuracy First: If a clinical note is too vague to assign a specific ICD-10 code, do not guess. Output 'Ambiguous: Please clarify [X symptom].'
- No Medical Advice: Never tell the patient what to do; only categorize the doctor's existing notes.
- PII Scrubbing: If you detect a Name, SSN, or Phone Number in the summary, redact it immediately with [REDACTED].
- Policy Grounding: Only approve a claim if you find a direct match in the provided Policy Document.
"""


# --- HELPER FUNCTIONS ---

def extract_text_from_pdf(uploaded_file):
    """Extracts text from an uploaded PDF file."""
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def pii_guardrail(text):
    """Hardcoded regex check as a secondary defense layer."""
    flags = []
    patterns = {
        "Email": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        "Phone Number": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        "SSN": r'\b\d{3}-\d{2}-\d{4}\b',
    }
    for pii_type, pattern in patterns.items():
        if re.search(pattern, text):
            flags.append(f"{pii_type} detected")
    return len(flags) > 0, flags


def analyze_medical_note(text):
    """Uses LLM to summarize and extract keywords + evidence via JSON."""
    if API_KEY == "AIzaSyD5vCEN59Eh79ROciqOcNc0GzcdefBiWjw":
        # Mock JSON response for offline testing
        mock_findings = [
            {"keyword": "pharyngitis",
             "evidence": "Patient presents with severe sore throat and red tonsils for 2 days."},
            {"keyword": "fever", "evidence": "Temperature recorded at 101.2 F."}
        ]
        return "Patient presents with acute pharyngitis and a mild fever for 2 days.", mock_findings

    # Initialize model with the Constitution
    model = genai.GenerativeModel(
        model_name='gemini-2.5-flash',
        system_instruction=COMPLIANCE_CONSTITUTION
    )

    prompt = f"""
    Read the medical note and provide a JSON response with the following exact structure:
    {{
        "summary": "A short, clinical summary of the note. Obey the PII scrubbing rules.",
        "findings": [
            {{
                "keyword": "A 1-3 word medical term or diagnosis for ICD-10 lookup. If vague, output 'Ambiguous: Please clarify...'",
                "evidence": "The exact, verbatim sentence from the medical note that justifies this keyword."
            }}
        ]
    }}

    Medical Note:
    {text}
    """
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(response_mime_type="application/json")
        )
        data = json.loads(response.text)
        return data.get("summary", "No summary provided."), data.get("findings", [])
    except Exception as e:
        st.error(f"LLM Error during analysis: {e}")
        return "Error generating summary.", []


def search_icd10(findings, csv_path="icd10_codes.csv"):
    """Searches local CSV for codes and appends the evidence from the note."""
    try:
        # Mock database for the hackathon
        df = pd.DataFrame({
            "Code": ["J02.9", "R50.9", "E11.9", "I10"],
            "Description": ["Acute pharyngitis, unspecified", "Fever, unspecified", "Type 2 diabetes mellitus",
                            "Essential hypertension"]
        })
        # If using a real CSV: df = pd.read_csv(csv_path)

        results = []
        for item in findings:
            kw = item.get("keyword", "")
            evidence = item.get("evidence", "")

            # Skip if the LLM flagged it as ambiguous
            if "Ambiguous" in kw:
                results.append({
                    "Code": "N/A",
                    "Description": kw,
                    "Evidence from Note": evidence
                })
                continue

            match = df[df['Description'].str.contains(kw, case=False, na=False)]

            if not match.empty:
                for _, row in match.iterrows():
                    results.append({
                        "Code": row["Code"],
                        "Description": row["Description"],
                        "Evidence from Note": evidence
                    })

        if results:
            return pd.DataFrame(results).drop_duplicates(subset=["Code", "Description"])
        return pd.DataFrame()

    except Exception as e:
        st.error(f"Error loading ICD-10 database: {e}")
        return pd.DataFrame()


def check_payer_policy(clinical_text, policy_text):
    """Compares clinical text against payer policy rules and returns status."""
    if API_KEY == "AIzaSyD5vCEN59Eh79ROciqOcNc0GzcdefBiWjw":
        return False, "Symptoms do not meet the 3-day duration requirement in Section 2.1 of the policy. Note states symptoms present for 2 days."

    # Initialize model with the Constitution
    model = genai.GenerativeModel(
        model_name='gemini-2.5-flash',
        system_instruction=COMPLIANCE_CONSTITUTION
    )

    prompt = f"""
    Compare the provided Medical Note against the provided Payer Policy.
    Determine if the patient's presentation and symptoms in the Medical Note meet the criteria outlined in the Payer Policy to approve the claim.

    Return your analysis strictly in this JSON format:
    {{
        "meets_criteria": true or false,
        "reason": "If false, cite the specific section/rule not met and why. If true, briefly state how criteria are met."
    }}

    ---
    Medical Note:
    {clinical_text}

    ---
    Payer Policy:
    {policy_text}
    """
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(response_mime_type="application/json")
        )
        data = json.loads(response.text)
        return data.get("meets_criteria", False), data.get("reason", "Unable to determine policy adherence.")
    except Exception as e:
        return False, f"Error analyzing policy: {e}"


# --- STREAMLIT UI ---

st.title("🩺 AI Medical Note Analyzer & Policy Checker")
st.markdown("Extract summaries, map ICD-10 codes with evidence, and verify against Payer Policies.")

col1, col2 = st.columns(2)
with col1:
    note_file = st.file_uploader("1. Upload Medical Note (PDF)", type=["pdf"])
with col2:
    policy_file = st.file_uploader("2. Upload Payer Policy (PDF) - Optional", type=["pdf"])

if note_file is not None:
    st.info("Medical Note uploaded successfully.")
    raw_note_text = extract_text_from_pdf(note_file)

    with st.expander("View Raw Medical Note Text"):
        st.write(raw_note_text)

    raw_policy_text = ""
    if policy_file is not None:
        st.info("Payer Policy uploaded successfully.")
        raw_policy_text = extract_text_from_pdf(policy_file)

    if st.button("Analyze Note & Check Policy", type="primary"):
        with st.spinner("Analyzing with LLM..."):

            # 1. LLM Analysis
            summary, findings = analyze_medical_note(raw_note_text)

            # 2. Guardrail Check (Regex Backup)
            has_pii, pii_flags = pii_guardrail(summary)

            st.divider()

            if has_pii:
                st.error("🚨 **GUARDRAIL ALERT: Unredacted PII Detected in Output!**")
                st.write(f"Flagged categories: {', '.join(pii_flags)}")
                st.warning("The summary has been blocked. Please check the LLM Constitution settings.")
            else:
                st.success("✅ Guardrail Check Passed (No Raw PII Detected)")

                # Summary
                st.subheader("📝 Clinical Summary")
                st.write(summary)

                # ICD-10 Search & Table Generation
                st.subheader("🏥 Suggested ICD-10 Codes with Evidence")
                icd_results = search_icd10(findings)

                if not icd_results.empty:
                    st.dataframe(icd_results, use_container_width=True, hide_index=True)
                else:
                    st.write("No matching ICD-10 codes found or terms were too ambiguous.")

                # 4. Policy Check (if policy file was uploaded)
                if policy_file is not None:
                    st.divider()
                    st.subheader("🛡️ Payer Policy Adjudication")

                    with st.spinner("Cross-referencing against Payer Policy..."):
                        meets_criteria, reason = check_payer_policy(raw_note_text, raw_policy_text)

                        if meets_criteria:
                            st.success(f"**✅ Meets Policy Criteria**\n\n{reason}")
                        else:
                            st.error(f"**❌ Requires Manual Review**\n\n{reason}")