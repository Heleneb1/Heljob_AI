import logging
import tempfile
import os
import streamlit as st
from openai import OpenAI
import fitz  # PyMuPDF
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langdetect import detect

# Configuration générale
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
MODEL_NAME = "gpt-4o-mini"

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Initialisation session_state
for key in ["cv_data", "job_offer", "advice"]:
    if key not in st.session_state:
        st.session_state[key] = None

# Initialiser current_step séparément avec une valeur par défaut
if "current_step" not in st.session_state:
    st.session_state.current_step = "Charger les données"


# Fonctions utilitaires
def ingest_pdf(file):
    try:
        pdf_document = fitz.open(stream=file.read(), filetype="pdf")
        data = [Document(page_content=page.get_text(), metadata={"page_number": i})
                for i, page in enumerate(pdf_document)]
        pdf_document.close()
        return data
    except Exception as e:
        logging.error(f"Erreur lors du chargement du PDF : {e}")
        st.error("Erreur lors du chargement du PDF.")
        return None


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    return splitter.split_documents(documents)


def openai_request(messages, model=MODEL_NAME, max_tokens=400):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        if "quota" in str(e).lower() or "insufficient" in str(e).lower():
            st.error("🚫 Tu as atteint le quota de requêtes OpenAI pour le moment. Réessaie plus tard ou vérifie ton plan.")
        else:
            st.error(f"❌ Une erreur est survenue : {e}")
        logging.error(f"Erreur API OpenAI: {e}")
        return None


def sauvegarder_fichier_temporaire(contenu):
    try:
        with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8") as temp_file:
            temp_file.write(contenu)
            return temp_file.name
    except Exception as e:
        logging.error(f"Erreur lors de la sauvegarde du fichier temporaire : {e}")
        st.error("Impossible de sauvegarder le fichier temporaire.")
        return None


def afficher_animation_chargement(func, *args, **kwargs):
    with st.spinner("Chargement en cours..."):
        return func(*args, **kwargs)


# Fonctions OpenAI
def adapter_cv_et_generer_lettre(cv_data, job_offer):
    try:
        # Extraire le texte du CV
        cv_text = "\n".join([doc.page_content for doc in cv_data])
        
        lang = detect(job_offer)
        prompt = f"CV: {cv_text[:2000]}\n\nOffre: {job_offer[:2000]}\n\n"
        if lang == "en":
            prompt += "Adapt the CV to the job offer and write a professional cover letter in English."
        else:
            prompt += "Adapte le CV à l'offre et rédige une lettre de motivation professionnelle en français."

        messages = [
            {"role": "system", "content": "Tu es un assistant spécialisé en CV."},
            {"role": "user", "content": prompt}
        ]
        return openai_request(messages, max_tokens=600)
    except Exception as e:
        logging.error(f"Erreur API OpenAI: {e}")
        st.error("Erreur lors de l'adaptation du CV ou génération de la lettre.")
        return None


def generate_cv_advice(user_question):
    messages = [
        {"role": "system", "content": "Tu es un assistant spécialisé en CV."},
        {"role": "user", "content": user_question}
    ]
    return openai_request(messages)


# Fonction principale
def main():
    st.title("Heljob-AI ✨")
    st.sidebar.title("Menu 📖")
    
    # Liste des étapes
    steps = ["Charger les données", "Adapter le CV et générer la lettre", "Demander conseil"]
    
    # S'assurer que current_step est valide
    if st.session_state.current_step not in steps:
        st.session_state.current_step = steps[0]
    
    # Radio buttons pour la navigation
    step = st.sidebar.radio(
        "Étapes", 
        steps,
        index=steps.index(st.session_state.current_step)
    )
    
    # Mettre à jour l'étape courante
    st.session_state.current_step = step

    # ---------------------
    # Charger les données
    # ---------------------
    if step == "Charger les données":
        st.subheader("Chargement des données")
        cv_file = st.file_uploader("Télécharge ton CV ⬇️ (PDF)", type="pdf")
        job_offer_file = st.file_uploader("Télécharge l'offre d'emploi ⬇️ (TXT)", type="txt")
        job_offer_text = st.text_area("Ou colle l'offre d'emploi ici 📋 :", height=150)

        if st.button("Charger les données"):
            if cv_file:
                st.session_state.cv_data = ingest_pdf(cv_file)

            if job_offer_text.strip():
                st.session_state.job_offer = job_offer_text
            elif job_offer_file:
                try:
                    st.session_state.job_offer = job_offer_file.read().decode("utf-8")
                except Exception as e:
                    st.error(f"Erreur lors de la lecture de l'offre d'emploi : {e}")

            if st.session_state.get("cv_data") and st.session_state.get("job_offer"):
                st.success("Les données ont été chargées avec succès. 🎉")
            else:
                st.error("Veuillez fournir un CV et une offre avant de continuer 🛑.")
        
        # Bouton Continuer
        if st.session_state.get("cv_data") and st.session_state.get("job_offer"):
            if st.button("Continuer ➡️", type="secondary"):
                st.session_state.current_step = "Adapter le CV et générer la lettre"
                st.rerun()

    # ---------------------
    # Adapter le CV et générer la lettre
    # ---------------------
    elif step == "Adapter le CV et générer la lettre":
        st.header("Adapter le CV et générer la lettre de motivation")
        if st.session_state.get("cv_data") and st.session_state.get("job_offer"):
            if st.button("Lancer l'adaptation et la génération"):
                result = afficher_animation_chargement(
                    adapter_cv_et_generer_lettre,
                    st.session_state.cv_data,
                    st.session_state.job_offer
                )
                if result:
                    st.success("✅ Génération terminée !")
                    st.text_area("Résultat :", result, height=400)
                    
                    temp_filename = sauvegarder_fichier_temporaire(result)
                    if temp_filename:
                        with open(temp_filename, "r", encoding="utf-8") as f:
                            file_data = f.read()
                        st.download_button(
                            label="📥 Télécharger le résultat",
                            data=file_data,
                            file_name="CV_et_lettre.txt",
                            mime="text/plain"
                        )
            
            # Bouton Continuer vers la section conseil
            if st.button("Continuer vers Demander conseil ➡️", type="secondary"):
                st.session_state.current_step = "Demander conseil"
                st.rerun()
        else:
            st.warning("⚠️ Veuillez charger le CV et l'offre avant de continuer.")
            if st.button("⬅️ Retour au chargement"):
                st.session_state.current_step = "Charger les données"
                st.rerun()

    # ---------------------
    # Demander conseil
    # ---------------------
    elif step == "Demander conseil":
        st.header("Demander conseil sur ton CV")
        user_input = st.text_area("Pose ta question :", "", height=100)
        
        if st.button("Envoyer"):
            if user_input.strip():
                if "advice_cache" not in st.session_state:
                    st.session_state["advice_cache"] = {}

                if user_input in st.session_state["advice_cache"]:
                    advice = st.session_state["advice_cache"][user_input]
                else:
                    advice = afficher_animation_chargement(generate_cv_advice, user_input)
                    if advice:
                        st.session_state["advice_cache"][user_input] = advice

                if advice:
                    st.info(advice)

                    advice_file = sauvegarder_fichier_temporaire(advice)
                    if advice_file:
                        with open(advice_file, "r", encoding="utf-8") as f:
                            advice_data = f.read()
                        st.download_button(
                            label="📥 Télécharger le conseil",
                            data=advice_data,
                            file_name="advice.txt",
                            mime="text/plain"
                        )
            else:
                st.warning("Veuillez entrer une question avant de continuer.")


if __name__ == "__main__":
    main()