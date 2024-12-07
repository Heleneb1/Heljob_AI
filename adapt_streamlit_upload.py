import logging
import tempfile
import time
import os

# Bibliothèques tierces
import streamlit as st
import ollama
import fitz  # PyMuPDF pour lire les fichiers PDF
from langchain.schema import Document
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langdetect import detect
#  Pour ouvrir en localhost: streamlit run adapt_streamlit_upload.py

# Configuration du système de journalisation pour suivre l'exécution du programme
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Constantes utilisées dans le programme
MODEL_NAME = "llama3.2"  # Nom du modèle utilisé
EMBEDDING_MODEL = "nomic-embed-text"  # Modèle pour les embeddings
VECTOR_STORE_NAME = "simple-rag"  # Nom de la base de données vectorielle
PERSIST_DIRECTORY = "./chroma_db"


def ingest_pdf(file):
    """Charger et traiter un document PDF uploadé."""
    # import fitz  # PyMuPDF
    try:
        pdf_document = fitz.open(stream=file.read(), filetype="pdf")
        data = [
            Document(page_content=page.get_text(), metadata={"page_number": page_num})
            for page_num, page in enumerate(pdf_document)
        ]
        pdf_document.close()
        return data
    except Exception as e:
        logging.error(f"Erreur lors du chargement du PDF : {e}")
        return None

# Fonction pour lire un fichier texte (comme une offre d'emploi)
def read_text_file(file):
    """Lire un fichier texte uploadé."""
    try:
        return file.read().decode("utf-8")  # Assure-toi que c'est bien encodé en UTF-8
    except Exception as e:
        logging.error(f"Erreur lors du chargement du fichier texte : {e}")
        return None

# Fonction pour diviser le contenu en morceaux plus petits pour traitement
def split_documents(documents):
    """Diviser les documents en plus petits morceaux."""
    # Diviseur de texte basé sur la longueur, avec chevauchement pour le contexte
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    chunks = text_splitter.split_documents(documents)
    logging.debug("Documents divisés en morceaux.")
    return chunks

# Fonction pour adapter un CV à une offre d'emploi
def adapter_cv_ollama(cv_data, job_offer):
    """Adapter le CV avec Ollama."""
    
    try:
        lang = detect(job_offer)  # Retourne 'en' pour l'anglais, 'fr' pour le français, etc.
        if lang == "en":
            prompt = f"""
            Here is my structured CV: {cv_data}
            Here is the job offer: {job_offer}
            Suggest relevant changes to make my resume a better match for this offer.
            Provided a complete CV, passing the ATS, adapted and updated to best match the offer according to my background.
                        """
        else:
            prompt = f"""
            Voici mon CV structuré : {cv_data}
            Voici l'offre d'emploi : {job_offer}
            Propose des modifications pertinentes pour que mon CV corresponde mieux à cette offre.
            Fournis un CV complet, passant les ATS, adapté et mis à jour pour correspondre au mieux à l'offre en fonction de mon parcours.
                        """
        logging.debug("Début de l'adaptation du CV...")
        
        # Envoi de la requête au modèle et récupération de la réponse
        response = ollama.generate(model=MODEL_NAME, prompt=prompt)
        
        if not response or "response" not in response:
            raise ValueError("Erreur dans la réponse d'Ollama.")

        cv_modifie = response.get("response", "")
        logging.debug("Adaptation du CV terminée.")
        return cv_modifie
    except Exception as e:
        logging.error(f"Erreur lors de l'adaptation du CV : {e}")
        return None

# Fonction pour générer une lettre de motivation à partir d'un CV et d'une offre
def generer_lettre_motivation_ollama(cv_data, job_offer):
    """Générer une lettre de motivation avec Ollama."""
    try:
        lang = detect(job_offer)  # Retourne 'en' pour l'anglais, 'fr' pour le français, etc.
        if lang == "en":
            prompt = f"""
            Here is my structured CV: {cv_data}
            Here is the job offer: {job_offer}
            Write a professional cover letter tailored to this job offer. Mention the recruiting company and explain how my values align with theirs. The cover letter should be in English.
            """
        else:
            prompt = f"""
            Voici mon CV structuré : {cv_data}
            Voici l'offre d'emploi : {job_offer}
            Rédige une lettre de motivation professionnelle adaptée. Parle de l'entreprise qui recrute et pourquoi mes valeurs y correspondent. La lettre doit être en français.
            """

        logging.debug("Début de la génération de la lettre de motivation...")
        
        # Envoi de la requête au modèle et récupération de la réponse
        response = ollama.generate(model=MODEL_NAME, prompt=prompt)
        lettre_motivation = response.get("response", "")
        logging.debug("Génération de la lettre de motivation terminée.")
        return lettre_motivation
    except Exception as e:
        logging.error(f"Erreur lors de la génération de la lettre : {e}")
        return None

# Sauvegarde temporaire 
def sauvegarder_fichier_temporaire(contenu):
    """Sauvegarder le contenu dans un fichier temporaire."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8") as temp_file:
            temp_file.write(contenu)
            temp_filename = temp_file.name  # Retourne le nom du fichier temporaire
        logging.debug(f"Fichier temporaire sauvegardé avec succès : {temp_filename}")
        return temp_filename
    except Exception as e:
        logging.error(f"Erreur lors de la sauvegarde du fichier temporaire : {e}")
        return None

# def clean_temp_file(temp_file):
#     """Supprimer un fichier temporaire et gérer les erreurs."""
#     try:
#         if os.path.exists(temp_file):
#             os.remove(temp_file)
#             logging.info(f"Fichier temporaire supprimé : {temp_file}")
#         else:
#             logging.warning(f"Le fichier {temp_file} n'existe pas.")
#     except Exception as e:
#         logging.error(f"Erreur lors de la suppression du fichier {temp_file}: {e}")

def afficher_animation_chargement():
    """Afficher une animation de chargement avec Streamlit."""
    with st.spinner("Chargement en cours..."):
        time.sleep(60)  # Simule un délai pour l'animation
        
    
# Fonction principale orchestrant tout le processus
def main():
    # Ajouter du CSS personnalisé
    st.markdown(
        """
        <style>
        h1 {
            color: #0d92a3;
            padding-bottom: 2rem;
            font-size: 4rem;
        }
        textarea {
            border: 2px solid #0d92a3 !important; /* Couleur personnalisée */
            border-radius: 5px; /* Coins arrondis */
            padding: 10px; /* Espacement interne */
            box-shadow: 0 0 5px rgba(0, 0, 0, 0.1); /* Ombre légère */
        }
        /* Cibler le bouton secondaire "Browse files" */
        button[data-testid="stBaseButton-secondary"] {
            border: 2px solid #0d92a3 !important;
            background-color: #0d92a3 !important;
            color: white !important;
            border-radius: 5px; /* Coins arrondis */
            padding: 10px; /* Espacement interne */
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); /* Ombre légère */
            font-weight: bold;
            font-size: 14px;
        }
        /* Effet au survol */
        button[data-testid="stBaseButton-secondary"]:hover {
            background-color: #0b7983 !important; /* Couleur plus foncée */
            border-color: #0b7983 !important;
        }
        /* Effet au clic */
        button[data-testid="stBaseButton-secondary"]:active {
            background-color: #151515 !important; /* Couleur encore plus foncée */
            border-color: #fff !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    if "cv_data" not in st.session_state:
        st.session_state.cv_data = None

    if "job_offer" not in st.session_state:
        st.session_state.job_offer = None

    st.title("Heljob-AI 🤖" )

    st.text("CV Assistant ☑️")
    
    st.sidebar.title("Menu 📖")
    # Descriptif dans la barre latérale
    st.sidebar.markdown("Insère ton CV et ton offre, et laisse la magie opérer ✨")
    step = st.sidebar.radio("Étapes", ["Charger les données", "Adapter le CV", "Générer la lettre"])

    # Charger le CV et l'offre
    if step == "Charger les données":
        st.subheader("Chargement des données")
                
        # Upload des fichiers
        cv_file = st.file_uploader("Télécharge ton CV ⬇️ (PDF)", type="pdf")
        st.text("Télécharge, ou colle le texte de l'offre d'emploi dans le champs dédié")
        job_offer_file = st.file_uploader("Télécharge l'offre d'emploi ⬇️ (TXT)", type="txt")
        # Champ pour coller l'annonce
        job_offer_text = st.text_area(" Ou colle l'offre d'emploi ici 📋 :", height=150)

        if st.button("Charger les données"):
            
            if cv_file:
                try:
                    st.session_state.cv_data = ingest_pdf(cv_file)
                except Exception as e:
                    st.error(f"Erreur lors de la lecture du CV : {e}")
            
            # Vérifier si l'utilisateur a collé ou uploadé l'offre
            if job_offer_text.strip():
                st.session_state.job_offer = job_offer_text
            elif job_offer_file:
                try:
                    st.session_state.job_offer = job_offer_file.read().decode("utf-8")
                except Exception as e:
                    st.error(f"Erreur lors de la lecture de l'offre d'emploi : {e}")
            
            if st.session_state.cv_data and st.session_state.job_offer:
                st.success("Les données ont été chargées avec succès. 🎉")
            else:
                st.error("Veuillez fournir un CV au format PDF et une offre d'emploi de type texte avant de continuer 🛑.")

    # Adapter le CV à l'offre
    elif step == "Adapter le CV":
        st.header("Adapter le CV à l'offre")
        if st.session_state.cv_data and st.session_state.job_offer:
            if st.button("Adapter le CV"):
                afficher_animation_chargement()
                cv_adapte = adapter_cv_ollama(st.session_state.cv_data, st.session_state.job_offer)
                if cv_adapte:
                    temp_filename = sauvegarder_fichier_temporaire(cv_adapte)  # Sauvegarde dans un fichier temporaire
                    if temp_filename:
                        st.success("CV adapté généré avec succès.")
                        st.download_button(
                            label="Télécharger le CV adapté",
                            data=open(temp_filename, "r", encoding="utf-8").read(),  # Lit le fichier temporaire pour le téléchargement ajout utf-8 pour lecture des caractères spéciaux
                            file_name="CV_adapte.txt",
                            mime="text/plain"
                        )
        else:
            st.warning("Veuillez charger le CV et l'offre avant de continuer.")

    # Générer la lettre de motivation
    elif step == "Générer la lettre":
        st.header("Générer une lettre de motivation")
        if st.session_state.cv_data and st.session_state.job_offer:
            if st.button("Générer la lettre"):
                afficher_animation_chargement()
                lettre_motivation = generer_lettre_motivation_ollama(st.session_state.cv_data, st.session_state.job_offer)
                if lettre_motivation:
                    temp_filename=sauvegarder_fichier_temporaire(lettre_motivation)
                    if temp_filename:
                        st.success("Lettre de motivation générée avec succès.")
                        st.download_button(
                        label="Télécharger la lettre de motivation",
                        data=open(temp_filename, "r", encoding="utf-8").read(),
                        file_name="Lettre_motivation.txt",
                        mime="text/plain"
                    )
        else:
            st.warning("Veuillez charger le CV et l'offre avant de continuer.")

       


if __name__ == "__main__":
    main()
