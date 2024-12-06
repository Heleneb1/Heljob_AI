import os
import streamlit as st
import logging
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import ollama
import fitz  # Bibliothèque PyMuPDF pour lire les fichiers PDF pip install PyMuPDF

# streamlit run adapt_streamlit_upload.py Pour ouvrir en localhost

# Configuration du système de journalisation pour suivre l'exécution du programme
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Constantes utilisées dans le programme
CV_PATH = os.path.abspath("./data/Cv.pdf")  # Chemin du CV
OFFER_PATH = os.path.abspath("./data/offer.txt")  # Chemin de l'offre d'emploi
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
    logging.info("Documents divisés en morceaux.")
    return chunks

@st.cache_resource
def load_vector_db():
    """Load or create the vector database."""
    # Pull the embedding model if not already available
    ollama.pull(EMBEDDING_MODEL)

    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)

    if os.path.exists(PERSIST_DIRECTORY):
        vector_db = Chroma(
            embedding_function=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY,
        )
        logging.info("Loaded existing vector database.")
    else:
        # Load and process the PDF document
        data = ingest_pdf(CV_PATH)
        if data is None:
            return None

        # Split the documents into chunks
        chunks = split_documents(data)

        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY,
        )
        vector_db.persist()
        logging.info("Vector database created and persisted.")
    return vector_db

# Fonction pour créer un système de recherche basé sur plusieurs requêtes
def create_retriever(vector_db, llm, cv_data, job_offer):
    """Créer un retriever multi-query."""
    
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],  # Réduisez les variables d'entrée
        template="""
        Voici mon CV structuré : {{cv_data}}
        Voici l'offre d'emploi : {{job_offer}}
        Propose des modifications pertinentes pour que mon CV corresponde mieux à cette offre.
        Fournis un CV adapté et mis à jour.
        Original question: {question}"""
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(),
        llm,
        prompt=QUERY_PROMPT,       
    )
    logging.info("Retriever créé.")
    return retriever

# Fonction pour adapter un CV à une offre d'emploi
def adapter_cv_ollama(cv_data, job_offer):
    """Adapter le CV avec Ollama."""
    try:
        prompt = f"""
        Voici mon CV structuré : {cv_data}
        Voici l'offre d'emploi : {job_offer}
        Propose des modifications pertinentes pour que mon CV corresponde mieux à cette offre.
        Fournis un CV complet, passant les ATS, adapté et mis à jour pour correspondre au mieux à l'offre en fonction de mon parcours.
        """
        logging.info("Début de l'adaptation du CV...")
        
        # Envoi de la requête au modèle et récupération de la réponse
        response = ollama.generate(model=MODEL_NAME, prompt=prompt)
        
        if not response or "response" not in response:
            raise ValueError("Erreur dans la réponse d'Ollama.")

        cv_modifie = response.get("response", "")
        logging.info("Adaptation du CV terminée.")
        return cv_modifie
    except Exception as e:
        logging.error(f"Erreur lors de l'adaptation du CV : {e}")
        return None

# Fonction pour générer une lettre de motivation à partir d'un CV et d'une offre
def generer_lettre_motivation_ollama(cv_data, job_offer):
    """Générer une lettre de motivation avec Ollama."""
    try:
        prompt = f"""
        Voici mon CV structuré : {cv_data}
        Voici l'offre d'emploi : {job_offer}
        Rédige une lettre de motivation professionnelle adaptée, parle de l'entreprise qui recrute et pourquoi mes valeurs y correspondent
        """
        logging.info("Début de la génération de la lettre de motivation...")
        
        # Envoi de la requête au modèle et récupération de la réponse
        response = ollama.generate(model=MODEL_NAME, prompt=prompt)
        lettre_motivation = response.get("response", "")
        logging.info("Génération de la lettre de motivation terminée.")
        return lettre_motivation
    except Exception as e:
        logging.error(f"Erreur lors de la génération de la lettre : {e}")
        return None

# Fonction pour sauvegarder du contenu dans un fichier
def sauvegarder_fichier(contenu, nom_fichier):
    """Sauvegarder le contenu dans un fichier."""
    try:
        with open(nom_fichier, "w", encoding="utf-8") as f:
            f.write(contenu)
        logging.info(f"Fichier {nom_fichier} sauvegardé avec succès.")
    except Exception as e:
        logging.error(f"Erreur lors de la sauvegarde du fichier : {e}")

def create_chain(retriever, llm):
    """Create the chain"""
    # RAG prompt
    template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": lambda x: x}
        | prompt
        | llm
        | StrOutputParser()
    )

    logging.info("Chain created successfully.")
    return chain


    
# Fonction principale orchestrant tout le processus
def main():
    if "cv_data" not in st.session_state:
        st.session_state.cv_data = None

    if "job_offer" not in st.session_state:
        st.session_state.job_offer = None

    st.title("CV Assistant")
    st.sidebar.title("Menu")
    step = st.sidebar.radio("Étapes", ["Charger les données", "Adapter le CV", "Générer la lettre"])

    # Charger le CV et l'offre
    if step == "Charger les données":
        st.header("Chargement des données")
        
        # Upload des fichiers
        cv_file = st.file_uploader("Télécharge ton CV (PDF)", type="pdf")
        job_offer_file = st.file_uploader("Télécharge l'offre d'emploi (TXT)", type="txt")
        
        if st.button("Charger les fichiers"):
            if cv_file and job_offer_file:
                try:
                    # Lire les fichiers uploadés
                    st.session_state.cv_data = ingest_pdf(cv_file)
                    st.session_state.job_offer = job_offer_file.read().decode("utf-8")
                    st.success("Les données ont été chargées avec succès.")
                except Exception as e:
                    st.error(f"Erreur lors du traitement des fichiers : {e}")
            else:
                st.error("Veuillez télécharger un CV et une offre d'emploi.")

    # Adapter le CV à l'offre
    elif step == "Adapter le CV":
        st.header("Adapter le CV à l'offre")
        if st.button("Adapter le CV"):
            if st.session_state.cv_data and st.session_state.job_offer:
                cv_adapte = adapter_cv_ollama(st.session_state.cv_data, st.session_state.job_offer)
                if cv_adapte:
                    sauvegarder_fichier(cv_adapte, "./data/CV_adapte.txt")
                    st.success("CV adapté généré avec succès.")
                    st.download_button(
                        label="Télécharger le CV adapté",
                        data=cv_adapte,
                        file_name="./data/CV_adapte.txt",
                        mime="text/plain"
                    )
            else:
                st.error("Veuillez d'abord charger les données.")

    # Générer la lettre de motivation
    elif step == "Générer la lettre":
        st.header("Générer une lettre de motivation")
        if st.button("Générer la lettre"):
            if st.session_state.cv_data and st.session_state.job_offer:
                lettre_motivation = generer_lettre_motivation_ollama(st.session_state.cv_data, st.session_state.job_offer)
                if lettre_motivation:
                    sauvegarder_fichier(lettre_motivation, "./data/Lettre_motivation.txt")
                    st.success("Lettre de motivation générée avec succès.")
                    st.download_button(
                        label="Télécharger la lettre de motivation",
                        data=lettre_motivation,
                        file_name="./data/Lettre_motivation.txt",
                        mime="text/plain"
                    )
            else:
                st.error("Veuillez d'abord charger les données.")

if __name__ == "__main__":
    main()
