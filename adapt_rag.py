import os
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
import fitz  # Bibliothèque PyMuPDF pour lire les fichiers PDF

# Configuration du système de journalisation pour suivre l'exécution du programme
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
# en local
# Constantes utilisées dans le programme
# CV_PATH = os.path.abspath("./data/Cv.pdf")  # Chemin du CV
# OFFER_PATH = os.path.abspath("./data/offer.txt")  # Chemin de l'offre d'emploi
# MODEL_NAME = "llama3.2"  # Nom du modèle utilisé
# EMBEDDING_MODEL = "nomic-embed-text"  # Modèle pour les embeddings
# VECTOR_STORE_NAME = "simple-rag"  # Nom de la base de données vectorielle

# Constantes pour la version OpenAI
CV_PATH = os.path.abspath("./data/Cv.pdf")
OFFER_PATH = os.path.abspath("./data/offer.txt")
MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "text-embedding-3-small"
VECTOR_STORE_NAME = "simple-rag"


# Fonction pour charger un fichier PDF et l'extraire en texte
def ingest_pdf(doc_path=CV_PATH):
    """Charger et traiter un document PDF."""
    try:
        pdf_document = fitz.open(doc_path)  # Ouvre le fichier PDF
        logging.info("PDF chargé avec succès.")
        
        # Extraction du texte de chaque page du PDF sous forme de liste de `Document`
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
def read_text_file(file_path=OFFER_PATH):
    """Lire un fichier texte."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError as e:
        logging.error(f"Erreur d'encodage : {e}")
        return None

logging.info("Chargement des constantes terminé.")

# Fonction pour diviser le contenu en morceaux plus petits pour traitement
def split_documents(documents):
    """Diviser les documents en plus petits morceaux."""
    # Diviseur de texte basé sur la longueur, avec chevauchement pour le contexte
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    chunks = text_splitter.split_documents(documents)
    logging.info("Documents divisés en morceaux.")
    return chunks

# Création d'une base de données vectorielle pour rechercher des informations
def create_vector_db(chunks):
    """Créer une base de données vectorielle à partir des morceaux de documents."""
    # Téléchargement du modèle d'embedding si nécessaire
    ollama.pull(EMBEDDING_MODEL)

    # Création de la base de données vectorielle
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model=EMBEDDING_MODEL),
        collection_name=VECTOR_STORE_NAME,
    )
    logging.info("Base de données vectorielle créée.")
    return vector_db


# Instance du modèle de langage utilisé
llm = ChatOllama(model=MODEL_NAME)

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
    # Charger le CV
    cv_data = ingest_pdf(CV_PATH)
    if not cv_data:
        raise ValueError("Impossible de charger le CV.")

    # Charger l'offre d'emploi
    job_offer = read_text_file(OFFER_PATH)
    if not job_offer:
        raise ValueError("Impossible de charger l'offre d'emploi.")

    # Diviser les documents en morceaux
    cv_chunks = split_documents(cv_data)

    # Créer la base de données vectorielle
    vector_db = create_vector_db(cv_chunks)

    # Créer le retriever pour effectuer des recherches complexes
    retriever = create_retriever(vector_db, llm, cv_data, job_offer)

    # Adapter le CV à l'offre d'emploi
    cv_adapte = adapter_cv_ollama(cv_data, job_offer)
    if cv_adapte:
        sauvegarder_fichier(cv_adapte, "./data/CV_adapte.txt")

    # Générer la lettre de motivation adaptée
    lettre_motivation = generer_lettre_motivation_ollama(cv_data, job_offer)
    if lettre_motivation:
        sauvegarder_fichier(lettre_motivation, "./data/Lettre_motivation.txt")
    # Créer la chaîne
    chain = create_chain(retriever, llm)
    context = f"CV: {' '.join([doc.page_content for doc in cv_data])}\nJob Offer: {job_offer}"

    # logging.info(f"cv_data: {cv_data[:500]}")  # Affiche les 500 premiers caractères
    # logging.info(f"job_offer: {job_offer[:500]}")

    
    # Poser une question
    question = "Quels points de mon CV dois-je améliorer pour cette offre ?"
    # Get the response
    response = chain.invoke({
        "context": context,
        "question": question
    })
    print(response)

if __name__ == "__main__":
    main()
