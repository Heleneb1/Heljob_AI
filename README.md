<p align="center">
  <img src="https://raw.githubusercontent.com/Heleneb1/Heljob_AI/main/assets/banner.png" alt="ts-errors banner" />
</p>

# 🚀 HelJob_AI

> Adaptez votre CV et générez des lettres de motivation personnalisées grâce à l'intelligence artificielle

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/streamlit-1.0+-red.svg)](https://streamlit.io/)

[English version](#english-version) | [Version française](#version-française)

---

## Démo en ligne

Vous pouvez essayer l'application en ligne ici : [HelJob_AI Démo](https://heljob.streamlit.app/)

## 📋 Table des matières

- [À propos](#-à-propos)
- [Fonctionnalités](#-fonctionnalités)
- [Prérequis](#-prérequis)
- [Installation](#-installation)
  - [Version locale (LLaMA 3.2)](#version-locale-llama-32---gratuite)
  - [Version OpenAI (GPT-4)](#version-openai-gpt-4)
- [Utilisation](#-utilisation)
- [Structure du projet](#-structure-du-projet)
- [Contribuer](#-contribuer)
- [Licence](#-licence)
- [Auteur](#-auteur)

---

## 🎯 À propos

**HelJob_AI** est un outil innovant qui utilise l'intelligence artificielle pour vous aider dans votre recherche d'emploi. Il analyse votre CV et une offre d'emploi cible, puis génère automatiquement :

- ✅ Un CV adapté aux exigences spécifiques du poste
- ✅ Une lettre de motivation personnalisée et pertinente
- ✅ Des suggestions d'amélioration basées sur l'IA

## ✨ Fonctionnalités

| Fonctionnalité                   | Description                                                    |
| -------------------------------- | -------------------------------------------------------------- |
| 📄 **Extraction intelligente**   | Analyse des fichiers PDF (CV) et TXT (offres d'emploi)         |
| 🎯 **Adaptation du CV**          | Ajustement automatique du contenu selon les exigences du poste |
| ✍️ **Génération de lettre**      | Création d'une lettre de motivation unique et personnalisée    |
| 💬 **Assistant conversationnel** | Posez des questions sur votre CV (version OpenAI)              |
| 🖥️ **Interface intuitive**       | Interface utilisateur simplifiée avec Streamlit                |

## 🔧 Prérequis

- **Python** 3.10 ou plus récent
- **pip** (gestionnaire de paquets Python)
- **Git** pour cloner le dépôt
- **Clé API OpenAI** (uniquement pour la version OpenAI)

---

## 📦 Installation

### Version locale (LLaMA 3.2) - Gratuite

Cette version utilise le modèle open-source LLaMA 3.2 et fonctionne entièrement en local.

#### 1. Cloner le dépôt

```bash
git clone https://github.com/Heleneb1/HelJob_AI.git
cd HelJob_AI
```

#### 2. Créer un environnement virtuel

**Linux/macOS :**

```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows (c'est mon cas 😊):**

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

#### 4. Lancer l'application

```bash
streamlit run adapt_streamlit_upload.py
```

L'application s'ouvrira automatiquement dans votre navigateur à l'adresse `http://localhost:8501`

---

### Version OpenAI (GPT-4)

Cette version utilise l'API OpenAI pour des résultats optimisés et une interaction conversationnelle avancée.

#### 1. Cloner et préparer l'environnement

```bash
git clone https://github.com/Heleneb1/HelJob_AI.git
cd HelJob_AI
python -m venv .venv
```

**Linux/macOS :**

```bash
source .venv/bin/activate
```

**Windows :**

```bash
.venv\Scripts\activate
```

#### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

#### 3. Configurer la clé API OpenAI

Créez le fichier de configuration pour votre clé API :

```bash
mkdir -p .streamlit
```

Créez le fichier `.streamlit/secrets.toml` et ajoutez-y :

```toml
OPENAI_API_KEY = "votre_clé_api_ici"
```

> ⚠️ **Important :** Ne partagez jamais votre clé API publiquement. Le fichier `secrets.toml` est automatiquement ignoré par Git.

#### 4. Lancer l'application OpenAI

```bash
streamlit run adapt_streamlit_upload_openai.py
```

---

## 🎮 Utilisation

1. **Ouvrez l'application** dans votre navigateur
2. **Téléchargez votre CV** (format PDF)
3. **Collez l'offre d'emploi** (format texte)
4. **Cliquez sur "Générer"** pour obtenir :
   - Votre CV adapté
   - Une lettre de motivation personnalisée
5. **(Version OpenAI uniquement)** Utilisez le chat pour poser des questions sur votre CV

## 📁 Structure du projet

```
HelJob_AI/
│
├── adapt_streamlit_upload.py          # Interface Streamlit (version LLaMA)
├── adapt_streamlit_upload_openai.py   # Interface Streamlit (version OpenAI)
├── Modelfile_openai.py                # Fonctions d'interaction avec OpenAI
├── requirements.txt                    # Dépendances Python
├── .streamlit/
│   └── secrets.toml                   # Configuration API (non versionné)
├── .gitignore
└── README.md
```

## 🤝 Contribuer

Les contributions sont les bienvenues ! Voici comment participer :

1. 🍴 **Fork** le projet
2. 🔧 **Créez** votre branche (`git checkout -b feature/amelioration`)
3. 💾 **Committez** vos changements (`git commit -m 'Ajout d'une fonctionnalité'`)
4. 📤 **Push** vers la branche (`git push origin feature/amelioration`)
5. 🎉 **Ouvrez** une Pull Request

### Rapporter un bug

Si vous trouvez un bug, ouvrez une [issue](https://github.com/Heleneb1/HelJob_AI/issues) en décrivant :

- Le comportement attendu
- Le comportement observé
- Les étapes pour reproduire le problème

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 👩‍💻 Auteur

**Hélène B.**

Développeuse web full-stack passionnée par l'IA et les technologies innovantes.

- 🌐 [Portfolio](https://heleneb.netlify.app/)
- 🐙 [GitHub](https://github.com/Heleneb1)

---

<div align="center">

Créé avec ❤️ par Hélène B.

Si ce projet vous a été utile, n'hésitez pas à lui donner une ⭐ !

</div>

---

# English Version

> Adapt your CV and generate personalized cover letters using artificial intelligence

## Demo Online

You can try the online application here: [HelJob_AI Demo](https://heljob.streamlit.app/)

## 📋 Table of Contents

- [About](#-about)
- [Features](#-features)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
  - [Local Version (LLaMA 3.2)](#local-version-llama-32---free)
  - [OpenAI Version (GPT-4)](#openai-version-gpt-4)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)
- [Author](#-author)

---

## 🎯 About

**HelJob_AI** is an innovative tool that uses artificial intelligence to help you in your job search. It analyzes your CV and a target job offer, then automatically generates:

- ✅ A CV adapted to the specific job requirements
- ✅ A personalized and relevant cover letter
- ✅ AI-based improvement suggestions

## ✨ Features

| Feature                         | Description                                            |
| ------------------------------- | ------------------------------------------------------ |
| 📄 **Smart Extraction**         | Analysis of PDF files (CVs) and TXT files (job offers) |
| 🎯 **CV Adaptation**            | Automatic content adjustment based on job requirements |
| ✍️ **Letter Generation**        | Creation of a unique and personalized cover letter     |
| 💬 **Conversational Assistant** | Ask questions about your CV (OpenAI version)           |
| 🖥️ **Intuitive Interface**      | Simplified user interface with Streamlit               |

## 🔧 Prerequisites

- **Python** 3.10 or newer
- **pip** (Python package manager)
- **Git** to clone the repository
- **OpenAI API Key** (only for OpenAI version)

---

## 📦 Installation

### Local Version (LLaMA 3.2) - Free

This version uses the open-source LLaMA 3.2 model and runs entirely locally.

#### 1. Clone the repository

```bash
git clone https://github.com/Heleneb1/HelJob_AI.git
cd HelJob_AI
```

#### 2. Create a virtual environment

**Linux/macOS:**

```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### 3. Install dependencies

```bash
pip install -r requirements.txt
```

#### 4. Run the application

```bash
streamlit run adapt_streamlit_upload.py
```

The application will automatically open in your browser at `http://localhost:8501`

---

### OpenAI Version (GPT-4)

This version uses the OpenAI API for optimized results and advanced conversational interaction.

#### 1. Clone and prepare the environment

```bash
git clone https://github.com/Heleneb1/HelJob_AI.git
cd HelJob_AI
python -m venv .venv
```

**Linux/macOS:**

```bash
source .venv/bin/activate
```

**Windows:**

```bash
.venv\Scripts\activate
```

#### 2. Install dependencies

```bash
pip install -r requirements.txt
```

#### 3. Configure OpenAI API key

Create the configuration file for your API key:

```bash
mkdir -p .streamlit
```

Create the `.streamlit/secrets.toml` file and add:

```toml
OPENAI_API_KEY = "your_api_key_here"
```

> ⚠️ **Important:** Never share your API key publicly. The `secrets.toml` file is automatically ignored by Git.

#### 4. Run the OpenAI application

```bash
streamlit run adapt_streamlit_upload_openai.py
```

---

## 🎮 Usage

1. **Open the application** in your browser
2. **Upload your CV** (PDF format)
3. **Paste the job offer** (text format)
4. **Click "Generate"** to get:
   - Your adapted CV
   - A personalized cover letter
5. **(OpenAI version only)** Use the chat to ask questions about your CV

## 📁 Project Structure

```
HelJob_AI/
│
├── adapt_streamlit_upload.py          # Streamlit interface (LLaMA version)
├── adapt_streamlit_upload_openai.py   # Streamlit interface (OpenAI version)
├── Modelfile_openai.py                # OpenAI interaction functions
├── requirements.txt                    # Python dependencies
├── .streamlit/
│   └── secrets.toml                   # API configuration (not versioned)
├── .gitignore
└── README.md
```

## 🤝 Contributing

Contributions are welcome! Here's how to participate:

1. 🍴 **Fork** the project
2. 🔧 **Create** your branch (`git checkout -b feature/improvement`)
3. 💾 **Commit** your changes (`git commit -m 'Add feature'`)
4. 📤 **Push** to the branch (`git push origin feature/improvement`)
5. 🎉 **Open** a Pull Request

### Report a bug

If you find a bug, open an [issue](https://github.com/Heleneb1/HelJob_AI/issues) describing:

- Expected behavior
- Observed behavior
- Steps to reproduce the problem

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## 👩‍💻 Author

**Hélène B.**

Full-stack web developer passionate about AI and innovative technologies.

- 🌐 [Portfolio](https://your-portfolio.com)
- 💼 [LinkedIn](https://linkedin.com/in/your-profile)
- 🐙 [GitHub](https://github.com/Heleneb1)

---

<div align="center">

Created with ❤️ by Hélène B.

If this project was useful to you, don't hesitate to give it a ⭐!

</div>
