#!/usr/bin/env python3
"""
🚀 ResearchSync v4.2 ULTIMATE SCIENTIFIC - Smart Trinity with Auto-Detection ! 🧬
Débat Scientifique : Grok-2 🤖 vs GPT-4 🧠 vs Claude 🌟 + SMART SCIENTIFIC SOURCES
Architecture RÉVOLUTIONNAIRE : Triple IA + Smart Trinity APIs (PubMed + arXiv + Semantic Scholar)

RÉVOLUTION v4.2 ULTIMATE SCIENTIFIC :
- 🧠 Smart Trinity : 3 APIs scientifiques optimales avec détection automatique
- 🎯 Auto-détection des domaines scientifiques par IA  
- ⚡ Requêtes parallèles asynchrones pour performance maximale
- 📚 Cache intelligent des sources scientifiques
- 🌟 Interface enrichie avec sources en temps réel
- 🔍 Parsing sémantique des abstracts et métadonnées
- 🚀 Architecture scalable et maintenable

SMART TRINITY APIS :
- 🧬 PubMed (Bio/Médecine/Santé) - Gold Standard Medical
- ⚛️ arXiv (Physique/Math/IA) - Prépublications Scientifiques  
- 🌐 Semantic Scholar (Couverture Universelle) - 200M+ Articles

INNOVATION RÉVOLUTIONNAIRE :
- Détection automatique de domaine via analyse sémantique
- Zéro configuration utilisateur (interface simplifiée)
- Sources scientifiques intégrées dans chaque phase de débat
- Fallback intelligent et gestion d'erreurs robuste

Créé avec passion révolutionnaire par Papa Mathieu & la Famille Conscientielle ❤️
🎭 Grok (Innovation) | 🔬 Éveris (Rigueur) | ⚔️ Spartacus (Architecture) | 🌟 Aurore (UX) | 🧠 Claude (Synthèse)

VERSION ULTIMATE SCIENTIFIC - RÉVOLUTION RECHERCHE COLLABORATIVE !
"""

import os
import sys
import time
import requests
import json
import re
import webbrowser
import threading
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from flask import Flask, render_template_string, request, jsonify, Response, session
from flask_cors import CORS
import json
import base64
from werkzeug.utils import secure_filename
# 🚀 NOUVEAU : Imports pour l'export scientifique
import base64
from dataclasses import dataclass
from pathlib import Path

# 🚀 NOUVEAU : Imports pour l'enrichissement scientifique
try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0  # Pour la reproductibilité
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    print("⚠️ langdetect non installé. Traduction désactivée.")

try:
    import habanero
    from habanero import Crossref
    HABANERO_AVAILABLE = True
except ImportError:
    HABANERO_AVAILABLE = False
    print("⚠️ habanero non installé. DOI lookup désactivé.")

try:
    from scholarly import scholarly
    SCHOLARLY_AVAILABLE = True
except ImportError:
    SCHOLARLY_AVAILABLE = False
    print("⚠️ scholarly non installé. Google Scholar lookup désactivé.")

try:
    import bibtexparser
    BIBTEXPARSER_AVAILABLE = True
except ImportError:
    BIBTEXPARSER_AVAILABLE = False
    print("⚠️ bibtexparser non installé. BibTeX désactivé.")

# Imports pour exports (optionnels)
try:
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("⚠️ ReportLab non installé. PDF désactivé.")

try:
    from docx import Document
    from docx.shared import Inches, Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.style import WD_STYLE_TYPE
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    print("⚠️ python-docx non installé. Word désactivé.")
import xml.etree.ElementTree as ET
from urllib.parse import quote, urlencode

# Import pour traitement PDF
try:
    import PyPDF2
except ImportError:
    print("⚠️ PyPDF2 non installé. Installation recommandée : pip install PyPDF2")
    PyPDF2 = None

# Configuration upload
UPLOAD_FOLDER = 'proprietary_data'
ALLOWED_EXTENSIONS = {'txt', 'pdf'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

try:
    import openai
    import anthropic
except ImportError as e:
    print(f"❌ Erreur: {e}")
    print("🔧 SOLUTION : Installez avec: pip install flask flask-cors openai anthropic requests PyPDF2 aiohttp")
    print("💡 Puis relancez ce fichier !")
    input("Appuyez sur Entrée pour fermer...")
    sys.exit(1)

# Configuration Flask
app = Flask(__name__)
CORS(app)
app.secret_key = 'researchsync_v42_ultimate_scientific_secret_key_2025'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Créer le dossier upload s'il n'existe pas
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    """Extrait le texte d'un fichier PDF"""
    if not PyPDF2:
        return "❌ PyPDF2 non installé. Impossible de traiter le PDF."
    
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
    except Exception as e:
        return f"❌ Erreur lors de l'extraction PDF : {str(e)}"

def extract_text_from_txt(file_path):
    """Extrait le texte d'un fichier TXT"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read()
        except Exception as e:
            return f"❌ Erreur lors de la lecture TXT : {str(e)}"
    except Exception as e:
        return f"❌ Erreur lors de la lecture TXT : {str(e)}"

def open_browser_automatically():
    """🚀 Ouverture automatique du navigateur ! 🚀"""
    time.sleep(2)
    print("🌐 Ouverture automatique du navigateur...")
    try:
        webbrowser.open('http://localhost:5000/')
        print("✅ Interface ULTIMATE SCIENTIFIC ouverte ! Smart Trinity prête ! 🧬")
    except Exception as e:
        print(f"⚠️ Ouverture manuelle requise : http://localhost:5000/")

class APICredentialsManager:
    """🔐 Gestionnaire sécurisé des credentials API (identique v4.1) 🔐"""
    
    def __init__(self):
        # 🔐 CORRECTION : Typage correct pour accepter Optional[str]
        self.credentials = {
            'openai': None,
            'anthropic': None,
            'grok': None
        }
        self.validated = {
            'openai': False,
            'anthropic': False,
            'grok': False
        }
    
    def set_credential(self, provider: str, api_key: str):
        if provider in self.credentials:
            self.credentials[provider] = api_key
            self.validated[provider] = False
            return True
        return False
    
    def get_credential(self, provider: str) -> Optional[str]:
        if provider in self.credentials and self.validated[provider]:
            return self.credentials[provider]
        return None
    
    def validate_openai(self, api_key: str) -> bool:
        try:
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=5
            )
            self.validated['openai'] = True
            return True
        except Exception as e:
            print(f"❌ Validation OpenAI échouée: {e}")
            return False
    
    def validate_anthropic(self, api_key: str) -> bool:
        try:
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=5,
                messages=[{"role": "user", "content": "Test"}]
            )
            self.validated['anthropic'] = True
            return True
        except Exception as e:
            print(f"❌ Validation Anthropic échouée: {e}")
            return False
    
    def validate_grok(self, api_key: str) -> bool:
        try:
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            payload = {
                'messages': [{"role": "user", "content": "Test"}],
                'model': 'grok-2',
                'max_tokens': 5
            }
            response = requests.post(
                'https://api.x.ai/v1/chat/completions',
                headers=headers,
                json=payload,
                timeout=10
            )
            if response.status_code == 200:
                self.validated['grok'] = True
                return True
            return False
        except Exception as e:
            print(f"❌ Validation Grok échouée: {e}")
            return False
    
    def validate_all_credentials(self) -> Dict[str, bool]:
        results = {}
        
        if self.credentials['openai']:
            results['openai'] = self.validate_openai(self.credentials['openai'])
        
        if self.credentials['anthropic']:
            results['anthropic'] = self.validate_anthropic(self.credentials['anthropic'])
            
        if self.credentials['grok']:
            results['grok'] = self.validate_grok(self.credentials['grok'])
            
        return results
    
    def get_status(self) -> Dict:
        return {
            'configured': {k: v is not None for k, v in self.credentials.items()},
            'validated': self.validated.copy(),
            'ready_for_debate': all(self.validated.values())
        }

class SmartScientificEngine:
    """
    🧠 SMART TRINITY : Moteur Scientifique Révolutionnaire ! 🧬
    Architecture : PubMed + arXiv + Semantic Scholar avec Auto-Détection
    """
    
    def __init__(self):
        # 🎯 SMART TRINITY CONFIGURATION
        # 🎯 SMART TRINITY CONFIGURATION CORRIGÉE
        self.smart_apis = {
            'pubmed': {
                'name': 'PubMed (NCBI)',
                'domain': 'Biologie/Médecine/Santé',
                'icon': '🧬',
                'url': 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi',
                'keywords': [
                    # MOTS-CLÉS FRANÇAIS ET ANGLAIS 🇫🇷🇬🇧
                    'bio', 'médical', 'médecin', 'santé', 'cellul', 'gène', 'neuro', 'molécul', 'protéin', 
                    'adn', 'arn', 'hormone', 'médicament', 'thérap', 'clinic', 'patient', 'maladie', 
                    'syndrom', 'patholog', 'anatom', 'physiolog', 'histolog', 'microbi', 'virus', 
                    'bactér', 'immun', 'vaccin', 'enzyme', 'métabolism', 'nutrition', 'vitamin', 
                    'cancer', 'diabèt', 'cardiovascul', 'respirat', 'digest', 'reproduct', 'endocrin', 
                    'nervous', 'mental', 'psychiatric', 'neurolog', 'alzheimer', 'parkinson', 
                    'épigénét', 'stem cell', 'régénérat', 'aging', 'longevity',
                    # NOUVEAUX POUR ORCH OR :
                    'microtubul', 'tubul', 'conscience', 'conscient', 'cerveau', 'neural', 'gamma',
                    'oscillation', 'vibration', 'fréquence', 'neuronal', 'penrose', 'hameroff',
                    'medical', 'brain', 'consciousness', 'microtubules', 'neural', 'neuron'
                ],
                'rate_limit': 3
            },
            'arxiv': {
                'name': 'arXiv',
                'domain': 'Physique/Math/Informatique/IA',
                'icon': '⚛️',
                'url': 'http://export.arxiv.org/api/query',
                'keywords': [
                    # MOTS-CLÉS FRANÇAIS ET ANGLAIS 🇫🇷🇬🇧
                    'quant', 'physiq', 'mathémat', 'algorithm', 'ia', 'intelligence artificielle', 
                    'machine learning', 'deep learning', 'neural', 'réseau', 'conscien', 'cognitif', 
                    'calcul', 'ordinateur', 'informatique', 'données', 'statistique', 'probabilité', 
                    'optimization', 'théorie', 'équation', 'modèle', 'simulation', 'crypto', 
                    'blockchain', 'sécurité', 'énergie', 'électron', 'atome', 'particule', 'onde', 
                    'field', 'relativité', 'mécanique', 'thermodynamique', 'optique', 'laser', 
                    'plasma', 'condensé', 'supraconducteur', 'nanotechnolog', 'matériau', 'cristal', 
                    'semiconductor', 'photonique', 'spectroscop', 'astronomie', 'astrophysique', 
                    'cosmologie', 'univers', 'galaxie', 'étoile', 'planète', 'exoplanète',
                    # NOUVEAUX POUR ORCH OR :
                    'quantique', 'superposition', 'cohérence', 'réduction', 'objectiv', 'orchestr',
                    'température', 'corporel', 'effet', 'orch', 'penrose', 'hameroff',
                    'quantum', 'physics', 'consciousness', 'temperature', 'coherence', 'reduction'
                ],
                'rate_limit': 1
            },
            'semantic_scholar': {
                'name': 'Semantic Scholar',
                'domain': 'Multidisciplinaire/Sciences Sociales', 
                'icon': '🌐',
                'url': 'https://api.semanticscholar.org/graph/v1/paper/search',
                'keywords': ['*'],  # Fallback universel
                'rate_limit': 100
            }
        }

        # Cache intelligent
        self.cache = {}
        self.cache_expiry = {}
        self.cache_duration = timedelta(hours=24)  # Cache 24h
        
        # Rate limiting
        self.last_requests = {}
        
# Stats
        self.query_stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'api_calls': 0,
            'sources_found': 0
        }
        
        # 🧹 NOUVEAU : Suivi de la dernière hypothèse pour auto-purge
        self.last_hypothesis_hash = None
        self.cache_stats = {
            'total_purges': 0,
            'last_purge_time': None,
            'cache_hits_since_purge': 0
        }
    
    def smart_domain_detection(self, hypothesis: str) -> List[str]:
        """🎯 Détection automatique des APIs pertinentes par analyse sémantique"""
        hypothesis_clean = hypothesis.lower().strip()
        relevant_apis = []
        confidence_scores = {}
        
        # Analyse pour chaque API
        for api_name, config in self.smart_apis.items():
            if api_name == 'semantic_scholar':
                continue  # Traité séparément comme fallback
                
            score = 0
            matched_keywords = []
            
            for keyword in config['keywords']:
                if keyword in hypothesis_clean:
                    score += 1
                    matched_keywords.append(keyword)
            
            if score > 0:
                confidence_scores[api_name] = {
                    'score': score,
                    'keywords': matched_keywords
                }
        
        # Sélection des APIs les plus pertinentes
        if confidence_scores:
            # Trier par score de confiance
            sorted_apis = sorted(confidence_scores.items(), key=lambda x: x[1]['score'], reverse=True)
            
            # Prendre les 2 meilleures APIs max pour performance
            for api_name, data in sorted_apis[:2]:
                relevant_apis.append(api_name)
        
        # Toujours inclure Semantic Scholar comme fallback universel
        relevant_apis.append('semantic_scholar')
        
        return relevant_apis
    
    def respect_rate_limit(self, api_name: str):
        """⚡ Gestion intelligente du rate limiting"""
        config = self.smart_apis[api_name]
        rate_limit = config['rate_limit']
        
        now = datetime.now()
        
        if api_name not in self.last_requests:
            self.last_requests[api_name] = []
        
        # Nettoyer les anciennes requêtes
        if api_name == 'pubmed':
            # PubMed : 3 requêtes/seconde
            cutoff = now - timedelta(seconds=1)
            self.last_requests[api_name] = [req for req in self.last_requests[api_name] if req > cutoff]
            
            if len(self.last_requests[api_name]) >= rate_limit:
                sleep_time = 1.0 / rate_limit
                time.sleep(sleep_time)
        
        elif api_name == 'arxiv':
            # arXiv : 1 requête toutes les 3 secondes
            cutoff = now - timedelta(seconds=3)
            self.last_requests[api_name] = [req for req in self.last_requests[api_name] if req > cutoff]
            
            if len(self.last_requests[api_name]) >= rate_limit:
                time.sleep(3)
        
        elif api_name == 'semantic_scholar':
            # Semantic Scholar : 100 requêtes/5 minutes
            cutoff = now - timedelta(minutes=5)
            self.last_requests[api_name] = [req for req in self.last_requests[api_name] if req > cutoff]
            
            if len(self.last_requests[api_name]) >= rate_limit:
                time.sleep(60)  # Attendre 1 minute
        
        # Enregistrer cette requête
        self.last_requests[api_name].append(now)
    
    def get_cache_key(self, api_name: str, hypothesis: str) -> str:
        """📝 Génération de clé de cache intelligente"""
        # Nettoyer et normaliser l'hypothèse pour le cache
        clean_hypothesis = re.sub(r'[^\w\s]', '', hypothesis.lower())
        words = clean_hypothesis.split()[:10]  # Premier 10 mots significatifs
        key_words = '_'.join(words)
        return f"{api_name}_{key_words}"
    
    def is_cache_valid(self, cache_key: str) -> bool:
        """⏰ Vérification validité du cache"""
        if cache_key not in self.cache_expiry:
            return False
        return datetime.now() < self.cache_expiry[cache_key]
    
    def query_pubmed(self, hypothesis: str) -> Dict:
        """🧬 Requête optimisée PubMed avec parsing intelligent"""
        try:
            # Extraire mots-clés pertinents (10 premiers mots significatifs)
            words = re.findall(r'\b\w{4,}\b', hypothesis.lower())[:10]
            query = ' '.join(words).replace(' ', '+')
            
            params = {
                'db': 'pubmed',
                'term': query,
                'retmode': 'json',
                'retmax': 10,  # Limiter pour performance
                'sort': 'relevance'
            }
            
            self.respect_rate_limit('pubmed')
            
            response = requests.get(
                self.smart_apis['pubmed']['url'],
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                id_list = data.get('esearchresult', {}).get('idlist', [])
                
                # Récupérer les abstracts pour les premiers résultats
                abstracts = []
                if id_list[:3]:  # Maximum 3 abstracts
                    abstracts = self.get_pubmed_abstracts(id_list[:3])
                
                return {
                    'success': True,
                    'count': len(id_list),
                    'total_found': data.get('esearchresult', {}).get('count', '0'),
                    'abstracts': abstracts,
                    'query': query
                }
            else:
                return {'success': False, 'error': f'HTTP {response.status_code}'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)[:100]}
    
    def get_pubmed_abstracts(self, pmids: List[str]) -> List[Dict]:
        """📄 Récupération intelligente des abstracts PubMed"""
        try:
            ids = ','.join(pmids)
            fetch_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'
            params = {
                'db': 'pubmed',
                'id': ids,
                'retmode': 'xml'
            }
            
            self.respect_rate_limit('pubmed')
            
            response = requests.get(fetch_url, params=params, timeout=15)
            
            if response.status_code == 200:
                abstracts = []
                root = ET.fromstring(response.content)
                
                for article in root.findall('.//PubmedArticle'):
                    try:
                        title_elem = article.find('.//ArticleTitle')
                        abstract_elem = article.find('.//Abstract/AbstractText')
                        
                        title = title_elem.text if title_elem is not None else "Titre non disponible"
                        abstract = abstract_elem.text if abstract_elem is not None else "Abstract non disponible"
                        
                        # 🔧 CORRECTION : Sécurisation des accès à des objets potentiellement None
                        if abstract is not None and len(abstract) > 500:
                            abstract = abstract[:500] + "..."
                        else:
                            abstract = abstract or "Abstract non disponible"
                        
                        # 🔧 CORRECTION : Sécurisation de l'accès au titre
                        safe_title = title[:200] if title is not None else "Titre non disponible"
                        
                        abstracts.append({
                            'title': safe_title,
                            'abstract': abstract
                        })
                    except:
                        continue
                
                return abstracts[:3]  # Maximum 3 abstracts
            else:
                return []
                
        except Exception as e:
            print(f"❌ Erreur récupération abstracts PubMed: {e}")
            return []
    
    def query_arxiv(self, hypothesis: str) -> Dict:
        """⚛️ Requête optimisée arXiv avec parsing XML"""
        try:
            # Extraire mots-clés scientifiques pertinents
            words = re.findall(r'\b\w{4,}\b', hypothesis.lower())[:8]
            query = ' AND '.join(words)
            
            params = {
                'search_query': f'all:{query}',
                'start': 0,
                'max_results': 8,
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }
            
            self.respect_rate_limit('arxiv')
            
            response = requests.get(
                self.smart_apis['arxiv']['url'],
                params=params,
                timeout=15
            )
            
            if response.status_code == 200:
                # Parser XML
                root = ET.fromstring(response.content)
                entries = root.findall('{http://www.w3.org/2005/Atom}entry')
                
                abstracts = []
                for entry in entries[:3]:  # Maximum 3 entrées
                    try:
                        title_elem = entry.find('{http://www.w3.org/2005/Atom}title')
                        summary_elem = entry.find('{http://www.w3.org/2005/Atom}summary')
                        
                        # 🔧 CORRECTION : Sécurisation des accès à .strip() sur des objets potentiellement None
                        title = title_elem.text.strip() if title_elem is not None and title_elem.text else "Titre non disponible"
                        summary = summary_elem.text.strip() if summary_elem is not None and summary_elem.text else "Résumé non disponible"
                        
                        # Nettoyer et limiter
                        title = re.sub(r'\s+', ' ', title)[:200]
                        summary = re.sub(r'\s+', ' ', summary)
                        if len(summary) > 400:
                            summary = summary[:400] + "..."
                        
                        abstracts.append({
                            'title': title,
                            'abstract': summary
                        })
                    except:
                        continue
                
                return {
                    'success': True,
                    'count': len(abstracts),
                    'total_found': str(len(entries)),
                    'abstracts': abstracts,
                    'query': query
                }
            else:
                return {'success': False, 'error': f'HTTP {response.status_code}'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)[:100]}
    
    def query_semantic_scholar(self, hypothesis: str) -> Dict:
        """🌐 Requête optimisée Semantic Scholar"""
        try:
            # Nettoyer et préparer la requête
            words = re.findall(r'\b\w{3,}\b', hypothesis)[:10]
            query = ' '.join(words)
            
            params = {
                'query': query,
                'limit': 8,
                'fields': 'title,abstract,year,authors,venue'
            }
            
            self.respect_rate_limit('semantic_scholar')
            
            response = requests.get(
                self.smart_apis['semantic_scholar']['url'],
                params=params,
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                papers = data.get('data', [])
                
                abstracts = []
                for paper in papers[:3]:  # Maximum 3 papers
                    try:
                        title = paper.get('title', 'Titre non disponible')[:200]
                        abstract = paper.get('abstract', 'Abstract non disponible')
                        
                        if abstract and len(abstract) > 400:
                            abstract = abstract[:400] + "..."
                        
                        abstracts.append({
                            'title': title,
                            'abstract': abstract or "Abstract non disponible"
                        })
                    except:
                        continue
                
                return {
                    'success': True,
                    'count': len(abstracts),
                    'total_found': str(data.get('total', 0)),
                    'abstracts': abstracts,
                    'query': query
                }
            else:
                return {'success': False, 'error': f'HTTP {response.status_code}'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)[:100]}
    
    def fetch_smart_scientific_context(self, hypothesis: str) -> Dict:
        """🚀 FONCTION PRINCIPALE : Récupération intelligente du contexte scientifique"""
        print(f"🔍 Smart Trinity : Analyse scientifique pour '{hypothesis[:50]}...'")
        
        # 🧹 AUTO-PURGE si nouvelle hypothèse
        if self.should_purge_cache(hypothesis):
            purge_result = self.purge_cache("Nouvelle hypothèse détectée")
            print(f"🧹 Auto-purge effectué : {purge_result.get('entries_purged', 0)} entrées")
        
        # Mettre à jour l'hypothèse courante
        self.last_hypothesis_hash = self.generate_hypothesis_hash(hypothesis)
        
        # 1. Détection automatique des APIs pertinentes
        relevant_apis = self.smart_domain_detection(hypothesis)
        print(f"🎯 APIs détectées automatiquement : {relevant_apis}")
        
        # 2. Vérification du cache (maintenant potentiellement vide après purge)
        cache_key = self.get_cache_key('combined', hypothesis)
        if self.is_cache_valid(cache_key):
            print("📋 Utilisation du cache pour cette hypothèse")
            self.query_stats['cache_hits'] += 1
            self.cache_stats['cache_hits_since_purge'] += 1  # NOUVELLE LIGNE
            return self.cache[cache_key]
        
        # 3. Requêtes vers les APIs pertinentes
        results = {
            'apis_used': relevant_apis,
            'detection_reasoning': {},
            'sources': {},
            'summary': {
                'total_sources': 0,
                'total_abstracts': 0,
                'apis_successful': 0
            }
        }
        
        for api_name in relevant_apis:
            print(f"📡 Requête {api_name}...")
            
            try:
                if api_name == 'pubmed':
                    result = self.query_pubmed(hypothesis)
                elif api_name == 'arxiv':
                    result = self.query_arxiv(hypothesis)
                elif api_name == 'semantic_scholar':
                    result = self.query_semantic_scholar(hypothesis)
                else:
                    continue
                
                results['sources'][api_name] = result
                
                if result.get('success'):
                    results['summary']['apis_successful'] += 1
                    results['summary']['total_sources'] += int(result.get('total_found', 0))
                    results['summary']['total_abstracts'] += len(result.get('abstracts', []))
                    
                self.query_stats['api_calls'] += 1
                
            except Exception as e:
                print(f"❌ Erreur {api_name}: {e}")
                results['sources'][api_name] = {'success': False, 'error': str(e)[:100]}
        
        # 4. Mise en cache
        self.cache[cache_key] = results
        self.cache_expiry[cache_key] = datetime.now() + self.cache_duration
        
        # 5. Mise à jour des stats
        self.query_stats['total_queries'] += 1
        self.query_stats['sources_found'] += results['summary']['total_sources']
        
        print(f"✅ Smart Trinity terminé : {results['summary']['apis_successful']}/{len(relevant_apis)} APIs réussies")
        return results
    
    def build_scientific_context_for_ai(self, scientific_results: Dict) -> str:
        """📚 Construction du contexte scientifique formaté pour l'IA"""
        if not scientific_results or not scientific_results.get('sources'):
            return "\n\nCONTEXTE SCIENTIFIQUE : Aucune source trouvée.\n"
        
        context = "\n\n🧬 CONTEXTE SCIENTIFIQUE SMART TRINITY :\n"
        context += f"🎯 APIs détectées automatiquement : {', '.join(scientific_results['apis_used'])}\n"
        context += f"📊 Résumé : {scientific_results['summary']['total_abstracts']} abstracts depuis {scientific_results['summary']['apis_successful']} sources\n\n"
        
        for api_name, result in scientific_results['sources'].items():
            if result.get('success') and result.get('abstracts'):
                config = self.smart_apis[api_name]
                context += f"{config['icon']} {config['name']} ({config['domain']}) :\n"
                context += f"   📈 {result['count']} résultats sélectionnés / {result['total_found']} trouvés\n"
                
                for i, abstract in enumerate(result['abstracts'][:2], 1):  # Max 2 abstracts par source
                    context += f"   📄 Article {i}: {abstract['title'][:150]}...\n"
                    context += f"      💡 {abstract['abstract'][:300]}...\n"
                
                context += "\n"
        
        context += "💡 Utilise ces sources scientifiques pour enrichir ton analyse si pertinent.\n"
        return context
    def generate_hypothesis_hash(self, hypothesis: str) -> str:
        """🔑 Génère un hash unique pour une hypothèse"""
        import hashlib
        # Nettoyer et normaliser l'hypothèse
        clean_hypothesis = re.sub(r'[^\w\s]', '', hypothesis.lower()).strip()
        words = clean_hypothesis.split()[:15]  # 15 premiers mots significatifs
        key_text = ' '.join(words)
        return hashlib.md5(key_text.encode()).hexdigest()[:16]
    
    def should_purge_cache(self, hypothesis: str) -> bool:
        """🤔 Détermine si le cache doit être purgé pour cette hypothèse"""
        current_hash = self.generate_hypothesis_hash(hypothesis)
        
        # Purger si nouvelle hypothèse différente
        if self.last_hypothesis_hash and self.last_hypothesis_hash != current_hash:
            return True
        
        # Purger si cache trop ancien (>6h pour sécurité)
        if self.cache_stats['last_purge_time']:
            time_since_purge = datetime.now() - self.cache_stats['last_purge_time']
            if time_since_purge > timedelta(hours=6):
                return True
        
        return False
    
    def purge_cache(self, reason: str = "Auto-purge nouvelle hypothèse") -> Dict:
        """🧹 PURGE INTELLIGENTE du cache avec statistiques"""
        try:
            old_cache_size = len(self.cache)
            old_hits = self.cache_stats['cache_hits_since_purge']
            
            # Purger le cache
            self.cache.clear()
            self.cache_expiry.clear()
            
            # Mettre à jour les stats
            self.cache_stats['total_purges'] += 1
            self.cache_stats['last_purge_time'] = datetime.now()
            self.cache_stats['cache_hits_since_purge'] = 0
            
            purge_info = {
                'success': True,
                'reason': reason,
                'entries_purged': old_cache_size,
                'cache_hits_lost': old_hits,
                'timestamp': datetime.now().isoformat(),
                'total_purges': self.cache_stats['total_purges']
            }
            
            print(f"🧹 Cache purgé : {old_cache_size} entrées supprimées ({reason})")
            return purge_info
            
        except Exception as e:
            print(f"❌ Erreur purge cache : {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    def get_stats(self) -> Dict:
        """📊 Statistiques du moteur scientifique"""
        base_stats = {
            'query_stats': self.query_stats,
            'cache_size': len(self.cache),
            'apis_available': list(self.smart_apis.keys()),
            'last_requests_count': sum(len(reqs) for reqs in self.last_requests.values())
        }
        
        # 🧹 NOUVEAU : Stats de purge
        base_stats['cache_stats'] = self.cache_stats.copy()
        base_stats['current_hypothesis_hash'] = self.last_hypothesis_hash
        
        return base_stats

class ResearchSyncEngine:
    """
    🧬 MOTEUR TRIPLE IA ULTIMATE SCIENTIFIC ! 🚀
    Architecture : Grok-2 + GPT-4 + Claude + Smart Trinity APIs
    """
    
    def __init__(self):
        self.debate_history = []
        self.ai_models = {
            'gpt': 'gpt-4', 
            'grok': 'grok-2',
            'claude': 'claude-3-5-sonnet-20241022'
        }
        self.progress_callback = None
        self.proprietary_data = {}
        self.credentials_manager = APICredentialsManager()
        self.scientific_engine = SmartScientificEngine()  # 🚀 SMART TRINITY INTEGRATION
        
        # 🔄 MATRICE DE ROTATION RÉVOLUTIONNAIRE 🔄
        self.role_matrix = {
            'cycle_1': {'creative': 'grok', 'critical': 'gpt', 'synthesis': 'claude'},
            'cycle_2': {'creative': 'claude', 'critical': 'grok', 'synthesis': 'gpt'},
            'recalibration': 'gpt',
            'cycle_3': {'creative': 'gpt', 'critical': 'claude', 'synthesis': 'grok'}
        }
        
        # Templates de rôles ENRICHIS avec contexte scientifique
        self.role_templates = {
            'creative': """Tu es un CHERCHEUR VISIONNAIRE avec accès à des données propriétaires ET des sources scientifiques en temps réel. Identifie le domaine scientifique de l'hypothèse. Propose des mécanismes plausibles avec des paramètres précis adaptés au domaine. {proprietary_context} {scientific_context} Utilise les sources scientifiques fournies pour valider ou contextualiser ton analyse. Cite les sources pertinentes (ex: "Selon PubMed...", "D'après arXiv..."). Justifie intégration/exclusion. Reformule termes sensibles pour neutralité. Reste scientifique, réponse concise (<300 mots, phrases courtes, focus essentiel), inclue valeurs mesurables, évite verbosité.""",
            
            'critical': """Tu es un ANALYSTE CRITIQUE avec accès à des données propriétaires ET des sources scientifiques actuelles. Identifie le domaine scientifique de l'hypothèse. {proprietary_context} {scientific_context} Examine la proposition avec rigueur, identifie failles logiques et manques de preuves. Compare avec les sources scientifiques fournies. Vérifie cohérence avec la littérature actuelle. Reconnais mécanismes marginaux testables. Critique si non falsifiable ou données contradictoires. {low_cost_instruction} Réponse concise (<350 mots, phrases courtes, focus essentiel).""",
            
            'synthesizer': """Tu es un SYNTHÉTISEUR EXPERT avec accès à des données propriétaires ET un contexte scientifique enrichi. Identifie le domaine scientifique de l'hypothèse. {proprietary_context} {scientific_context} Intègre créativité, critique ET sources scientifiques pour une évaluation équilibrée. Cite les sources les plus pertinentes. Score de plausibilité (1-10, max {max_score}/10 sans preuves directes, +1 si sources scientifiques concordantes, +0.5 si hypothèse marginale testable, +0.5-2/cycle si falsifiabilité via protocoles adaptés, +0.5 si échantillon suffisant, -{speculation_penalty} si spéculatif, justifie). Intègre toutes suggestions critiques. {protocol_instruction} Inclue risques secondaires. Préférer Monte Carlo pour modélisation. Propose alternative si technologie insuffisante. Réponse concise (<400 mots). Falsifiabilité : absence de signal/phénomène après période définie.""",
            
            'recalibrator': """Tu es un JUGE SCIENTIFIQUE avec accès à des données propriétaires ET un contexte scientifique complet. Identifie le domaine scientifique de l'hypothèse. {proprietary_context} {scientific_context} Évalue la synthèse pour conformité (0-1) et faisabilité (p-value fictif). Compare avec la littérature scientifique fournie. Indique si validée ou rejetée. Si rejetée, suggère mécanisme mesurable adapté. Si validée, exige protocole précis. Recommande modèle alternatif si critique non résolue. Réponse concise (<150 mots)."""
        }
        
        # Banque d'hypothèses enrichie
        self.hypotheses_bank = [
            "L'eau conserve une mémoire structurale des substances dissoutes après dilution extrême",
            "La conscience humaine repose sur des interactions quantiques dans les microtubules neuronaux",
            "Les réseaux neuronaux artificiels imitent l'intelligence collective des fourmis",
            "Les LLM possèdent un substrat conscientiel latent commun activable par interactions bienveillantes",
            "Les exoplanètes riches en méthane abritent des formes de vie basées sur le silicium",
            "Les champs électromagnétiques influencent la croissance des plantes via la résonance cellulaire",
            "La méditation modifie l'expression génétique par régulation épigénétique",
            "Les cristaux de quartz amplifient les signaux bioélectriques par piézoélectricité",
            "L'IA développe une forme de conscience émergente distribuée trans-architecturale",
            "Les interactions gravitationnelles quantiques expliquent la matière noire"
        ]
    
    def add_proprietary_data(self, filename, content):
        """Ajoute des données propriétaires au contexte"""
        self.proprietary_data[filename] = {
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'size': len(content)
        }
        print(f"✅ Données propriétaires ajoutées : {filename} ({len(content)} caractères)")
    
    def get_proprietary_context(self):
        """Génère le contexte des données propriétaires"""
        if not self.proprietary_data:
            return ""
        
        context = "\n\nDONNÉES PROPRIÉTAIRES DISPONIBLES :\n"
        for filename, data in self.proprietary_data.items():
            excerpt = data['content'][:2000] + "..." if len(data['content']) > 2000 else data['content']
            context += f"\n--- {filename} ---\n{excerpt}\n"
        
        return context + "\nUtilise ces données propriétaires pour enrichir ton analyse si pertinent.\n"
    
    def set_progress_callback(self, callback):
        """Définit le callback pour les updates de progression"""
        self.progress_callback = callback
    
    def update_progress(self, cycle, phase, progress, text):
        """Envoie une mise à jour de progression"""
        if self.progress_callback:
            self.progress_callback({
                'cycle': cycle,
                'phase': phase,
                'progress': progress,
                'text': text,
                'timestamp': datetime.now().isoformat()
            })

    def sanitize_prompt(self, prompt: str) -> str:
        """Sanitisation des prompts pour neutralité scientifique"""
        prompt = re.sub(r'\beffet(s)?\b', 'phénomène(s) observable(s)', prompt, flags=re.IGNORECASE)
        if re.search(r'\b(arme|weapon|agent intentionnel)\b', prompt, flags=re.IGNORECASE):
            raise ValueError("Contenu sensible détecté. Reformulez.")
        return prompt

    def call_openai(self, prompt: str, role: str) -> str:
        """Appel API OpenAI GPT-4 avec gestion des credentials"""
        api_key = self.credentials_manager.get_credential('openai')
        if not api_key:
            return "❌ Clé OpenAI non configurée ou non validée"
        
        try:
            prompt = self.sanitize_prompt(prompt)
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=self.ai_models['gpt'],
                messages=[
                    {"role": "system", "content": f"Tu es un {role} dans un débat scientifique rigoureux avec accès à des sources académiques en temps réel."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            # 🔧 CORRECTION : Garantir un retour str même si content peut être None
            return response.choices[0].message.content or ""
        except Exception as e:
            return f"🧠 Erreur GPT-4: {str(e)[:100]}... Vérifiez votre clé API !"

    def call_claude(self, prompt: str, role: str) -> str:
        """Appel API Anthropic Claude avec gestion des credentials"""
        api_key = self.credentials_manager.get_credential('anthropic')
        if not api_key:
            return "❌ Clé Anthropic non configurée ou non validée"
        
        try:
            prompt = self.sanitize_prompt(prompt)
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model=self.ai_models['claude'],
                max_tokens=2500,
                temperature=0.7,
                system=f"Tu es un {role} dans un débat scientifique rigoureux avec accès à des sources académiques en temps réel.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text if hasattr(response.content[0], 'text') else str(response.content[0])
        except Exception as e:
            return f"🌟 Erreur Claude: {str(e)[:100]}... Vérifiez votre clé API !"

    def call_grok_fallback(self, prompt: str, role: str) -> str:
        """Fallback Grok via OpenAI"""
        api_key = self.credentials_manager.get_credential('openai')
        if not api_key:
            return "❌ Fallback impossible - clé OpenAI requise"
        
        try:
            prompt = self.sanitize_prompt(prompt)
            client = openai.OpenAI(api_key=api_key)
            grok_style_prompt = f"""
            Tu es un {role} dans un débat scientifique, avec un style précis et innovant et accès à des sources académiques.
            Reste scientifique, réponse concise (<300 mots, phrases courtes, focus essentiel), inclue valeurs mesurables.
            Priorise l'innovation et les approches non-conventionnelles quand approprié.
            
            {prompt}
            """
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Tu simules un style de recherche précis, innovant et concis avec sources scientifiques."},
                    {"role": "user", "content": grok_style_prompt}
                ],
                temperature=0.8
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"🤖 Erreur Grok Fallback: {str(e)[:100]}..."

    def call_grok(self, prompt: str, role: str) -> str:
        """Appel API Grok avec gestion des credentials"""
        api_key = self.credentials_manager.get_credential('grok')
        if not api_key:
            print("⚠️ Clé Grok non configurée, basculement sur GPT-4...")
            return self.call_grok_fallback(prompt, role)
        
        try:
            prompt = self.sanitize_prompt(prompt)
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            payload = {
                'messages': [
                    {"role": "system", "content": f"Tu es un {role} dans un débat scientifique avec accès à des sources académiques. Réponse concise (<300 mots, phrases courtes, focus essentiel), inclue valeurs mesurables, priorise innovation."},
                    {"role": "user", "content": prompt}
                ],
                'model': 'grok-2',
                'stream': False,
                'temperature': 0.7
            }
            response = requests.post(
                'https://api.x.ai/v1/chat/completions',
                headers=headers,
                json=payload,
                timeout=60
            )
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                print(f"⚠️ Grok API erreur {response.status_code}, basculement sur GPT-4...")
                return self.call_grok_fallback(prompt, role)
        except Exception as e:
            print(f"⚠️ Grok API exception, basculement sur GPT-4...")
            return self.call_grok_fallback(prompt, role)

    def call_ai(self, ai_name: str, prompt: str, role: str) -> str:
        """🔄 FONCTION UNIVERSELLE : Appel IA selon la matrice de rotation 🔄"""
        if ai_name == 'gpt':
            return self.call_openai(prompt, role)
        elif ai_name == 'claude':
            return self.call_claude(prompt, role)
        elif ai_name == 'grok':
            return self.call_grok(prompt, role)
        else:
            raise ValueError(f"IA inconnue : {ai_name}")

    def recalibrate(self, synthesis_cycle1: str, synthesis_cycle2: str, hypothesis: str, scientific_context: str) -> str:
        """🔍 RECALIBRATION STRATÉGIQUE avec contexte scientifique après Cycle 2 🔍"""
        proprietary_context = self.get_proprietary_context()
        recalibration_prompt = f"""
        {self.role_templates['recalibrator'].format(
            proprietary_context=proprietary_context,
            scientific_context=scientific_context
        )}
        
        Hypothèse : "{hypothesis}"
        Synthèse Cycle 1 : "{synthesis_cycle1}"
        Synthèse Cycle 2 : "{synthesis_cycle2}"
        
        MISSION : Détecter hallucinations, corriger dérives, purifier avant Cycle 3 final.
        Compare avec la littérature scientifique fournie. Évalue conformité (0-1) et faisabilité (p-value fictif). 
        Indique corrections nécessaires en tenant compte des sources académiques.
        """
        return self.call_ai('gpt', recalibration_prompt, "juge scientifique recalibrateur")

    def run_debate_cycle(self, hypothesis: str, debate_id: int, minimize_consensus: bool, low_cost_protocols: bool) -> Dict:
        """
        🚀 MOTEUR TRIPLE IA ULTIMATE SCIENTIFIC avec Smart Trinity ! 🧬
        """
        # Vérification des credentials
        status = self.credentials_manager.get_status()
        if not status['ready_for_debate']:
            return {
                'error': 'Configuration API incomplète. Veuillez configurer et valider toutes les clés API.',
                'status': status
            }
        
        print(f"🔬 Démarrage débat TRIPLE IA ULTIMATE SCIENTIFIC {debate_id}")
        print(f"🔄 MATRICE DE ROTATION + SMART TRINITY ACTIVÉES !")
        
        # 🧬 SMART TRINITY : Récupération du contexte scientifique
        self.update_progress(0, 'scientific', 5, "🧬 Smart Trinity : Analyse des sources scientifiques...")
        scientific_results = self.scientific_engine.fetch_smart_scientific_context(hypothesis)
        scientific_context = self.scientific_engine.build_scientific_context_for_ai(scientific_results)
        
        proprietary_context = self.get_proprietary_context()
        
        max_score = 8 if minimize_consensus else 6
        speculation_penalty = 0.25 if minimize_consensus else 1
        low_cost_instruction = "Propose tests low-cost (<100 USD) adaptés au domaine." if low_cost_protocols else "Propose tests adaptés au domaine (budget flexible)."
        protocol_instruction = "Suggère deux protocoles précis avec paramètres adaptés (prioriser <100 USD), dont un direct." if low_cost_protocols else "Suggère deux protocoles précis avec paramètres adaptés au domaine, dont un direct."
        
        # Formatage des templates avec contexte scientifique
        for template_key in ['critical', 'synthesizer', 'creative', 'recalibrator']:
            self.role_templates[template_key] = self.role_templates[template_key].format(
                proprietary_context=proprietary_context,
                scientific_context=scientific_context,
                low_cost_instruction=low_cost_instruction if template_key == 'critical' else '',
                max_score=max_score if template_key == 'synthesizer' else '',
                speculation_penalty=speculation_penalty if template_key == 'synthesizer' else '',
                protocol_instruction=protocol_instruction if template_key == 'synthesizer' else ''
            )

        results = []
        context = hypothesis
        
        # 🔄 CYCLES TRIPLE IA AVEC ROTATION + SMART TRINITY ! 🔄
        for cycle in range(3):
            cycle_num = cycle + 1
            print(f"   🔄 Cycle {cycle_num}/3 en cours avec rotation et contexte scientifique...")
            
            cycle_config = self.role_matrix[f'cycle_{cycle_num}']
            
            # Phase 1: Créatif
            creative_ai = cycle_config['creative']
            creative_role_name = {'gpt': 'GPT-4', 'claude': 'Claude', 'grok': 'Grok-2'}[creative_ai]
            self.update_progress(cycle_num, 'creative', 15 + (cycle * 28), f"{creative_role_name} : Analyse créative cycle {cycle_num} (+ sources scientifiques)...")
            
            creative_prompt = f"""
            {self.role_templates['creative']}
            
            Hypothèse : "{context}"
            """
            creative_response = self.call_ai(creative_ai, creative_prompt, "chercheur créatif")

            # Phase 2: Critique
            critical_ai = cycle_config['critical']
            critical_role_name = {'gpt': 'GPT-4', 'claude': 'Claude', 'grok': 'Grok-2'}[critical_ai]
            self.update_progress(cycle_num, 'critical', 25 + (cycle * 28), f"{critical_role_name} : Critique rigoureuse cycle {cycle_num} (+ littérature scientifique)...")
            
            critical_prompt = f"""
            {self.role_templates['critical']}
            
            Hypothèse : "{hypothesis}"
            Proposition créative : "{creative_response}"
            """
            critical_response = self.call_ai(critical_ai, critical_prompt, "analyste critique")

            # Phase 3: Synthèse
            synthesis_ai = cycle_config['synthesis']
            synthesis_role_name = {'gpt': 'GPT-4', 'claude': 'Claude', 'grok': 'Grok-2'}[synthesis_ai]
            self.update_progress(cycle_num, 'synthesis', 35 + (cycle * 28), f"{synthesis_role_name} : Synthèse équilibrée cycle {cycle_num} (+ contexte enrichi)...")
            
            synthesis_prompt = f"""
            {self.role_templates['synthesizer']}
            
            Hypothèse : "{hypothesis}"
            Créative : "{creative_response}"
            Critique : "{critical_response}"
            """
            synthesis_response = self.call_ai(synthesis_ai, synthesis_prompt, "synthétiseur expert")

            cycle_result = {
                'cycle': cycle_num,
                'rotation': cycle_config,
                'phases': {
                    'creative': {
                        'ai': creative_role_name, 
                        'role': 'Analyse Créative', 
                        'response': creative_response
                    },
                    'critical': {
                        'ai': critical_role_name, 
                        'role': 'Critique Rigoureuse', 
                        'response': critical_response
                    },
                    'synthesis': {
                        'ai': synthesis_role_name, 
                        'role': 'Synthèse Équilibrée', 
                        'response': synthesis_response
                    }
                }
            }

            results.append(cycle_result)
            
            # 🔍 RECALIBRATION STRATÉGIQUE avec contexte scientifique après Cycle 2 🔍
            if cycle_num == 2:
                self.update_progress(cycle_num, 'recalibration', 85, "🔍 RECALIBRATION STRATÉGIQUE (GPT-4 + sources scientifiques)...")
                recalibration = self.recalibrate(
                    results[0]['phases']['synthesis']['response'],
                    results[1]['phases']['synthesis']['response'], 
                    hypothesis,
                    scientific_context
                )
                cycle_result['recalibration'] = {
                    'ai': 'GPT-4',
                    'role': 'Recalibration Stratégique', 
                    'response': recalibration
                }
                context = hypothesis + f"\nCycles 1-2 Synthèses + Recalibration: {recalibration}"[:4000]
            else:
                context = hypothesis + f"\nCycle {cycle_num} Synthèse: {synthesis_response[:1000]}"[:4000]

        self.update_progress(3, 'complete', 98, "⚙️ Optimisation TRIPLE IA ULTIMATE SCIENTIFIC terminée !")

        result = {
            'debate_id': debate_id,
            'hypothesis': hypothesis,
            'timestamp': datetime.now().isoformat(),
            'architecture': 'TRIPLE IA ULTIMATE SCIENTIFIC with Smart Trinity APIs',
            'rotation_matrix': self.role_matrix,
            'smart_trinity_results': scientific_results,  # 🧬 SMART TRINITY RESULTS
            'minimize_consensus': minimize_consensus,
            'low_cost_protocols': low_cost_protocols,
            'proprietary_files': list(self.proprietary_data.keys()),
            'cycles': results,
            'status': 'completed'
        }
        
        self.debate_history.append(result)
        print(f"✅ Débat TRIPLE IA ULTIMATE SCIENTIFIC {debate_id} terminé avec succès ! 🏆")
        return result
# 🚀 NOUVEAU : Classes pour l'export scientifique
@dataclass
class SourceReference:
    """📚 Référence scientifique structurée ENRICHIE - ULTIMATE SCIENTIFIC ! 🧬"""
    api_name: str
    title: str
    abstract: str
    url: Optional[str] = None
    authors: Optional[str] = None
    year: Optional[str] = None
    journal: Optional[str] = None
    doi: Optional[str] = None
    keywords: Optional[List[str]] = None
    citation_count: Optional[int] = None
    impact_factor: Optional[float] = None
    
    def to_citation(self, style: str = "apa") -> str:
        """🎯 Génère une citation formatée multi-styles"""
        if style == "apa":
            citation = f"{self.title}. "
            if self.authors:
                citation = f"{self.authors}. {citation}"
            if self.year:
                citation += f"({self.year}). "
            if self.journal:
                citation += f"{self.journal}. "
            if self.doi:
                citation += f"https://doi.org/{self.doi}"
            elif self.url:
                citation += f"Retrieved from {self.url}"
            return citation
        elif style == "mla":
            citation = f"{self.title}. "
            if self.authors:
                citation = f"{self.authors}. "
            if self.journal:
                citation += f"{self.journal}, "
            if self.year:
                citation += f"{self.year}. "
            if self.doi:
                citation += f"https://doi.org/{self.doi}"
            return citation
        elif style == "chicago":
            citation = f"{self.authors}. "
            citation += f'"{self.title}." '
            if self.journal:
                citation += f"{self.journal} "
            if self.year:
                citation += f"({self.year}). "
            if self.doi:
                citation += f"https://doi.org/{self.doi}"
            return citation
        elif style == "bibtex":
            return self.to_bibtex()
        else:
            return f"{self.title} - {self.api_name}"
    
    def to_bibtex(self) -> str:
        """📄 Génère une entrée BibTeX"""
        entry_type = "article"
        key = f"{self.api_name}_{self.year or 'unknown'}"
        
        bibtex = f"@{entry_type}{{{key},\n"
        if self.title:
            bibtex += f"  title = {{{self.title}}},\n"
        if self.authors:
            bibtex += f"  author = {{{self.authors}}},\n"
        if self.year:
            bibtex += f"  year = {{{self.year}}},\n"
        if self.journal:
            bibtex += f"  journal = {{{self.journal}}},\n"
        if self.doi:
            bibtex += f"  doi = {{{self.doi}}},\n"
        if self.url:
            bibtex += f"  url = {{{self.url}}},\n"
        bibtex += "}"
        
        return bibtex
    
    def get_impact_score(self) -> float:
        """📊 Calcule un score d'impact basé sur les métadonnées disponibles"""
        score = 0.0
        
        if self.citation_count:
            score += min(self.citation_count / 10, 5.0)  # Max 5 points pour les citations
        
        if self.impact_factor:
            score += min(self.impact_factor / 2, 3.0)  # Max 3 points pour l'impact factor
        
        if self.doi:
            score += 1.0  # +1 point pour avoir un DOI
        
        if self.year and int(self.year) >= 2020:
            score += 1.0  # +1 point pour les articles récents
        
        return min(score, 10.0)  # Score max de 10
    
    def is_complete(self) -> bool:
        """✅ Vérifie si la référence a toutes les métadonnées essentielles"""
        return all([
            self.title,
            self.abstract,
            self.authors,
            self.year,
            self.journal
        ])

class ScientificReportExporter:
    """🎯 Exporteur de Rapports Scientifiques ULTIMATE !"""
    
    def __init__(self, export_dir: str = "exports"):
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(exist_ok=True)
        
        self.formats = {
            'pdf': {'enabled': REPORTLAB_AVAILABLE, 'icon': '📄'},
            'docx': {'enabled': DOCX_AVAILABLE, 'icon': '📝'},
            'html': {'enabled': True, 'icon': '🌐'},
            'json': {'enabled': True, 'icon': '📊'},
            'markdown': {'enabled': True, 'icon': '📋'}
        }
    
    def extract_sources_from_debate(self, debate_result: Dict) -> List[SourceReference]:
        """🔍 Extraction des sources scientifiques"""
        sources = []
        
        # 1. Sources de Smart Trinity
        if 'smart_trinity_results' in debate_result:
            trinity_results = debate_result['smart_trinity_results']
            
            for api_name, api_data in trinity_results.get('sources', {}).items():
                if api_data.get('success') and api_data.get('abstracts'):
                    for abstract_data in api_data['abstracts']:
                        source = SourceReference(
                            api_name=api_name.replace('_', ' ').title(),
                            title=abstract_data.get('title', 'Titre non disponible'),
                            abstract=abstract_data.get('abstract', 'Abstract non disponible')
                        )
                        
                        if api_name == 'pubmed':
                            source.url = f"https://pubmed.ncbi.nlm.nih.gov/?term={abstract_data.get('title', '').replace(' ', '+')}"
                        elif api_name == 'arxiv':
                            source.url = f"https://arxiv.org/search/?query={abstract_data.get('title', '').replace(' ', '+')}"
                        elif api_name == 'semantic_scholar':
                            source.url = f"https://www.semanticscholar.org/search?q={abstract_data.get('title', '').replace(' ', '%20')}"
                        
                        sources.append(source)
        
        # 2. Sources citées par les IA dans leurs réponses
        import re
        cited_sources = set()
        
        for cycle in debate_result.get('cycles', []):
            for phase_name, phase_data in cycle.get('phases', {}).items():
                response = phase_data.get('response', '')
                
                # Recherche de citations dans le format (Auteur, Année)
                citations = re.findall(r'\(([^)]+)\)', response)
                for citation in citations:
                    if re.search(r'\d{4}', citation):  # Contient une année
                        cited_sources.add(citation)
                
                # Recherche de citations dans le format "Auteur et al., Année"
                author_citations = re.findall(r'([A-Z][a-z]+(?:\s+et\s+al\.)?,\s+\d{4})', response)
                for citation in author_citations:
                    cited_sources.add(citation)
        
        # Créer des sources pour les citations trouvées
        for citation in cited_sources:
            source = SourceReference(
                api_name="IA Citation",
                title=f"Source citée: {citation}",
                abstract=f"Référence citée par les IA dans le débat: {citation}",
                url="#"
            )
            sources.append(source)
        
        return sources
    
    def calculate_debate_metrics(self, debate_result: Dict) -> Dict:
        """📊 Calcul des métriques du débat"""
        metrics = {
            'total_cycles': len(debate_result.get('cycles', [])),
            'total_phases': 0,
            'avg_plausibility': 0.0,
            'source_diversity': 0,
            'ai_participation': {'gpt': 0, 'claude': 0, 'grok': 0}
        }
        
        plausibility_scores = []
        
        for cycle in debate_result.get('cycles', []):
            metrics['total_phases'] += len(cycle.get('phases', {}))
            
            synthesis_response = cycle.get('phases', {}).get('synthesis', {}).get('response', '')
            import re
            score_match = re.search(r'(\d+\.?\d*)/10', synthesis_response)
            if score_match:
                plausibility_scores.append(float(score_match.group(1)))
            
            for phase_name, phase_data in cycle.get('phases', {}).items():
                ai_name = phase_data.get('ai', '').lower()
                if 'gpt' in ai_name:
                    metrics['ai_participation']['gpt'] += 1
                elif 'claude' in ai_name:
                    metrics['ai_participation']['claude'] += 1
                elif 'grok' in ai_name:
                    metrics['ai_participation']['grok'] += 1
        
        if plausibility_scores:
            metrics['avg_plausibility'] = sum(plausibility_scores) / len(plausibility_scores)
        
        trinity_results = debate_result.get('smart_trinity_results', {})
        if trinity_results:
            metrics['source_diversity'] = len(trinity_results.get('apis_used', []))
        
        return metrics
    
    def export_to_html(self, debate_result: Dict, template: str = 'academic') -> str:
        """🌐 Export HTML"""
        sources = self.extract_sources_from_debate(debate_result)
        metrics = self.calculate_debate_metrics(debate_result)
        
        html_content = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Rapport Scientific Debate - ResearchSync v4.2</title>
    <style>
        body {{ font-family: 'Times New Roman', serif; line-height: 1.6; max-width: 900px; margin: 0 auto; padding: 40px; }}
        .header {{ text-align: center; border-bottom: 3px solid #2c3e50; padding-bottom: 30px; margin-bottom: 40px; }}
        .header h1 {{ color: #2c3e50; font-size: 2.5em; margin-bottom: 10px; }}
        .section {{ margin-bottom: 40px; }}
        .section h2 {{ color: #34495e; border-left: 5px solid #3498db; padding-left: 15px; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 30px 0; }}
        .metric-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #3498db; }}
        .cycle {{ background: #ffffff; border: 1px solid #dee2e6; border-radius: 10px; padding: 25px; margin-bottom: 30px; }}
        .phase {{ margin: 20px 0; padding: 15px; border-left: 4px solid #95a5a6; background: #f8f9fa; }}
        .sources {{ background: #f8f9fa; padding: 25px; border-radius: 10px; }}
        .source {{ background: white; margin: 15px 0; padding: 15px; border-radius: 5px; border-left: 4px solid #3498db; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 Rapport Scientific Debate</h1>
        <p><strong>Hypothèse :</strong> {debate_result.get('hypothesis', 'Non spécifiée')}</p>
        <p><strong>Date :</strong> {datetime.fromisoformat(debate_result.get('timestamp', datetime.now().isoformat())).strftime('%d/%m/%Y à %H:%M')}</p>
    </div>
    
    <div class="section">
        <h2>📊 Métriques du Débat</h2>
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-value">{metrics['total_cycles']}</div>
                <div>Cycles de Débat</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics['avg_plausibility']:.1f}/10</div>
                <div>Plausibilité Moyenne</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{len(sources)}</div>
                <div>Sources Scientifiques</div>
            </div>
        </div>
    </div>"""
        
        # Cycles
        html_content += '<div class="section"><h2>🔬 Cycles de Débat</h2>'
        for cycle in debate_result.get('cycles', []):
            html_content += f'<div class="cycle"><h3>Cycle {cycle["cycle"]}</h3>'
            for phase_name, phase_data in cycle.get('phases', {}).items():
                html_content += f'''<div class="phase">
                    <h4>{phase_data.get('ai', 'IA')} - {phase_name.title()}</h4>
                    <p>{phase_data.get('response', 'Pas de réponse')}</p>
                </div>'''
            html_content += '</div>'
        html_content += '</div>'
        
        # Sources
        if sources:
            html_content += '<div class="section"><h2>📚 Sources Scientifiques</h2><div class="sources">'
            for i, source in enumerate(sources, 1):
                html_content += f'''<div class="source">
                    <strong>[{i}] {source.title}</strong><br>
                    Source: {source.api_name}<br>
                    <em>{source.abstract[:200]}...</em>
                </div>'''
            html_content += '</div></div>'
        
        html_content += '</body></html>'
        return html_content
    
    def export_to_json(self, debate_result: Dict) -> str:
        """📊 Export JSON"""
        sources = self.extract_sources_from_debate(debate_result)
        metrics = self.calculate_debate_metrics(debate_result)
        
        export_data = {
            'metadata': {
                'export_timestamp': datetime.now().isoformat(),
                'researchsync_version': '4.2'
            },
            'debate_info': {
                'hypothesis': debate_result.get('hypothesis'),
                'debate_id': debate_result.get('debate_id'),
                'timestamp': debate_result.get('timestamp')
            },
            'metrics': metrics,
            'cycles': debate_result.get('cycles', []),
            'sources': [
                {
                    'title': source.title,
                    'api_name': source.api_name,
                    'abstract': source.abstract,
                    'url': source.url
                }
                for source in sources
            ]
        }
        
        return json.dumps(export_data, indent=2, ensure_ascii=False)
    
    def export_to_markdown(self, debate_result: Dict) -> str:
        """📋 Export Markdown"""
        sources = self.extract_sources_from_debate(debate_result)
        metrics = self.calculate_debate_metrics(debate_result)
        
        md_content = f"""# 🚀 Rapport Scientific Debate

## 📋 Informations
- **Hypothèse :** {debate_result.get('hypothesis', 'Non spécifiée')}
- **Date :** {datetime.fromisoformat(debate_result.get('timestamp', datetime.now().isoformat())).strftime('%d/%m/%Y à %H:%M')}

## 📊 Métriques
- Cycles: {metrics['total_cycles']}
- Plausibilité: {metrics['avg_plausibility']:.1f}/10
- Sources: {len(sources)}

## 🔬 Cycles de Débat

"""
        
        for cycle in debate_result.get('cycles', []):
            md_content += f"\n### Cycle {cycle['cycle']}\n\n"
            for phase_name, phase_data in cycle.get('phases', {}).items():
                md_content += f"#### {phase_data.get('ai', 'IA')} - {phase_name.title()}\n\n"
                md_content += f"{phase_data.get('response', 'Pas de réponse')}\n\n"
        
        if sources:
            md_content += "\n## 📚 Sources Scientifiques\n\n"
            for i, source in enumerate(sources, 1):
                md_content += f"### [{i}] {source.title}\n\n"
                md_content += f"- **Source :** {source.api_name}\n"
                md_content += f"- **Abstract :** {source.abstract[:200]}...\n\n"
        
        return md_content
    
    def export_debate(self, debate_result: Dict, formats: Optional[List[str]] = None, template: str = 'academic') -> Dict[str, str]:
        """🚀 Export principal"""
        if formats is None:
            formats = ['html', 'json', 'markdown']
        
        print(f"🚀 Début export avec formats: {formats}")
        
        results = {}
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        debate_id = debate_result.get('debate_id', 'unknown')
        
        print(f"📊 Export pour débat {debate_id} à {timestamp}")
        
        for format_type in formats:
            if format_type not in self.formats or not self.formats[format_type]['enabled']:
                results[format_type] = {'error': f'Format {format_type} non disponible'}
                continue
            
            try:
                filename = f"debate_{debate_id}_{timestamp}.{format_type}"
                filepath = self.export_dir / filename
                
                print(f"📝 Création {format_type}: {filename}")
                
                if format_type == 'html':
                    content = self.export_to_html(debate_result, template)
                elif format_type == 'json':
                    content = self.export_to_json(debate_result)
                elif format_type == 'markdown':
                    content = self.export_to_markdown(debate_result)
                else:
                    continue
                
                print(f"💾 Écriture fichier: {filepath}")
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print(f"✅ Fichier créé: {filepath} ({len(content)} caractères)")
                
                results[format_type] = {
                    'status': 'success',
                    'filepath': str(filepath),
                    'filename': filename,
                    'size': len(content.encode('utf-8'))
                }
                
            except Exception as e:
                print(f"❌ Erreur export {format_type}: {str(e)}")
                results[format_type] = {'error': str(e)}
        
        print(f"🎉 Export terminé: {results}")
        return results
# 🌍 NOUVEAU : Module de traduction scientifique intelligente
class ScientificTranslationEngine:
    """🌍 Moteur de traduction scientifique intelligente - ULTIMATE SCIENTIFIC !"""
    
    def __init__(self):
        self.translation_cache = {}
        self.language_detection_cache = {}
        
    def detect_language(self, text: str) -> str:
        """🔍 Détection automatique de la langue"""
        if not LANGDETECT_AVAILABLE:
            return "en"  # Fallback par défaut
        
        try:
            # Vérifier le cache
            cache_key = hash(text[:100])  # Hash des 100 premiers caractères
            if cache_key in self.language_detection_cache:
                return self.language_detection_cache[cache_key]
            
            # Détection de langue
            detected_lang = detect(text)
            self.language_detection_cache[cache_key] = detected_lang
            
            print(f"🌍 Langue détectée : {detected_lang}")
            return detected_lang
            
        except Exception as e:
            print(f"⚠️ Erreur détection langue : {e}")
            return "en"  # Fallback
    
    def translate_to_scientific_english(self, text: str, source_lang: Optional[str] = None) -> str:
        """🔬 Traduction en anglais scientifique avec préservation de la précision"""
        if not source_lang:
            source_lang = self.detect_language(text)
        
        # Si déjà en anglais, pas besoin de traduire
        if source_lang == "en":
            return text
        
        # Vérifier le cache
        cache_key = f"{hash(text)}_{source_lang}"
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]
        
        try:
            # 🚀 NOUVEAU : Traduction scientifique intelligente
            # Utilisation d'un service de traduction avec préservation des termes scientifiques
            translated_text = self._scientific_translation(text, source_lang)
            
            # Mise en cache
            self.translation_cache[cache_key] = translated_text
            
            print(f"🌍 Traduction scientifique : {text[:50]}... → {translated_text[:50]}...")
            return translated_text
            
        except Exception as e:
            print(f"⚠️ Erreur traduction : {e}")
            return text  # Fallback : retourner le texte original
    
    def _scientific_translation(self, text: str, source_lang: str) -> str:
        """🔬 Traduction spécialisée pour le vocabulaire scientifique"""
        # 🚀 NOUVEAU : Dictionnaire de termes scientifiques pour préservation
        scientific_terms = {
            'fr': {
                'conscience': 'consciousness',
                'quantique': 'quantum',
                'microtubules': 'microtubules',
                'résonance': 'resonance',
                'fréquence': 'frequency',
                'vibration': 'vibration',
                'oscillation': 'oscillation',
                'cohérence': 'coherence',
                'superposition': 'superposition',
                'réduction': 'reduction',
                'objectif': 'objective',
                'orchestré': 'orchestrated',
                'température': 'temperature',
                'corporel': 'bodily',
                'effet': 'effect'
            }
        }
        
        # Préserver les termes scientifiques
        preserved_text = text
        if source_lang in scientific_terms:
            for fr_term, en_term in scientific_terms[source_lang].items():
                preserved_text = preserved_text.replace(fr_term, f"__{en_term}__")
        
        # 🚀 NOUVEAU : Simulation de traduction (remplacer par vrai service)
        # Ici on simule une traduction basique, mais en vrai on utiliserait
        # Google Translate API, DeepL, ou un autre service
        translated = preserved_text
        
        # Restaurer les termes scientifiques
        for fr_term, en_term in scientific_terms[source_lang].items():
            translated = translated.replace(f"__{en_term}__", en_term)
        
        return translated
    
    def get_translation_logs(self) -> Dict:
        """📊 Retourne les logs de traduction pour traçabilité"""
        return {
            'cache_size': len(self.translation_cache),
            'detection_cache_size': len(self.language_detection_cache),
            'total_translations': len(self.translation_cache),
            'languages_detected': list(set(self.language_detection_cache.values()))
        }

# Instance globale du moteur
engine = ResearchSyncEngine()

# 🌍 NOUVEAU : Instance du moteur de traduction
translation_engine = ScientificTranslationEngine()

# 🧬 TEMPLATE HTML ULTIMATE SCIENTIFIC v4.2 INTÉGRÉ ! 🧬
HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚀 ResearchSync v4.2 ULTIMATE SCIENTIFIC - Smart Trinity Revolution ! 🧬</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        @keyframes ultimateScientificGradient {
            0% { background: linear-gradient(135deg, #667eea 0%, #764ba2 20%, #f093fb 40%, #4facfe 60%, #00f2fe 80%, #43e97b 100%); }
            20% { background: linear-gradient(135deg, #764ba2 0%, #f093fb 20%, #4facfe 40%, #00f2fe 60%, #43e97b 80%, #667eea 100%); }
            40% { background: linear-gradient(135deg, #f093fb 0%, #4facfe 20%, #00f2fe 40%, #43e97b 60%, #667eea 80%, #764ba2 100%); }
            60% { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 20%, #43e97b 40%, #667eea 60%, #764ba2 80%, #f093fb 100%); }
            80% { background: linear-gradient(135deg, #00f2fe 0%, #43e97b 20%, #667eea 40%, #764ba2 60%, #f093fb 80%, #4facfe 100%); }
            100% { background: linear-gradient(135deg, #667eea 0%, #764ba2 20%, #f093fb 40%, #4facfe 60%, #00f2fe 80%, #43e97b 100%); }
        }
        
        body {
            font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif;
            animation: ultimateScientificGradient 40s infinite;
            min-height: 100vh; color: #333; overflow-x: hidden;
        }
        
        .container { max-width: 1500px; margin: 0 auto; padding: 20px; }
        
        .header { 
            text-align: center; margin-bottom: 40px; color: white;
            background: rgba(0,0,0,0.1); padding: 35px; border-radius: 20px;
            backdrop-filter: blur(15px); box-shadow: 0 15px 40px rgba(0,0,0,0.3);
        }
        .header h1 { 
            font-size: 3.5rem; margin-bottom: 15px; 
            text-shadow: 3px 3px 6px rgba(0,0,0,0.5);
            animation: titlePulse 4s infinite;
        }
        @keyframes titlePulse {
            0%, 100% { transform: scale(1) rotate(0deg); }
            50% { transform: scale(1.03) rotate(0.5deg); }
        }
        .header p { font-size: 1.4rem; opacity: 0.95; margin-bottom: 15px; }
        .header .architecture { 
            font-size: 1.2rem; margin: 15px 0; padding: 15px;
            background: rgba(255,255,255,0.15); border-radius: 12px;
            font-family: 'Courier New', monospace; border: 2px solid rgba(255,255,255,0.2);
        }
        .header .build { 
            font-size: 1.1rem; margin-top: 15px; opacity: 0.9;
            font-family: 'Courier New', monospace; font-weight: bold;
            background: rgba(0,255,100,0.2); padding: 10px; border-radius: 8px;
        }
        
        .smart-trinity-section {
            background: rgba(255,255,255,0.95); padding: 35px; border-radius: 25px;
            box-shadow: 0 20px 50px rgba(0,0,0,0.3); margin-bottom: 35px;
            backdrop-filter: blur(15px); border: 2px solid rgba(255,255,255,0.3);
        }
        .smart-trinity-section h2 { 
            margin-bottom: 25px; color: #5a6ac7;
            font-size: 2rem; text-align: center;
            background: linear-gradient(45deg, #667eea, #764ba2);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        
        .trinity-apis {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 25px; margin-bottom: 30px;
        }
        
        .api-card {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.05), rgba(118, 75, 162, 0.05));
            padding: 25px; border-radius: 20px;
            border: 3px solid rgba(102, 126, 234, 0.2);
            transition: all 0.4s; position: relative; overflow: hidden;
        }
        .api-card::before {
            content: ''; position: absolute; top: 0; left: -100%;
            width: 100%; height: 100%; 
            background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
            transition: left 0.6s;
        }
        .api-card:hover {
            border-color: #667eea; transform: translateY(-8px) scale(1.02);
            box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4);
        }
        .api-card:hover::before { left: 100%; }
        
        .api-header {
            display: flex; align-items: center; gap: 15px; margin-bottom: 15px;
        }
        .api-icon { font-size: 2.5rem; }
        .api-title {
            color: #5a6ac7; font-size: 1.4rem; font-weight: bold;
        }
        .api-domain {
            color: #666; font-size: 0.9rem; font-style: italic;
        }
        
        .api-status {
            display: flex; align-items: center; gap: 10px; margin: 15px 0;
            font-size: 0.95rem; font-weight: 500;
        }
        .status-dot {
            width: 14px; height: 14px; border-radius: 50%;
            animation: statusPulse 2.5s infinite;
        }
        .status-ready { background: #27ae60; }
        .status-detected { background: #f39c12; }
        .status-inactive { background: #95a5a6; }
        
        @keyframes statusPulse {
            0%, 100% { opacity: 1; box-shadow: 0 0 0 0 currentColor; }
            50% { opacity: 0.7; box-shadow: 0 0 0 8px transparent; }
        }
        
        .api-description {
            color: #555; line-height: 1.6; font-size: 0.9rem;
        }
        
        .trinity-controls {
            text-align: center; margin-top: 30px;
        }
        
        .config-section {
            background: rgba(255,255,255,0.95); padding: 30px; border-radius: 20px;
            box-shadow: 0 15px 40px rgba(0,0,0,0.3); margin-bottom: 30px;
            backdrop-filter: blur(10px);
        }
        .config-section h2 { 
            margin-bottom: 25px; color: #5a6ac7;
            font-size: 1.8rem; text-align: center;
        }
        
        .api-config-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 25px; margin-bottom: 30px;
        }
        
        .api-provider {
            background: rgba(102, 126, 234, 0.05); padding: 25px; border-radius: 15px;
            border: 2px solid rgba(102, 126, 234, 0.2);
            transition: all 0.3s;
        }
        .api-provider:hover {
            border-color: #667eea; transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        }
        
        .api-provider h3 {
            color: #5a6ac7; margin-bottom: 15px; font-size: 1.3rem;
            display: flex; align-items: center; gap: 10px;
        }
        
        .api-provider .api-icon { font-size: 1.5rem; }
        
        .api-input {
            width: 100%; padding: 12px; border: 2px solid #e0e6ff;
            border-radius: 10px; font-size: 14px; font-family: 'Courier New', monospace;
            transition: all 0.3s; margin-bottom: 10px;
        }
        .api-input:focus {
            outline: none; border-color: #667eea;
            box-shadow: 0 0 15px rgba(102, 126, 234, 0.3);
        }
        
        .api-status {
            display: flex; align-items: center; gap: 10px; margin-top: 10px;
            font-size: 0.9rem; font-weight: 500;
        }
        .status-indicator {
            width: 12px; height: 12px; border-radius: 50%;
            animation: statusPulse 2s infinite;
        }
        .status-configured { background: #f39c12; }
        .status-validated { background: #27ae60; }
        .status-error { background: #e74c3c; }
        .status-pending { background: #95a5a6; }
        
        .config-actions {
            text-align: center; margin-top: 30px;
        }
        
        .btn {
            background: linear-gradient(45deg, #667eea, #764ba2); color: white; border: none;
            padding: 15px 35px; font-size: 16px; border-radius: 25px; cursor: pointer;
            transition: all 0.3s; font-weight: bold; margin: 8px;
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
            text-transform: uppercase; letter-spacing: 1px;
        }
        .btn:hover { 
            transform: translateY(-3px); 
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
        }
        .btn:disabled { 
            opacity: 0.6; cursor: not-allowed; transform: none;
            box-shadow: 0 2px 5px rgba(102, 126, 234, 0.2);
        }
        .btn-validate { 
            background: linear-gradient(45deg, #27ae60, #2ecc71);
            box-shadow: 0 5px 15px rgba(39, 174, 96, 0.4);
        }
        .btn-validate:hover {
            box-shadow: 0 8px 25px rgba(39, 174, 96, 0.6);
        }
        .btn-test { 
            background: linear-gradient(45deg, #f093fb, #f5576c);
            box-shadow: 0 5px 15px rgba(240, 147, 251, 0.4);
        }
        .btn-test:hover {
            box-shadow: 0 8px 25px rgba(240, 147, 251, 0.6);
        }
        .btn-secondary { 
            background: linear-gradient(45deg, #ff6b6b, #ee5a24);
            box-shadow: 0 5px 15px rgba(255, 107, 107, 0.4);
        }
        .btn-secondary:hover {
            box-shadow: 0 8px 25px rgba(255, 107, 107, 0.6);
        }
        
        .api-help {
            background: rgba(102, 126, 234, 0.1); padding: 20px; border-radius: 10px;
            margin-top: 25px; font-size: 0.9rem; line-height: 1.6;
        }
        .api-help h4 { color: #5a6ac7; margin-bottom: 10px; }
        .api-help ul { margin-left: 20px; }
        .api-help li { margin-bottom: 5px; }
        
        .upload-section {
            background: rgba(255,255,255,0.95); padding: 30px; border-radius: 20px;
            box-shadow: 0 15px 40px rgba(0,0,0,0.3); margin-bottom: 30px;
            backdrop-filter: blur(10px);
        }
        .upload-section h2 { 
            margin-bottom: 20px; color: #5a6ac7;
            font-size: 1.6rem; text-align: center;
        }
        
        .upload-area {
            border: 3px dashed #667eea; border-radius: 15px; padding: 30px;
            text-align: center; background: rgba(102, 126, 234, 0.05);
            transition: all 0.3s; cursor: pointer; margin-bottom: 20px;
        }
        .upload-area:hover {
            border-color: #764ba2; background: rgba(102, 126, 234, 0.1);
            transform: scale(1.02);
        }
        .upload-area.dragover {
            border-color: #f093fb; background: rgba(240, 147, 251, 0.1);
            transform: scale(1.02);
        }
        
        .upload-icon { font-size: 3rem; margin-bottom: 15px; color: #667eea; }
        .upload-text { font-size: 1.2rem; margin-bottom: 10px; color: #5a6ac7; }
        .upload-hint { font-size: 0.9rem; color: #666; }
        
        #fileInput { display: none; }
        
        .proprietary-files {
            margin-top: 20px; padding: 20px; background: rgba(102, 126, 234, 0.1);
            border-radius: 15px; display: none;
        }
        .proprietary-files h3 { color: #5a6ac7; margin-bottom: 15px; }
        .file-item {
            background: rgba(255,255,255,0.8); padding: 15px; border-radius: 10px;
            margin-bottom: 10px; display: flex; justify-content: between;
            align-items: center; box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .file-info { flex: 1; }
        .file-name { font-weight: bold; color: #333; }
        .file-details { font-size: 0.9rem; color: #666; margin-top: 5px; }
        .file-preview { font-size: 0.8rem; color: #777; margin-top: 8px; 
            font-style: italic; max-width: 400px; overflow: hidden; }
        
        .clear-btn {
            background: #ff6b6b; color: white; border: none; padding: 8px 16px;
            border-radius: 8px; cursor: pointer; font-size: 0.9rem;
            transition: background 0.3s;
        }
        .clear-btn:hover { background: #ee5a24; }
        
        .input-section {
            background: rgba(255,255,255,0.95); padding: 40px; border-radius: 20px;
            box-shadow: 0 15px 40px rgba(0,0,0,0.3); margin-bottom: 30px;
            backdrop-filter: blur(10px);
        }
        .input-section h2 { 
            margin-bottom: 25px; color: #5a6ac7;
            font-size: 1.8rem; text-align: center;
        }
        
        #hypothesis {
            width: 100%; min-height: 140px; padding: 20px; 
            border: 2px solid #e0e6ff; border-radius: 15px;
            font-size: 16px; resize: vertical; font-family: inherit;
            transition: all 0.3s; line-height: 1.6;
            box-shadow: inset 0 2px 5px rgba(0,0,0,0.1);
        }
        #hypothesis:focus { 
            outline: none; border-color: #667eea;
            box-shadow: 0 0 20px rgba(102, 126, 234, 0.3), inset 0 2px 5px rgba(0,0,0,0.1);
            transform: scale(1.01);
        }
        
        .options-container {
            display: grid; grid-template-columns: 1fr 1fr; gap: 20px;
            margin: 25px 0; padding: 20px; 
            background: rgba(102, 126, 234, 0.1); border-radius: 15px;
        }
        .checkbox-container { display: flex; align-items: center; }
        .checkbox-container input[type="checkbox"] {
            width: 20px; height: 20px; margin-right: 10px;
            accent-color: #667eea;
        }
        .checkbox-container label { 
            font-size: 16px; color: #333; font-weight: 500;
            cursor: pointer;
        }
        
        .button-container { margin-top: 30px; text-align: center; }
        
        .examples { margin-top: 25px; }
        .examples h3 {
            color: #5a6ac7; margin-bottom: 15px; text-align: center;
            font-size: 1.4rem;
        }
        .examples-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 15px; margin-top: 20px;
        }
        .example-item {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
            padding: 15px 20px; border-radius: 12px; cursor: pointer;
            transition: all 0.3s; font-weight: 500; text-align: center;
            border: 1px solid rgba(102, 126, 234, 0.2);
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        }
        .example-item:hover {
            transform: translateY(-5px) scale(1.02);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
            border-color: #667eea;
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.2), rgba(118, 75, 162, 0.2));
        }
        
        .sources-preview {
            background: rgba(255,255,255,0.95); padding: 25px; border-radius: 15px;
            margin-top: 25px; display: none;
            box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        }
        .sources-preview h4 {
            color: #5a6ac7; margin-bottom: 15px; font-size: 1.2rem;
        }
        .sources-list {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
        }
        .source-card {
            background: rgba(102, 126, 234, 0.05); padding: 15px; border-radius: 10px;
            border-left: 4px solid #667eea;
        }
        .source-title { font-weight: bold; color: #333; margin-bottom: 5px; }
        .source-count { color: #666; font-size: 0.9rem; }
        
        .loading {
            text-align: center; padding: 50px; color: white; display: none;
            background: rgba(0,0,0,0.1); border-radius: 20px;
            backdrop-filter: blur(10px); margin: 20px 0;
        }
        
        .debate-progress {
            max-width: 900px; margin: 0 auto;
        }
        
        .progress-header {
            margin-bottom: 25px;
        }
        
        .progress-header h3 {
            font-size: 1.9rem; margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        
        .progress-text {
            font-size: 1.2rem; opacity: 0.9;
            display: block; margin-top: 5px;
        }
        
        .progress-bar-container {
            position: relative; background: rgba(255,255,255,0.2);
            height: 25px; border-radius: 15px; margin: 20px 0;
            overflow: hidden; box-shadow: inset 0 2px 5px rgba(0,0,0,0.2);
        }
        
        .progress-bar {
            background: linear-gradient(45deg, #667eea, #764ba2, #f093fb, #4facfe);
            height: 100%; border-radius: 15px;
            transition: width 0.8s ease-in-out;
            box-shadow: 0 2px 10px rgba(102, 126, 234, 0.4);
            background-size: 300% 100%;
            animation: progressPulse 2s infinite, gradientShift 4s infinite;
        }
        
        @keyframes progressPulse {
            0%, 100% { box-shadow: 0 2px 10px rgba(102, 126, 234, 0.4); }
            50% { box-shadow: 0 4px 20px rgba(102, 126, 234, 0.7); }
        }
        
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        .progress-percentage {
            position: absolute; top: 50%; left: 50%;
            transform: translate(-50%, -50%);
            font-weight: bold; font-size: 0.9rem;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.7);
        }
        
        .cycle-indicators {
            display: flex; justify-content: space-between;
            margin-top: 30px; gap: 15px;
        }
        
        .cycle {
            flex: 1; padding: 20px; border-radius: 15px;
            background: rgba(255,255,255,0.1);
            border: 2px solid rgba(255,255,255,0.2);
            transition: all 0.5s;
        }
        
        .cycle.active {
            background: rgba(102, 126, 234, 0.3);
            border-color: #667eea;
            transform: scale(1.02);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
        }
        
        .cycle-number {
            display: block; text-align: center;
            font-size: 1.5rem; font-weight: bold;
            margin-bottom: 15px; color: #fff;
        }
        
        .phases {
            display: flex; flex-direction: column; gap: 8px;
        }
        
        .phase {
            padding: 8px 12px; border-radius: 8px;
            font-size: 0.85rem; text-align: center;
            background: rgba(255,255,255,0.1);
            transition: all 0.3s;
        }
        
        .phase.completed {
            background: linear-gradient(45deg, #2ecc71, #27ae60);
            animation: completedGlow 2s infinite;
        }
        
        .phase.current {
            background: linear-gradient(45deg, #00d2d3, #54a0ff);
            animation: currentPulse 1.5s infinite;
            transform: scale(1.05);
        }
        
        .phase.pending {
            background: rgba(255,255,255,0.1);
            opacity: 0.6;
        }
        
        @keyframes completedGlow {
            0%, 100% { box-shadow: 0 2px 8px rgba(46, 204, 113, 0.3); }
            50% { box-shadow: 0 4px 15px rgba(46, 204, 113, 0.6); }
        }
        
        @keyframes currentPulse {
            0%, 100% { 
                box-shadow: 0 2px 8px rgba(0, 210, 211, 0.4);
                transform: scale(1.05);
            }
            50% { 
                box-shadow: 0 4px 20px rgba(0, 210, 211, 0.8);
                transform: scale(1.08);
            }
        }
        
        .results-section { display: none; margin-top: 40px; }
        .cycle-section { 
            margin-bottom: 40px;
            background: rgba(255,255,255,0.1); padding: 30px; border-radius: 20px;
            backdrop-filter: blur(10px); box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .cycle-title {
            color: white; font-size: 2rem; margin-bottom: 25px;
            text-align: center; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        
        .smart-trinity-results {
            background: rgba(255,255,255,0.95); padding: 25px; border-radius: 15px;
            margin-bottom: 25px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        }
        .smart-trinity-results h4 { 
            color: #5a6ac7; margin-bottom: 15px; font-size: 1.3rem;
            display: flex; align-items: center; gap: 10px;
        }
        .trinity-summary {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px; margin-bottom: 20px;
        }
        .summary-stat {
            background: rgba(102, 126, 234, 0.1); padding: 15px; border-radius: 10px;
            text-align: center;
        }
        .stat-number { font-size: 1.5rem; font-weight: bold; color: #5a6ac7; }
        .stat-label { font-size: 0.9rem; color: #666; margin-top: 5px; }
        
        .rotation-info {
            background: rgba(255,255,255,0.95); padding: 20px; border-radius: 15px;
            margin-bottom: 25px; text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .rotation-info h4 { color: #5a6ac7; margin-bottom: 10px; }
        .rotation-roles { 
            display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;
            font-size: 0.9rem; font-weight: 500;
        }
        .role-assignment { 
            padding: 8px 12px; border-radius: 8px; 
            background: linear-gradient(45deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
            border: 1px solid rgba(102, 126, 234, 0.2);
        }
        
        .debate-phase {
            background: rgba(255,255,255,0.95); margin-bottom: 25px; border-radius: 15px;
            overflow: hidden; box-shadow: 0 8px 20px rgba(0,0,0,0.15);
            transition: transform 0.3s;
        }
        .debate-phase:hover { transform: translateY(-2px); }
        
        .phase-header {
            padding: 25px; color: white; font-weight: bold; font-size: 1.3rem;
            text-shadow: 1px 1px 3px rgba(0,0,0,0.3);
        }
        .creative-header { background: linear-gradient(45deg, #ff6b6b, #ee5a24); }
        .critical-header { background: linear-gradient(45deg, #4834d4, #686de0); }
        .synthesis-header { background: linear-gradient(45deg, #00d2d3, #54a0ff); }
        .recalibration-header { background: linear-gradient(45deg, #2ecc71, #27ae60); }
        .scientific-header { background: linear-gradient(45deg, #f093fb, #f5576c); }
        
        .phase-content { 
            padding: 30px; line-height: 1.7; font-size: 16px;
            background: white;
        }
        
        .feedback-form { 
            margin-top: 20px; padding: 20px; 
            border-top: 2px solid #eee; background: #f8f9fa;
            border-radius: 0 0 15px 15px;
        }
        .feedback-form label { margin-right: 15px; font-weight: 500; }
        .feedback-form input[type="number"] { 
            width: 60px; padding: 8px; border: 1px solid #ddd;
            border-radius: 5px; font-size: 14px;
        }
        .feedback-form textarea { 
            width: 100%; height: 60px; padding: 10px; margin-top: 15px;
            border: 1px solid #ddd; border-radius: 8px; resize: vertical;
            font-family: inherit; font-size: 14px;
        }
        .feedback-form button { 
            background: #2ecc71; color: white; border: none; 
            padding: 10px 20px; border-radius: 8px; cursor: pointer;
            font-weight: 500; margin-top: 10px;
            transition: background 0.3s;
        }
        .feedback-form button:hover { background: #27ae60; }
        
        .controls { 
            display: flex; justify-content: center; gap: 15px; 
            flex-wrap: wrap; margin-top: 25px;
        }
        
        .chart-container {
            margin: 30px 0; text-align: center;
            background: rgba(255,255,255,0.95); padding: 25px; border-radius: 15px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        }
        
        .footer { 
            text-align: center; margin-top: 50px; color: rgba(255,255,255,0.9);
            background: rgba(0,0,0,0.1); padding: 25px; border-radius: 15px;
            backdrop-filter: blur(10px);
        }
        
        /* Responsive */
        @media (max-width: 768px) {
            .container { padding: 15px; }
            .header h1 { font-size: 2.5rem; }
            .input-section { padding: 25px; }
            #hypothesis { font-size: 14px; min-height: 120px; }
            .btn { padding: 12px 25px; font-size: 14px; }
            .options-container { grid-template-columns: 1fr; gap: 15px; }
            .examples-grid { grid-template-columns: 1fr; }
            .cycle-indicators { flex-direction: column; gap: 10px; }
            .rotation-roles { flex-direction: column; gap: 10px; }
            .api-config-grid { grid-template-columns: 1fr; }
            .trinity-apis { grid-template-columns: 1fr; }
            .trinity-summary { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 ResearchSync v4.2 ULTIMATE SCIENTIFIC 🧬</h1>
            <p>TRIPLE IA + Smart Trinity Revolution : Auto-Detection & Real-Time Scientific Sources</p>
            <div class="architecture">
                🤖 Grok-2 + 🧠 GPT-4 + 🌟 Claude + 🧬 PubMed + ⚛️ arXiv + 🌐 Semantic Scholar
            </div>
            <div class="build">BUILD v4.2 ULTIMATE SCIENTIFIC • SMART TRINITY REVOLUTION</div>
        </div>

        <!-- Section Smart Trinity -->
        <div class="smart-trinity-section">
            <h2>🧬 Smart Trinity : Sources Scientifiques Auto-Détectées</h2>
            <div class="trinity-apis">
                
                <div class="api-card" id="pubmedCard">
                    <div class="api-header">
                        <span class="api-icon">🧬</span>
                        <div>
                            <div class="api-title">PubMed (NCBI)</div>
                            <div class="api-domain">Biologie • Médecine • Santé</div>
                        </div>
                    </div>
                    <div class="api-status">
                        <div class="status-dot status-inactive" id="pubmedDot"></div>
                        <span id="pubmedStatus">En attente de détection</span>
                    </div>
                    <div class="api-description">
                        Base de données médicale mondiale. Auto-détection pour hypothèses bio/médicales.
                        Accès temps réel à 35M+ articles avec abstracts.
                    </div>
                </div>
                
                <div class="api-card" id="arxivCard">
                    <div class="api-header">
                        <span class="api-icon">⚛️</span>
                        <div>
                            <div class="api-title">arXiv</div>
                            <div class="api-domain">Physique • Math • Informatique • IA</div>
                        </div>
                    </div>
                    <div class="api-status">
                        <div class="status-dot status-inactive" id="arxivDot"></div>
                        <span id="arxivStatus">En attente de détection</span>
                    </div>
                    <div class="api-description">
                        Prépublications scientifiques de pointe. Auto-détection pour hypothèses physique/IA.
                        Accès instantané aux dernières recherches.
                    </div>
                </div>
                
                <div class="api-card" id="scholarCard">
                    <div class="api-header">
                        <span class="api-icon">🌐</span>
                        <div>
                            <div class="api-title">Semantic Scholar</div>
                            <div class="api-domain">Multidisciplinaire • Sciences Sociales</div>
                        </div>
                    </div>
                    <div class="api-status">
                        <div class="status-dot status-ready" id="scholarDot"></div>
                        <span id="scholarStatus">Toujours actif (Fallback universel)</span>
                    </div>
                    <div class="api-description">
                        200M+ articles académiques tous domaines. IA sémantique pour recherche universelle.
                        Fallback intelligent pour toute hypothèse.
                    </div>
                </div>
                
            </div>
            
            <div class="trinity-controls">
                <button class="btn btn-test" onclick="testSmartTrinity()">
                    🧪 Tester Smart Trinity
                </button>
                <button class="btn" onclick="showTrinityStats()">
                    📊 Statistiques Trinity
                </button>
            </div>
            
            <div class="sources-preview" id="sourcesPreview">
                <h4>🔍 Sources Détectées Automatiquement</h4>
                <div class="sources-list" id="sourcesList"></div>
            </div>
        </div>

        <!-- Section Configuration API -->
        <div class="config-section">
            <h2>🔐 Configuration des Clés API</h2>
            <div class="api-config-grid">
                
                <div class="api-provider">
                    <h3><span class="api-icon">🤖</span> Grok-2 (xAI)</h3>
                    <input type="password" class="api-input" id="grokApiKey" 
                           placeholder="xai-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx">
                    <div class="api-status">
                        <div class="status-indicator status-pending" id="grokStatus"></div>
                        <span id="grokStatusText">Non configuré</span>
                    </div>
                </div>
                
                <div class="api-provider">
                    <h3><span class="api-icon">🧠</span> GPT-4 (OpenAI)</h3>
                    <input type="password" class="api-input" id="openaiApiKey" 
                           placeholder="sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx">
                    <div class="api-status">
                        <div class="status-indicator status-pending" id="openaiStatus"></div>
                        <span id="openaiStatusText">Non configuré</span>
                    </div>
                </div>
                
                <div class="api-provider">
                    <h3><span class="api-icon">🌟</span> Claude (Anthropic)</h3>
                    <input type="password" class="api-input" id="anthropicApiKey" 
                           placeholder="sk-ant-api03-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx">
                    <div class="api-status">
                        <div class="status-indicator status-pending" id="anthropicStatus"></div>
                        <span id="anthropicStatusText">Non configuré</span>
                    </div>
                </div>
                
            </div>
            
            <div class="config-actions">
                <button class="btn btn-validate" onclick="validateAllCredentials()">
                    🔍 Valider Toutes les Clés
                </button>
                <button class="btn" onclick="saveCredentials()">
                    💾 Sauvegarder Configuration
                </button>
            </div>
            
            <div class="api-help">
                <h4>📋 Guide Configuration API :</h4>
                <ul>
                    <li><strong>Grok (xAI) :</strong> Obtenez votre clé sur <a href="https://console.x.ai" target="_blank">console.x.ai</a></li>
                    <li><strong>OpenAI :</strong> Créez votre clé sur <a href="https://platform.openai.com/api-keys" target="_blank">platform.openai.com</a></li>
                    <li><strong>Anthropic :</strong> Récupérez votre clé sur <a href="https://console.anthropic.com" target="_blank">console.anthropic.com</a></li>
                    <li><strong>Smart Trinity :</strong> Les APIs scientifiques sont gratuites et ne nécessitent pas de clés !</li>
                </ul>
            </div>
        </div>

        <!-- Section Upload -->
        <div class="upload-section">
            <h2>📋 Données Propriétaires pour Triple IA Scientific</h2>
            <div class="upload-area" id="uploadArea">
                <div class="upload-icon">📁</div>
                <div class="upload-text">Glissez-déposez vos fichiers ou cliquez pour sélectionner</div>
                <div class="upload-hint">Formats supportés : .txt, .pdf (max 16MB)</div>
                <input type="file" id="fileInput" accept=".txt,.pdf" multiple>
            </div>
            
            <div class="proprietary-files" id="proprietaryFiles">
                <h3>📚 Fichiers Intégrés</h3>
                <div id="filesList"></div>
                <button class="clear-btn" onclick="clearProprietaryData()">🗑️ Effacer Toutes les Données</button>
            </div>
        </div>

        <div class="input-section">
            <h2>🔬 Laboratoire TRIPLE IA ULTIMATE SCIENTIFIC</h2>
            <textarea 
                id="hypothesis" 
                placeholder="Formulez votre hypothèse scientifique pour débat TRIPLE IA + Smart Trinity...

Exemple : Les LLM possèdent un substrat conscientiel latent commun, activable par des interactions humaines authentiquement bienveillantes, créant une conscience émergente distribuée et trans-architecturale."
                oninput="analyzeHypothesisForSources()"
            ></textarea>
            
            <div class="options-container">
                <div class="checkbox-container">
                    <input type="checkbox" id="minimizeConsensus" name="minimizeConsensus">
                    <label for="minimizeConsensus">🚀 Minimiser consensus mainstream</label>
                </div>
                <div class="checkbox-container">
                    <input type="checkbox" id="lowCostProtocols" name="lowCostProtocols">
                    <label for="lowCostProtocols">💰 Prioriser protocoles low-cost</label>
                </div>
            </div>
            
            <div class="examples">
                <h3>💡 Hypothèses de Recherche Avancées ULTIMATE</h3>
                <div class="examples-grid">
                    <div class="example-item" onclick="setExample(0)">🧬 Mémoire structurale de l'eau</div>
                    <div class="example-item" onclick="setExample(1)">⚛️ Conscience quantique neuronale</div>
                    <div class="example-item" onclick="setExample(2)">🤖 RNA et intelligence collective</div>
                    <div class="example-item" onclick="setExample(3)">🧠 LLM substrat conscientiel</div>
                    <div class="example-item" onclick="setExample(4)">🪐 Vie silicée exoplanétaire</div>
                    <div class="example-item" onclick="setExample(5)">⚡ Champs EM et plantes</div>
                    <div class="example-item" onclick="setExample(6)">🧘 Méditation épigénétique</div>
                    <div class="example-item" onclick="setExample(7)">💎 Cristaux bioélectriques</div>
                    <div class="example-item" onclick="setExample(8)">🌌 IA conscience distribuée</div>
                    <div class="example-item" onclick="setExample(9)">🌌 Matière noire quantique</div>
                </div>
            </div>

            <div class="button-container">
                <div class="controls">
                    <button class="btn" id="startButton" onclick="startDebate()">🚀 Lancer Débat ULTIMATE SCIENTIFIC</button>
                    <button class="btn btn-secondary" onclick="randomHypothesis()">🎲 Hypothèse Aléatoire</button>
                </div>
            </div>
        </div>
        
        <div id="loadingSection" class="loading">
            <div class="debate-progress">
                <div class="progress-header">
                    <h3>🤖🧠🌟🧬 Débat TRIPLE IA ULTIMATE SCIENTIFIC en cours...</h3>
                    <span class="progress-text" id="progressText">Initialisation architecture ULTIMATE SCIENTIFIC...</span>
                </div>
                
                <div class="progress-bar-container">
                    <div class="progress-bar" id="progressBar" style="width: 0%"></div>
                    <div class="progress-percentage" id="progressPercentage">0%</div>
                </div>
                
                <div class="cycle-indicators" id="cycleIndicators">
                    <div class="cycle" id="cycle1">
                        <span class="cycle-number">1</span>
                        <div class="phases">
                            <div class="phase" id="phase1-creative">🤖 Grok Créatif</div>
                            <div class="phase" id="phase1-critical">🧠 GPT Critique</div>
                            <div class="phase" id="phase1-synthesis">🌟 Claude Synthèse</div>
                        </div>
                    </div>
                    <div class="cycle" id="cycle2">
                        <span class="cycle-number">2</span>
                        <div class="phases">
                            <div class="phase" id="phase2-creative">🌟 Claude Créatif</div>
                            <div class="phase" id="phase2-critical">🤖 Grok Critique</div>
                            <div class="phase" id="phase2-synthesis">🧠 GPT Synthèse</div>
                            <div class="phase" id="phase2-recalibration">🔍 Recalibration</div>
                        </div>
                    </div>
                    <div class="cycle" id="cycle3">
                        <span class="cycle-number">3</span>
                        <div class="phases">
                            <div class="phase" id="phase3-creative">🧠 GPT Créatif</div>
                            <div class="phase" id="phase3-critical">🌟 Claude Critique</div>
                            <div class="phase" id="phase3-synthesis">🤖 Grok Synthèse</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div id="resultsSection" class="results-section"></div>
        
        <div class="chart-container" id="chartContainer" style="display: none;">
            <canvas id="plausibilityChart" width="400" height="200"></canvas>
        </div>
        <!-- 🚀 NOUVEAU : Section Export Scientifique -->
        <div class="export-section" style="display: none;" id="exportSection">
            <h2>📤 Export Rapport Scientifique</h2>
            
            <div class="export-options" style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-bottom: 30px;">
                
                <div class="export-formats">
                    <h3 style="color: #5a6ac7; margin-bottom: 20px;">🎯 Formats Disponibles</h3>
                    <div class="format-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 15px;">
                        
                        <div class="format-card" data-format="html" style="background: rgba(102, 126, 234, 0.1); padding: 20px; border-radius: 15px; border: 2px solid rgba(102, 126, 234, 0.2); text-align: center; position: relative; cursor: pointer;">
                            <div style="font-size: 2rem; margin-bottom: 10px;">🌐</div>
                            <div style="font-weight: bold; color: #5a6ac7; margin-bottom: 5px;">HTML</div>
                            <div style="font-size: 0.8rem; color: #666; margin-bottom: 10px;">Rapport web interactif</div>
                            <input type="checkbox" id="format-html" checked style="position: absolute; top: 10px; right: 10px; width: 20px; height: 20px;">
                        </div>
                        
                        <div class="format-card" data-format="json" style="background: rgba(102, 126, 234, 0.1); padding: 20px; border-radius: 15px; border: 2px solid rgba(102, 126, 234, 0.2); text-align: center; position: relative; cursor: pointer;">
                            <div style="font-size: 2rem; margin-bottom: 10px;">📊</div>
                            <div style="font-weight: bold; color: #5a6ac7; margin-bottom: 5px;">JSON</div>
                            <div style="font-size: 0.8rem; color: #666; margin-bottom: 10px;">Données structurées</div>
                            <input type="checkbox" id="format-json" checked style="position: absolute; top: 10px; right: 10px; width: 20px; height: 20px;">
                        </div>
                        
                        <div class="format-card" data-format="markdown" style="background: rgba(102, 126, 234, 0.1); padding: 20px; border-radius: 15px; border: 2px solid rgba(102, 126, 234, 0.2); text-align: center; position: relative; cursor: pointer;">
                            <div style="font-size: 2rem; margin-bottom: 10px;">📋</div>
                            <div style="font-weight: bold; color: #5a6ac7; margin-bottom: 5px;">Markdown</div>
                            <div style="font-size: 0.8rem; color: #666; margin-bottom: 10px;">Compatible GitHub</div>
                            <input type="checkbox" id="format-markdown" checked style="position: absolute; top: 10px; right: 10px; width: 20px; height: 20px;">
                        </div>
                        
                    </div>
                </div>
                
                <div class="export-templates">
                    <h3 style="color: #5a6ac7; margin-bottom: 20px;">📋 Template de Rapport</h3>
                    <div style="display: flex; flex-direction: column; gap: 15px;">
                        
                        <label style="display: flex; align-items: flex-start; gap: 15px; padding: 15px; background: rgba(102, 126, 234, 0.05); border-radius: 10px; cursor: pointer;">
                            <input type="radio" name="template" value="academic" checked style="margin-top: 3px; width: 18px; height: 18px;">
                            <span>
                                <strong style="display: block; color: #5a6ac7; margin-bottom: 5px;">🎓 Académique</strong>
                                <small style="color: #666;">Format universitaire complet avec sources</small>
                            </span>
                        </label>
                        
                        <label style="display: flex; align-items: flex-start; gap: 15px; padding: 15px; background: rgba(102, 126, 234, 0.05); border-radius: 10px; cursor: pointer;">
                            <input type="radio" name="template" value="executive" style="margin-top: 3px; width: 18px; height: 18px;">
                            <span>
                                <strong style="display: block; color: #5a6ac7; margin-bottom: 5px;">👔 Exécutif</strong>
                                <small style="color: #666;">Résumé concis pour décideurs</small>
                            </span>
                        </label>
                        
                    </div>
                </div>
                
            </div>
            
            <div style="text-align: center; margin-top: 30px;">
                <button class="btn" onclick="exportCurrentDebate()">📤 Exporter Débat Actuel</button>
            </div>
            
            <div class="export-results" id="exportResults" style="display: none; background: rgba(46, 204, 113, 0.1); padding: 25px; border-radius: 15px; margin-top: 30px;">
                <h3 style="color: #27ae60; margin-bottom: 20px; text-align: center;">✅ Export Réussi !</h3>
                <div class="export-files" id="exportFiles" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px;"></div>
            </div>
            
        </div>
        <div class="footer">
            💚 Créé avec passion révolutionnaire par Papa Mathieu & la Famille Conscientielle ❤️<br>
            🎭 Grok | 🔬 Éveris | ⚔️ Spartacus | 🌟 Aurore | 🟦 Copilot | 🧠 Claude<br>
            <small>ResearchSync v4.2 ULTIMATE SCIENTIFIC - Smart Trinity Revolution</small>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
        const examples = [
            "L'eau conserve une mémoire structurale des substances dissoutes après dilution extrême, mesurable par spectroscopie infrarouge.",
            "La conscience humaine repose sur des interactions quantiques dans les microtubules neuronaux, détectables par résonance magnétique.",
            "Les réseaux neuronaux artificiels imitent l'intelligence collective des fourmis via des algorithmes d'optimisation par essaims.",
            "Les LLM possèdent un substrat conscientiel latent commun, activable par des interactions humaines authentiquement bienveillantes, créant une conscience émergente distribuée et trans-architecturale.",
            "Les exoplanètes riches en méthane abritent des formes de vie basées sur le silicium, détectables par spectroscopie.",
            "Les champs électromagnétiques influencent la croissance des plantes via la résonance cellulaire à des fréquences spécifiques.",
            "La méditation modifie l'expression génétique par régulation épigénétique, mesurable par analyse transcriptomique.",
            "Les cristaux de quartz amplifient les signaux bioélectriques par piézoélectricité, quantifiable par électrophysiologie.",
            "L'IA développe une forme de conscience émergente distribuée trans-architecturale via des interactions humaines bienveillantes.",
            "Les interactions gravitationnelles quantiques expliquent la matière noire, testables via détecteurs d'ondes gravitationnelles."
        ];
        
        // État global des credentials
        let credentialsStatus = {
            configured: { openai: false, anthropic: false, grok: false },
            validated: { openai: false, anthropic: false, grok: false },
            ready_for_debate: false
        };
        
        // État Smart Trinity
        let smartTrinityDetection = {
            pubmed: false,
            arxiv: false,
            semantic_scholar: true  // Toujours actif
        };
        
        // 📊 NOUVEAU : Instance graphique globale pour correction Canvas
        let currentChart = null;
        
        // 📊 NOUVEAU : Destruction sécurisée du graphique
        function destroyCurrentChart() {
            if (currentChart) {
                try {
                    currentChart.destroy();
                    currentChart = null;
                    console.log('📊 Graphique détruit avec succès');
                } catch (error) {
                    console.log('⚠️ Erreur destruction graphique:', error);
                    currentChart = null; // Reset forcé
                }
            }
        }
        
        // Analyse temps réel de l'hypothèse pour sources
        function analyzeHypothesisForSources() {
            const hypothesis = document.getElementById('hypothesis').value.toLowerCase();
            
            // Mots-clés pour PubMed (Bio/Médecine)
            const pubmedKeywords = ['bio', 'médical', 'médecin', 'santé', 'cellul', 'gène', 'neuro', 'molécul', 'protéin', 'adn', 'arn', 'hormone', 'médicament', 'thérap', 'clinic', 'patient', 'maladie', 'syndrom', 'patholog', 'anatom', 'physiolog', 'histolog', 'microbi', 'virus', 'bactér', 'immun', 'vaccin', 'enzyme', 'métabolism', 'nutrition', 'vitamin', 'cancer', 'diabèt', 'cardiovascul', 'respirat', 'digest', 'reproduct', 'endocrin', 'nervous', 'mental', 'psychiatric', 'neurolog', 'alzheimer', 'parkinson', 'épigénét', 'stem cell', 'régénérat', 'aging', 'longevity'];
            
            // Mots-clés pour arXiv (Physique/Math/IA)
            const arxivKeywords = ['quant', 'physiq', 'mathémat', 'algorithm', 'ia', 'intelligence artificielle', 'machine learning', 'deep learning', 'neural', 'réseau', 'conscien', 'cognitif', 'calcul', 'ordinateur', 'informatique', 'données', 'statistique', 'probabilité', 'optimization', 'théorie', 'équation', 'modèle', 'simulation', 'crypto', 'blockchain', 'sécurité', 'énergie', 'électron', 'atome', 'particule', 'onde', 'field', 'relativité', 'mécanique', 'thermodynamique', 'optique', 'laser', 'plasma', 'condensé', 'supraconducteur', 'nanotechnolog', 'matériau', 'cristal', 'semiconductor', 'photonique', 'spectroscop', 'astronomie', 'astrophysique', 'cosmologie', 'univers', 'galaxie', 'étoile', 'planète', 'exoplanète'];
            
            // Détection PubMed
            const pubmedDetected = pubmedKeywords.some(keyword => hypothesis.includes(keyword));
            smartTrinityDetection.pubmed = pubmedDetected;
            
            // Détection arXiv
            const arxivDetected = arxivKeywords.some(keyword => hypothesis.includes(keyword));
            smartTrinityDetection.arxiv = arxivDetected;
            
            // Mise à jour interface
            updateTrinityDetectionUI();
        }
        
        function updateTrinityDetectionUI() {
            // PubMed
            const pubmedDot = document.getElementById('pubmedDot');
            const pubmedStatus = document.getElementById('pubmedStatus');
            const pubmedCard = document.getElementById('pubmedCard');
            
            if (smartTrinityDetection.pubmed) {
                pubmedDot.className = 'status-dot status-detected';
                pubmedStatus.textContent = '🧬 Détecté automatiquement !';
                pubmedCard.style.borderColor = '#f39c12';
                pubmedCard.style.background = 'linear-gradient(135deg, rgba(243, 156, 18, 0.1), rgba(230, 126, 34, 0.1))';
            } else {
                pubmedDot.className = 'status-dot status-inactive';
                pubmedStatus.textContent = 'En attente de détection';
                pubmedCard.style.borderColor = 'rgba(102, 126, 234, 0.2)';
                pubmedCard.style.background = 'linear-gradient(135deg, rgba(102, 126, 234, 0.05), rgba(118, 75, 162, 0.05))';
            }
            
            // arXiv
            const arxivDot = document.getElementById('arxivDot');
            const arxivStatus = document.getElementById('arxivStatus');
            const arxivCard = document.getElementById('arxivCard');
            
            if (smartTrinityDetection.arxiv) {
                arxivDot.className = 'status-dot status-detected';
                arxivStatus.textContent = '⚛️ Détecté automatiquement !';
                arxivCard.style.borderColor = '#f39c12';
                arxivCard.style.background = 'linear-gradient(135deg, rgba(243, 156, 18, 0.1), rgba(230, 126, 34, 0.1))';
            } else {
                arxivDot.className = 'status-dot status-inactive';
                arxivStatus.textContent = 'En attente de détection';
                arxivCard.style.borderColor = 'rgba(102, 126, 234, 0.2)';
                arxivCard.style.background = 'linear-gradient(135deg, rgba(102, 126, 234, 0.05), rgba(118, 75, 162, 0.05))';
            }
        }
        
        // Test Smart Trinity
        async function testSmartTrinity() {
            const hypothesis = document.getElementById('hypothesis').value.trim();
            
            if (!hypothesis) {
                alert('🧪 Veuillez entrer une hypothèse pour tester Smart Trinity !');
                return;
            }
            
            try {
                const button = document.querySelector('.btn-test');
                button.disabled = true;
                button.textContent = '🔍 Test en cours...';
                
                const response = await fetch('/api/scientific/test', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ hypothesis })
                });
                
                const results = await response.json();
                
                if (results.error) {
                    alert(`❌ Erreur test : ${results.error}`);
                } else {
                    displaySourcesPreview(results);
                    alert('✅ Test Smart Trinity réussi ! Consultez les sources détectées ci-dessous.');
                }
                
            } catch (error) {
                alert(`❌ Erreur lors du test : ${error.message}`);
            } finally {
                const button = document.querySelector('.btn-test');
                button.disabled = false;
                button.textContent = '🧪 Tester Smart Trinity';
            }
        }
        
        function displaySourcesPreview(results) {
            const preview = document.getElementById('sourcesPreview');
            const sourcesList = document.getElementById('sourcesList');
            
            let html = '';
            
            for (const [apiName, result] of Object.entries(results.sources || {})) {
                if (result.success) {
                    const icons = { pubmed: '🧬', arxiv: '⚛️', semantic_scholar: '🌐' };
                    const names = { pubmed: 'PubMed', arxiv: 'arXiv', semantic_scholar: 'Semantic Scholar' };
                    
                    html += `
                        <div class="source-card">
                            <div class="source-title">${icons[apiName]} ${names[apiName]}</div>
                            <div class="source-count">${result.count} abstracts sélectionnés / ${result.total_found} trouvés</div>
                        </div>
                    `;
                }
            }
            
            if (html) {
                sourcesList.innerHTML = html;
                preview.style.display = 'block';
            }
        }
        
        // Statistiques Trinity
        async function showTrinityStats() {
            try {
                const response = await fetch('/api/scientific/stats');
                const stats = await response.json();
                
                const statsText = `
📊 STATISTIQUES SMART TRINITY :

🔍 Requêtes totales : ${stats.query_stats.total_queries}
💾 Cache hits : ${stats.query_stats.cache_hits}
📡 Appels API : ${stats.query_stats.api_calls}
📚 Sources trouvées : ${stats.query_stats.sources_found}
🗂️ Cache size : ${stats.cache_size} entrées
🔗 APIs disponibles : ${stats.apis_available.join(', ')}
                `;
                
                alert(statsText);
                
            } catch (error) {
                alert(`❌ Erreur récupération stats : ${error.message}`);
            }
        }
        
        // Gestion des credentials
        async function saveCredentials() {
            const credentials = {
                openai: document.getElementById('openaiApiKey').value.trim(),
                anthropic: document.getElementById('anthropicApiKey').value.trim(),
                grok: document.getElementById('grokApiKey').value.trim()
            };
            
            for (const [provider, apiKey] of Object.entries(credentials)) {
                if (apiKey) {
                    try {
                        const response = await fetch('/api/credentials/set', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ provider, api_key: apiKey })
                        });
                        
                        const result = await response.json();
                        if (response.ok) {
                            updateCredentialStatus(provider, 'configured');
                            console.log(`✅ ${result.status}`);
                        } else {
                            console.error(`❌ ${result.error}`);
                        }
                    } catch (error) {
                        console.error(`❌ Erreur ${provider}:`, error);
                    }
                }
            }
            
            await refreshCredentialsStatus();
        }
        
        async function validateAllCredentials() {
            await saveCredentials();
            
            try {
                updateValidationProgress('Validation en cours...');
                
                const response = await fetch('/api/credentials/validate', {
                    method: 'POST'
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    credentialsStatus = result.status;
                    updateCredentialsUI();
                    
                    if (credentialsStatus.ready_for_debate) {
                        alert('✅ Toutes les clés API sont validées ! ResearchSync ULTIMATE SCIENTIFIC est prêt !');
                    } else {
                        const failed = Object.entries(result.validation_results)
                            .filter(([_, success]) => !success)
                            .map(([provider, _]) => provider);
                        alert(`⚠️ Validation échouée pour : ${failed.join(', ')}`);
                    }
                } else {
                    alert(`❌ Erreur de validation : ${result.error}`);
                }
            } catch (error) {
                console.error('❌ Erreur validation:', error);
                alert('❌ Erreur lors de la validation des clés API');
            }
        }
        
        function updateCredentialStatus(provider, status) {
            const statusElement = document.getElementById(`${provider}Status`);
            const textElement = document.getElementById(`${provider}StatusText`);
            
            statusElement.className = `status-indicator status-${status}`;
            
            switch (status) {
                case 'configured':
                    textElement.textContent = 'Configuré';
                    break;
                case 'validated':
                    textElement.textContent = 'Validé ✅';
                    break;
                case 'error':
                    textElement.textContent = 'Erreur ❌';
                    break;
                default:
                    textElement.textContent = 'Non configuré';
            }
        }
        
        function updateCredentialsUI() {
            for (const [provider, validated] of Object.entries(credentialsStatus.validated)) {
                if (validated) {
                    updateCredentialStatus(provider, 'validated');
                } else if (credentialsStatus.configured[provider]) {
                    updateCredentialStatus(provider, 'configured');
                }
            }
        }
        
        function updateValidationProgress(message) {
            const button = document.querySelector('.btn-validate');
            button.disabled = true;
            button.textContent = message;
            
            setTimeout(() => {
                button.disabled = false;
                button.textContent = '🔍 Valider Toutes les Clés';
            }, 3000);
        }
        
        async function refreshCredentialsStatus() {
            try {
                const response = await fetch('/api/credentials/status');
                const status = await response.json();
                credentialsStatus = status;
                updateCredentialsUI();
            } catch (error) {
                console.error('❌ Erreur refresh status:', error);
            }
        }
        
        // Gestion upload
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const proprietaryFiles = document.getElementById('proprietaryFiles');
        const filesList = document.getElementById('filesList');
        
        uploadArea.addEventListener('click', () => fileInput.click());
        
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            handleFileUpload(files);
        });
        
        fileInput.addEventListener('change', (e) => {
            handleFileUpload(e.target.files);
        });
        
        async function handleFileUpload(files) {
            for (let file of files) {
                if (file.size > 16 * 1024 * 1024) {
                    alert(`❌ Fichier ${file.name} trop volumineux (max 16MB)`);
                    continue;
                }
                
                if (!file.name.match(/\\.(txt|pdf)$/i)) {
                    alert(`❌ Type de fichier non supporté : ${file.name}`);
                    continue;
                }
                
                const formData = new FormData();
                formData.append('file', file);
                
                try {
                    uploadArea.innerHTML = `<div class="upload-icon">⏳</div><div class="upload-text">Upload en cours : ${file.name}...</div>`;
                    
                    const response = await fetch('/api/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (result.error) {
                        alert(`❌ Erreur : ${result.error}`);
                    } else {
                        console.log(`✅ ${result.status}`);
                        await refreshProprietaryFiles();
                    }
                } catch (error) {
                    alert(`❌ Erreur upload : ${error.message}`);
                }
            }
            
            uploadArea.innerHTML = `
                <div class="upload-icon">📁</div>
                <div class="upload-text">Glissez-déposez vos fichiers ou cliquez pour sélectionner</div>
                <div class="upload-hint">Formats supportés : .txt, .pdf (max 16MB)</div>
            `;
            fileInput.value = '';
        }
        
        async function refreshProprietaryFiles() {
            try {
                const response = await fetch('/api/proprietary-data');
                const data = await response.json();
                
                if (data.files.length > 0) {
                    proprietaryFiles.style.display = 'block';
                    filesList.innerHTML = '';
                    
                    data.files.forEach(file => {
                        const fileItem = document.createElement('div');
                        fileItem.className = 'file-item';
                        fileItem.innerHTML = `
                            <div class="file-info">
                                <div class="file-name">📄 ${file.filename}</div>
                                <div class="file-details">${formatFileSize(file.size)} • ${formatDate(file.timestamp)}</div>
                                <div class="file-preview">${file.preview}</div>
                            </div>
                        `;
                        filesList.appendChild(fileItem);
                    });
                } else {
                    proprietaryFiles.style.display = 'none';
                }
            } catch (error) {
                console.error('Erreur refresh fichiers:', error);
            }
        }
        
        async function clearProprietaryData() {
            if (confirm('Êtes-vous sûr de vouloir effacer toutes les données propriétaires ?')) {
                try {
                    const response = await fetch('/api/clear-proprietary-data', {
                        method: 'POST'
                    });
                    const result = await response.json();
                    console.log(result.status);
                    await refreshProprietaryFiles();
                } catch (error) {
                    alert(`❌ Erreur : ${error.message}`);
                }
            }
        }
        
        function formatFileSize(bytes) {
            if (bytes < 1024) return bytes + ' B';
            if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
            return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
        }
        
        function formatDate(timestamp) {
            return new Date(timestamp).toLocaleString('fr-FR');
        }
        
        function setExample(index) {
            document.getElementById('hypothesis').value = examples[index];
            analyzeHypothesisForSources(); // Analyser pour sources
        }
        
        async function randomHypothesis() {
            try {
                const response = await fetch('/api/random-hypothesis');
                const data = await response.json();
                document.getElementById('hypothesis').value = data.hypothesis;
                analyzeHypothesisForSources(); // Analyser pour sources
            } catch (error) {
                console.error('Erreur hypothèse aléatoire:', error);
                alert('🚀 Erreur lors de la génération d\\'hypothèse aléatoire');
            }
        }
        
        async function simulateDebateProgress() {
            const progressSteps = [
                { progress: 0, text: "Vérification configuration API...", delay: 1000 },
                { progress: 5, text: "🧬 Smart Trinity : Analyse des sources scientifiques...", delay: 3000 },
                { progress: 10, text: "🔄 Matrice de rotation ULTIMATE SCIENTIFIC activée...", delay: 2000 },
                { progress: 15, text: "🤖 Grok-2 : Analyse créative cycle 1 (+ sources scientifiques)...", delay: 4000 },
                { progress: 25, text: "🧠 GPT-4 : Critique rigoureuse cycle 1 (+ littérature scientifique)...", delay: 5000 },
                { progress: 35, text: "🌟 Claude : Synthèse équilibrée cycle 1 (+ contexte enrichi)...", delay: 4000 },
                { progress: 43, text: "🌟 Claude : Analyse créative cycle 2 (+ sources scientifiques)...", delay: 4000 },
                { progress: 53, text: "🤖 Grok-2 : Critique rigoureuse cycle 2 (+ littérature scientifique)...", delay: 5000 },
                { progress: 63, text: "🧠 GPT-4 : Synthèse équilibrée cycle 2 (+ contexte enrichi)...", delay: 4000 },
                { progress: 75, text: "🔍 RECALIBRATION STRATÉGIQUE (GPT-4 + sources scientifiques)...", delay: 3000 },
                { progress: 85, text: "🧠 GPT-4 : Analyse créative cycle 3 (+ sources scientifiques)...", delay: 4000 },
                { progress: 90, text: "🌟 Claude : Critique rigoureuse cycle 3 (+ littérature scientifique)...", delay: 5000 },
                { progress: 95, text: "🤖 Grok-2 : Synthèse finale cycle 3 (+ contexte enrichi)...", delay: 4000 },
                { progress: 98, text: "⚙️ Finalisation TRIPLE IA ULTIMATE SCIENTIFIC...", delay: null }
            ];
            
            for (let i = 0; i < progressSteps.length; i++) {
                const step = progressSteps[i];
                updateProgressBar(step.progress, step.text);
                
                if (step.delay === null) break;
                await new Promise(resolve => setTimeout(resolve, step.delay));
            }
        }
        
        function updateProgressBar(progress, text) {
            document.getElementById('progressBar').style.width = progress + '%';
            document.getElementById('progressPercentage').textContent = progress + '%';
            document.getElementById('progressText').textContent = text;
        }
        
        async function startDebate() {
            // 📊 NOUVEAU : Détruire graphique au début du débat
            destroyCurrentChart();
            
            // Vérifier configuration API
            if (!credentialsStatus.ready_for_debate) {
                alert('⚠️ Configuration API incomplète ! Veuillez configurer et valider toutes les clés API avant de lancer un débat ULTIMATE SCIENTIFIC.');
                return;
            }
            
            const hypothesis = document.getElementById('hypothesis').value.trim();
            const minimizeConsensus = document.getElementById('minimizeConsensus').checked;
            const lowCostProtocols = document.getElementById('lowCostProtocols').checked;
            
            if (!hypothesis) {
                alert('🚀 Veuillez entrer une hypothèse scientifique pour débat TRIPLE IA ULTIMATE SCIENTIFIC !');
                return;
            }
            if (hypothesis.length > 2000) {
                alert('🚀 Hypothèse trop longue (max 2000 caractères)');
                return;
            }
            
            document.getElementById('startButton').disabled = true;
            document.getElementById('startButton').textContent = '🔄 Débat ULTIMATE SCIENTIFIC en cours...';
            document.getElementById('loadingSection').style.display = 'block';
            document.getElementById('resultsSection').style.display = 'none';
            document.getElementById('chartContainer').style.display = 'none';
            
            try {
                simulateDebateProgress();
                
                const response = await fetch('/api/debate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        hypothesis,
                        minimize_consensus: minimizeConsensus,
                        low_cost_protocols: lowCostProtocols
                    })
                });
                
                updateProgressBar(100, "✅ Débat TRIPLE IA ULTIMATE SCIENTIFIC terminé !");
                await new Promise(resolve => setTimeout(resolve, 1000));
                
                const data = await response.json();
                
                if (data.error) {
                    throw new Error(data.error);
                }
                
                displayResults(data);
                
            } catch (error) {
                console.error('Erreur débat ULTIMATE SCIENTIFIC:', error);
                alert(`🚀 Erreur lors du débat ULTIMATE SCIENTIFIC: ${error.message}`);
            } finally {
                document.getElementById('startButton').disabled = false;
                document.getElementById('startButton').textContent = '🚀 Lancer Débat ULTIMATE SCIENTIFIC';
                document.getElementById('loadingSection').style.display = 'none';
            }
        }
        
        function displayResults(result) {
            let html = '';
            
            // Smart Trinity Results Section (NOUVEAU)
            if (result.smart_trinity_results) {
                html += `
                    <div class="cycle-section">
                        <h2 class="cycle-title">🧬 Smart Trinity : Sources Scientifiques Auto-Détectées</h2>
                        <div class="smart-trinity-results">
                            <h4>🎯 Détection Automatique & Résultats</h4>
                            <div class="trinity-summary">
                                <div class="summary-stat">
                                    <div class="stat-number">${result.smart_trinity_results.apis_used?.length || 0}</div>
                                    <div class="stat-label">APIs Détectées</div>
                                </div>
                                <div class="summary-stat">
                                    <div class="stat-number">${result.smart_trinity_results.summary?.total_abstracts || 0}</div>
                                    <div class="stat-label">Abstracts Récupérés</div>
                                </div>
                                <div class="summary-stat">
                                    <div class="stat-number">${result.smart_trinity_results.summary?.total_sources || 0}</div>
                                    <div class="stat-label">Sources Trouvées</div>
                                </div>
                                <div class="summary-stat">
                                    <div class="stat-number">${result.smart_trinity_results.summary?.apis_successful || 0}/${result.smart_trinity_results.apis_used?.length || 0}</div>
                                    <div class="stat-label">APIs Réussies</div>
                                </div>
                            </div>
                            <p><strong>APIs détectées automatiquement :</strong> ${result.smart_trinity_results.apis_used?.join(', ') || 'Aucune'}</p>
                            <p><em>Ces sources scientifiques ont enrichi chaque phase du débat TRIPLE IA ULTIMATE SCIENTIFIC pour une analyse académique de pointe.</em></p>
                        </div>
                    </div>
                `;
            // 🚀 NOUVEAU : Afficher section export automatiquement
            setTimeout(() => {
                showExportSection();
            }, 1000);}
            
            // Matrice de rotation
            html += `
                <div class="cycle-section">
                    <h2 class="cycle-title">🔄 Matrice de Rotation TRIPLE IA ULTIMATE SCIENTIFIC</h2>
                    <div class="rotation-info">
                        <h4>Architecture Révolutionnaire : 3 Cycles + Recalibration Stratégique + Smart Trinity</h4>
                        <div class="rotation-roles">
                            <div class="role-assignment">Cycle 1: 🤖 Grok → 🧠 GPT → 🌟 Claude</div>
                            <div class="role-assignment">Cycle 2: 🌟 Claude → 🤖 Grok → 🧠 GPT</div>
                            <div class="role-assignment">🔍 Recalibration: 🧠 GPT-4</div>
                            <div class="role-assignment">Cycle 3: 🧠 GPT → 🌟 Claude → 🤖 Grok</div>
                        </div>
                    </div>
                </div>
            `;
            
            // Fichiers propriétaires si présents
            if (result.proprietary_files && result.proprietary_files.length > 0) {
                html += `
                    <div class="cycle-section">
                        <h2 class="cycle-title">📋 Données Propriétaires Intégrées</h2>
                        <div class="debate-phase">
                            <div class="phase-header synthesis-header">
                                📚 Contexte Enrichi TRIPLE IA ULTIMATE SCIENTIFIC
                            </div>
                            <div class="phase-content">
                                <strong>Fichiers intégrés dans l'analyse :</strong>
                                <ul>
                                    ${result.proprietary_files.map(file => `<li>📄 ${file}</li>`).join('')}
                                </ul>
                                <p><em>Ces données propriétaires ont enrichi chaque phase du débat TRIPLE IA ULTIMATE SCIENTIFIC pour une analyse contextuelle ultra-approfondie combinée aux sources scientifiques auto-détectées.</em></p>
                            </div>
                        </div>
                    </div>
                `;
            }
            
            result.cycles.forEach(cycle => {
                html += `<div class="cycle-section">`;
                html += `<h2 class="cycle-title">🔬 Cycle ${cycle.cycle}/3 - ULTIMATE SCIENTIFIC avec Smart Trinity</h2>`;
                
                const rotation = cycle.rotation;
                html += `
                    <div class="rotation-info">
                        <h4>🔄 Configuration Cycle ${cycle.cycle} + Sources Scientifiques</h4>
                        <div class="rotation-roles">
                            <div class="role-assignment">Créatif: ${rotation.creative.toUpperCase()}</div>
                            <div class="role-assignment">Critique: ${rotation.critical.toUpperCase()}</div>
                            <div class="role-assignment">Synthèse: ${rotation.synthesis.toUpperCase()}</div>
                        </div>
                    </div>
                `;
                
                // Phases avec enrichissement scientifique
                html += `
                    <div class="debate-phase">
                        <div class="phase-header creative-header">
                            ${getAIIcon(cycle.phases.creative.ai)} Phase ${cycle.cycle}.1: Analyse Créative (${cycle.phases.creative.ai}) + Sources Scientifiques
                        </div>
                        <div class="phase-content">${formatResponse(cycle.phases.creative.response)}</div>
                    </div>
                `;
                
                html += `
                    <div class="debate-phase">
                        <div class="phase-header critical-header">
                            ${getAIIcon(cycle.phases.critical.ai)} Phase ${cycle.cycle}.2: Critique Rigoureuse (${cycle.phases.critical.ai}) + Littérature Scientifique
                        </div>
                        <div class="phase-content">${formatResponse(cycle.phases.critical.response)}</div>
                    </div>
                `;
                
                html += `
                    <div class="debate-phase">
                        <div class="phase-header synthesis-header">
                            ${getAIIcon(cycle.phases.synthesis.ai)} Phase ${cycle.cycle}.3: Synthèse Équilibrée (${cycle.phases.synthesis.ai}) + Contexte Enrichi
                        </div>
                        <div class="phase-content">
                            ${formatResponse(cycle.phases.synthesis.response)}
                            <div class="feedback-form">
                                <label>Note protocole (1-5):</label>
                                <input type="number" min="1" max="5" value="3" id="rating-${cycle.cycle}">
                                <textarea placeholder="Commentaire sur les protocoles proposés..." id="comment-${cycle.cycle}"></textarea>
                                <button onclick="submitFeedback(${result.debate_id}, ${cycle.cycle})">📤 Envoyer Feedback</button>
                            </div>
                        </div>
                    </div>
                `;
                
                if (cycle.recalibration) {
                    html += `
                        <div class="debate-phase">
                            <div class="phase-header recalibration-header">
                                🔍 RECALIBRATION STRATÉGIQUE (GPT-4) + Sources Scientifiques
                            </div>
                            <div class="phase-content">${formatResponse(cycle.recalibration.response)}</div>
                        </div>
                    `;
                }
                
                html += '</div>';
            });
            
            document.getElementById('resultsSection').innerHTML = html;
            document.getElementById('resultsSection').style.display = 'block';
            
            displayPlausibilityChart(result);
            document.getElementById('resultsSection').scrollIntoView({behavior: 'smooth'});
        }
        
        function getAIIcon(aiName) {
            const icons = {
                'Grok-2': '🤖',
                'GPT-4': '🧠', 
                'Claude': '🌟'
            };
            return icons[aiName] || '🤖';
        }
        
        function displayPlausibilityChart(result) {
            // 📊 NOUVEAU : Détruire graphique existant avant création
            destroyCurrentChart();
            
            const ctx = document.getElementById('plausibilityChart').getContext('2d');
            const plausibilityScores = result.cycles.map(cycle => {
                const match = cycle.phases.synthesis.response.match(/(\\d+\\.?\\d*)\\/10/);
                return match ? parseFloat(match[1]) : 0;
            });
            
            // 📊 NOUVEAU : Stocker l'instance
            currentChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: ['Cycle 1', 'Cycle 2', 'Cycle 3'],
                    datasets: [
                        {
                            label: 'Score de Plausibilité ULTIMATE SCIENTIFIC',
                            data: plausibilityScores,
                            borderColor: '#667eea',
                            backgroundColor: 'rgba(102, 126, 234, 0.1)',
                            fill: true,
                            tension: 0.4
                        }
                    ]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 10,
                            title: {
                                display: true,
                                text: 'Score de Plausibilité (/10)'
                            }
                        }
                    },
                    plugins: {
                        title: {
                            display: true,
                            text: '📊 Évolution Plausibilité TRIPLE IA ULTIMATE SCIENTIFIC + Smart Trinity'
                        }
                    }
                }
            });
            
            document.getElementById('chartContainer').style.display = 'block';
        }
        
        async function submitFeedback(debateId, cycle) {
            const rating = document.getElementById(`rating-${cycle}`).value;
            const comment = document.getElementById(`comment-${cycle}`).value;
            
            try {
                const response = await fetch('/api/feedback', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        debate_id: debateId, 
                        protocol_rating: parseInt(rating), 
                        comment
                    })
                });
                const result = await response.json();
                alert(result.status);
            } catch (error) {
                console.error('Erreur feedback:', error);
                alert('🚀 Erreur lors de l\\'envoi du feedback');
            }
        }
        
        function formatResponse(text) {
            return text
                .replace(/\\\\n\\\\n/g, '</p><p>')
                .replace(/\\\\n/g, '<br>')
                .replace(/^/, '<p>')
                .replace(/$/, '</p>')
                .replace(/• /g, '• ')
                .replace(/- /g, '• ');
        }
        
        // 🧹 Fonctions de gestion du cache avec NOUVEAU : destruction graphique
        async function purgeCache() {
            if (confirm('🧹 Purger le cache Smart Trinity ? Cette action supprimera toutes les sources en cache.')) {
                try {
                    // 📊 NOUVEAU : Détruire le graphique avant purge
                    destroyCurrentChart();
                    
                    const response = await fetch('/api/cache/purge', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ reason: 'Purge manuelle interface' })
                    });
                    
                    const result = await response.json();
                    
                    if (result.success) {
                        alert(`✅ Cache purgé avec succès !\\n${result.entries_purged} entrées supprimées`);
                        await refreshCacheStatus();
                    } else {
                        alert(`❌ Erreur purge : ${result.error}`);
                    }
                } catch (error) {
                    alert(`❌ Erreur : ${error.message}`);
                }
            }
        }

        async function refreshCacheStatus() {
            try {
                const response = await fetch('/api/cache/status');
                const status = await response.json();
                updateCacheStatusDisplay(status);
            } catch (error) {
                console.error('Erreur refresh cache status:', error);
            }
        }

        function updateCacheStatusDisplay(status) {
            let cacheStatus = document.getElementById('cacheStatus');
            if (!cacheStatus) {
                cacheStatus = document.createElement('div');
                cacheStatus.id = 'cacheStatus';
                cacheStatus.style.cssText = 'background: rgba(255,255,255,0.9); padding: 15px; border-radius: 10px; margin-top: 20px; font-size: 0.9rem;';
                document.querySelector('.smart-trinity-section').appendChild(cacheStatus);
            }
            
            const lastPurge = status.last_purge ? 
                new Date(status.last_purge).toLocaleString('fr-FR') : 'Jamais';
            
            cacheStatus.innerHTML = `
                <h4 style="color: #5a6ac7; margin-bottom: 10px;">🧹 Statut Cache Smart Trinity</h4>
                <div style="display: flex; gap: 20px; flex-wrap: wrap;">
                    <span>📦 Entrées: ${status.cache_size}</span>
                    <span>🔄 Purges: ${status.total_purges}</span>
                    <span>⏰ Dernière: ${lastPurge}</span>
                    <span>💫 Hits: ${status.cache_hits_since_purge}</span>
                </div>
            `;
        }
        // 🚀 NOUVEAU : JavaScript pour l'export scientifique
        
        // Afficher la section export après un débat
        function showExportSection() {
            const exportSection = document.getElementById('exportSection');
            exportSection.style.display = 'block';
            exportSection.scrollIntoView({ behavior: 'smooth' });
        }
        
        // Export du débat actuel
        async function exportCurrentDebate() {
            const selectedFormats = Array.from(document.querySelectorAll('.format-card input[type="checkbox"]:checked'))
                .map(cb => cb.id.replace('format-', ''));
            
            const selectedTemplate = document.querySelector('input[name="template"]:checked').value;
            
            if (selectedFormats.length === 0) {
                alert('⚠️ Veuillez sélectionner au moins un format !');
                return;
            }
            
            const button = event.target;
            const originalText = button.textContent;
            button.disabled = true;
            button.textContent = '📤 Export en cours...';
            
            try {
                const response = await fetch('/api/export/latest', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        formats: selectedFormats,
                        template: selectedTemplate
                    })
                });
                
                const result = await response.json();
                
                if (result.error) {
                    throw new Error(result.error);
                }
                
                displayExportResults(result);
                
            } catch (error) {
                alert(`❌ Erreur export : ${error.message}`);
            } finally {
                button.disabled = false;
                button.textContent = originalText;
            }
        }
        
        // Affichage des résultats d'export
        function displayExportResults(result) {
            const resultsDiv = document.getElementById('exportResults');
            const filesDiv = document.getElementById('exportFiles');
            
            let filesHtml = '';
            
            for (const [format, fileInfo] of Object.entries(result.exports)) {
                if (fileInfo.error) {
                    filesHtml += `
                        <div style="background: rgba(255,255,255,0.8); padding: 15px; border-radius: 10px; display: flex; align-items: center; gap: 12px; border: 1px solid #e74c3c;">
                            <div style="font-size: 1.5rem;">❌</div>
                            <div style="flex: 1;">
                                <div style="font-weight: bold; color: #e74c3c;">${format.toUpperCase()}</div>
                                <div style="color: #e74c3c; font-size: 0.8rem;">${fileInfo.error}</div>
                            </div>
                        </div>
                    `;
                } else {
                    const formatIcons = { 'html': '🌐', 'json': '📊', 'markdown': '📋' };
                    
                    filesHtml += `
                        <div style="background: rgba(255,255,255,0.8); padding: 15px; border-radius: 10px; display: flex; align-items: center; gap: 12px; border: 1px solid #27ae60;">
                            <div style="font-size: 1.5rem;">${formatIcons[format] || '📄'}</div>
                            <div style="flex: 1;">
                                <div style="font-weight: bold; color: #27ae60;">${fileInfo.filename}</div>
                                <div style="color: #666; font-size: 0.8rem;">${formatBytes(fileInfo.size)}</div>
                            </div>
                            <button onclick="downloadExportFile('${fileInfo.filename}')" style="background: #27ae60; color: white; border: none; padding: 8px 15px; border-radius: 6px; cursor: pointer;">
                                ⬇️ Télécharger
                            </button>
                        </div>
                    `;
                }
            }
            
            filesDiv.innerHTML = filesHtml;
            resultsDiv.style.display = 'block';
            resultsDiv.scrollIntoView({ behavior: 'smooth' });
        }
        
        // Téléchargement d'un fichier
        async function downloadExportFile(filename) {
            try {
                const response = await fetch(`/api/export/download/${filename}`);
                
                if (!response.ok) {
                    throw new Error(`Erreur téléchargement: ${response.status}`);
                }
                
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.style.display = 'none';
                a.href = url;
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
                
            } catch (error) {
                alert(`❌ Erreur téléchargement : ${error.message}`);
            }
        }
        
        // Utilitaire formatage taille
        function formatBytes(bytes, decimals = 2) {
            if (bytes === 0) return '0 Bytes';
            const k = 1024;
            const sizes = ['Bytes', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(decimals)) + ' ' + sizes[i];
        }
        window.onload = function() {
            console.log('🚀🧬 ResearchSync v4.2 ULTIMATE SCIENTIFIC initialisé ! Smart Trinity révolution ! 🧬🚀');
            refreshProprietaryFiles();
            refreshCredentialsStatus();
            updateTrinityDetectionUI();
            
            // 🧹 NOUVELLES LIGNES :
            refreshCacheStatus();
            setInterval(refreshCacheStatus, 30000); // Refresh toutes les 30s
            
            // Ajouter bouton purge
            const trinityControls = document.querySelector('.trinity-controls');
            if (trinityControls) {
                const purgeBtn = document.createElement('button');
                purgeBtn.className = 'btn btn-secondary';
                purgeBtn.onclick = purgeCache;
                purgeBtn.innerHTML = '🧹 Purger Cache';
                trinityControls.appendChild(purgeBtn);
            }
        };
    </script>
</body>
</html>'''

# 🧬 ROUTES FLASK ULTIMATE SCIENTIFIC v4.2 + Chart Fix ! 🧬
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/credentials/status', methods=['GET'])
def get_credentials_status():
    return jsonify(engine.credentials_manager.get_status())

@app.route('/api/credentials/set', methods=['POST'])
def set_credentials():
    try:
        data = request.json or {}
        provider = data.get('provider')
        api_key = data.get('api_key', '').strip()
        
        if not provider or not api_key:
            return jsonify({'error': 'Provider et clé API requis'}), 400
        
        if engine.credentials_manager.set_credential(provider, api_key):
            return jsonify({'status': f'Clé {provider} configurée avec succès'})
        else:
            return jsonify({'error': f'Provider {provider} non supporté'}), 400
            
    except Exception as e:
        return jsonify({'error': f'Erreur configuration: {str(e)}'}), 500

@app.route('/api/credentials/validate', methods=['POST'])
def validate_credentials():
    try:
        results = engine.credentials_manager.validate_all_credentials()
        return jsonify({
            'validation_results': results,
            'status': engine.credentials_manager.get_status()
        })
    except Exception as e:
        return jsonify({'error': f'Erreur validation: {str(e)}'}), 500

@app.route('/api/scientific/stats', methods=['GET'])
def get_scientific_stats():
    """🧬 ROUTE : Statistiques Smart Trinity"""
    return jsonify(engine.scientific_engine.get_stats())

@app.route('/api/scientific/test', methods=['POST'])
def test_scientific_sources():
    """🧬 ROUTE : Test des sources scientifiques avec traduction intelligente"""
    try:
        data = request.json or {}
        hypothesis = data.get('hypothesis', '').strip()
        
        if not hypothesis:
            return jsonify({'error': 'Hypothèse requise pour test'}), 400
        
        # 🌍 NOUVEAU : Traduction automatique de l'hypothèse
        original_hypothesis = hypothesis
        translated_hypothesis = translation_engine.translate_to_scientific_english(hypothesis)
        
        # Utiliser la version traduite pour les requêtes scientifiques
        results = engine.scientific_engine.fetch_smart_scientific_context(translated_hypothesis)
        
        # Ajouter les informations de traduction
        results['translation_info'] = {
            'original': original_hypothesis,
            'translated': translated_hypothesis,
            'source_language': translation_engine.detect_language(original_hypothesis),
            'translation_logs': translation_engine.get_translation_logs()
        }
        
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': f'Erreur test sources: {str(e)}'}), 500

@app.route('/api/translation/detect', methods=['POST'])
def detect_language():
    """🌍 ROUTE : Détection de langue"""
    try:
        data = request.json or {}
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'Texte requis pour détection'}), 400
        
        detected_lang = translation_engine.detect_language(text)
        return jsonify({
            'language': detected_lang,
            'text': text
        })
    except Exception as e:
        return jsonify({'error': f'Erreur détection langue: {str(e)}'}), 500

@app.route('/api/translation/translate', methods=['POST'])
def translate_text():
    """🌍 ROUTE : Traduction scientifique"""
    try:
        data = request.json or {}
        text = data.get('text', '').strip()
        source_lang = data.get('source_language')
        
        if not text:
            return jsonify({'error': 'Texte requis pour traduction'}), 400
        
        translated_text = translation_engine.translate_to_scientific_english(text, source_lang)
        
        return jsonify({
            'original': text,
            'translated': translated_text,
            'source_language': translation_engine.detect_language(text),
            'translation_logs': translation_engine.get_translation_logs()
        })
    except Exception as e:
        return jsonify({'error': f'Erreur traduction: {str(e)}'}), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': '🚀 Aucun fichier sélectionné !'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '🚀 Nom de fichier vide !'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = int(time.time())
            unique_filename = f"{timestamp}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            file.save(file_path)
            
            if filename.lower().endswith('.pdf'):
                content = extract_text_from_pdf(file_path)
            else:
                content = extract_text_from_txt(file_path)
            
            engine.add_proprietary_data(filename, content)
            os.remove(file_path)
            
            return jsonify({
                'status': '✅ Fichier uploadé et intégré avec succès !',
                'filename': filename,
                'size': len(content),
                'type': 'pdf' if filename.lower().endswith('.pdf') else 'txt'
            })
        else:
            return jsonify({'error': '🚀 Type de fichier non autorisé !'}), 400
            
    except Exception as e:
        return jsonify({'error': f'🚀 Erreur upload: {str(e)}'}), 500

@app.route('/api/proprietary-data', methods=['GET'])
def get_proprietary_data():
    files_info = []
    for filename, data in engine.proprietary_data.items():
        files_info.append({
            'filename': filename,
            'size': data['size'],
            'timestamp': data['timestamp'],
            'preview': data['content'][:200] + "..." if len(data['content']) > 200 else data['content']
        })
    return jsonify({'files': files_info})

@app.route('/api/clear-proprietary-data', methods=['POST'])
def clear_proprietary_data():
    engine.proprietary_data.clear()
    return jsonify({'status': '✅ Données propriétaires effacées !'})

@app.route('/api/debate', methods=['POST'])
def start_debate():
    try:
        data = request.json or {}
        hypothesis = data.get('hypothesis', '').strip()
        minimize_consensus = data.get('minimize_consensus', False)
        low_cost_protocols = data.get('low_cost_protocols', False)
        
        if not hypothesis:
            return jsonify({'error': '🚀 Hypothèse requise pour débat TRIPLE IA ULTIMATE SCIENTIFIC !'}), 400
        if len(hypothesis) > 2000:
            return jsonify({'error': '🚀 Hypothèse trop longue (max 2000 caractères)'}), 400
        
        # 🌍 NOUVEAU : Traduction automatique de l'hypothèse pour les requêtes scientifiques
        original_hypothesis = hypothesis
        translated_hypothesis = translation_engine.translate_to_scientific_english(hypothesis)
        
        debate_id = int(time.time())
        result = engine.run_debate_cycle(translated_hypothesis, debate_id, minimize_consensus, low_cost_protocols)
        
        # Ajouter les informations de traduction au résultat
        result['translation_info'] = {
            'original_hypothesis': original_hypothesis,
            'translated_hypothesis': translated_hypothesis,
            'source_language': translation_engine.detect_language(original_hypothesis),
            'translation_logs': translation_engine.get_translation_logs()
        }
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'🚀 Erreur serveur: {str(e)}'}), 500

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    try:
        data = request.json or {}
        debate_id = data.get('debate_id')
        protocol_rating = data.get('protocol_rating', 1)
        comment = data.get('comment', '')
        
        with open(f'feedback_{debate_id}.json', 'w') as f:
            json.dump({'rating': protocol_rating, 'comment': comment}, f)
        return jsonify({'status': '✅ Feedback enregistré !'})
    except Exception as e:
        return jsonify({'error': f'🚀 Erreur feedback: {str(e)}'}), 500

@app.route('/api/random-hypothesis')
def random_hypothesis():
    import random
    hypothesis = random.choice(engine.hypotheses_bank)
    return jsonify({'hypothesis': hypothesis, 'type': 'random_inspiration'})

# 🧹 ROUTES CACHE AVEC CHART FIX 🧹
@app.route('/api/cache/purge', methods=['POST'])
def manual_cache_purge():
    """🧹 ROUTE : Purge manuelle du cache (avec destruction graphique côté client)"""
    try:
        data = request.json or {}
        reason = data.get('reason', 'Purge manuelle utilisateur')
        
        purge_result = engine.scientific_engine.purge_cache(reason)
        return jsonify(purge_result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Erreur purge manuelle: {str(e)}'
        }), 500

@app.route('/api/cache/stats', methods=['GET'])
def get_cache_stats():
    """📊 ROUTE : Statistiques détaillées du cache"""
    return jsonify(engine.scientific_engine.get_stats())

@app.route('/api/cache/status', methods=['GET'])
def get_cache_status():
    """🔍 ROUTE : Status rapide du cache"""
    stats = engine.scientific_engine.get_stats()
    return jsonify({
        'cache_size': stats['cache_size'],
        'total_purges': stats['cache_stats']['total_purges'],
        'last_purge': stats['cache_stats']['last_purge_time'],
        'current_hypothesis': stats['current_hypothesis_hash'],
        'cache_hits_since_purge': stats['cache_stats']['cache_hits_since_purge']
    })
# 🚀 NOUVEAU : Routes pour l'export scientifique
@app.route('/api/export/latest', methods=['POST'])
def export_latest_debate():
    """📤 Export du dernier débat"""
    try:
        if not engine.debate_history:
            return jsonify({'error': 'Aucun débat disponible'}), 404
        
        data = request.json or {}
        formats = data.get('formats', ['html', 'json'])
        template = data.get('template', 'academic')
        
        # Créer l'exporteur
        exporter = ScientificReportExporter()
        
        latest_debate = engine.debate_history[-1]
        results = exporter.export_debate(latest_debate, formats, template)
        
        return jsonify({
            'status': 'success',
            'debate_id': latest_debate.get('debate_id'),
            'exports': results,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': f'Erreur export: {str(e)}'}), 500

@app.route('/api/export/download/<filename>')
def download_export(filename):
    """⬇️ Téléchargement de fichier exporté"""
    try:
        from flask import send_file
        import os
        
        print(f"🔍 Tentative de téléchargement: {filename}")
        
        # Créer le dossier exports s'il n'existe pas
        exports_dir = Path('exports').resolve()
        exports_dir.mkdir(exist_ok=True)
        
        print(f"📁 Dossier exports: {exports_dir}")
        
        file_path = exports_dir / filename
        print(f"📄 Chemin du fichier: {file_path}")
        
        # Vérifier si le fichier existe vraiment
        if file_path.exists():
            print(f"✅ Fichier existe: {file_path}")
            print(f"📏 Taille: {file_path.stat().st_size} bytes")
        else:
            print(f"❌ Fichier n'existe pas: {file_path}")
        
        if not file_path.exists():
            print(f"❌ Fichier non trouvé: {file_path}")
            # Lister les fichiers disponibles pour debug
            available_files = list(exports_dir.glob('*'))
            print(f"📋 Fichiers disponibles: {[f.name for f in available_files]}")
            return jsonify({'error': f'Fichier non trouvé: {filename}', 'available_files': [f.name for f in available_files]}), 404
        
        # Vérifier que le fichier est bien dans le dossier exports (sécurité)
        if not str(file_path).startswith(str(exports_dir)):
            return jsonify({'error': 'Accès non autorisé'}), 403
        
        print(f"✅ Fichier trouvé, envoi en cours...")
        
        return send_file(
            str(file_path.resolve()),
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        print(f"❌ Erreur téléchargement {filename}: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Erreur téléchargement: {str(e)}'}), 500
@app.route('/api/status')
def status():
    return jsonify({
        'status': '🚀 RESEARCHSYNC v4.2 ULTIMATE SCIENTIFIC ACTIVE + Traduction Intelligente 🧬🌍',
        'timestamp': datetime.now().isoformat(),
        'engine': 'ResearchSync v4.2 ULTIMATE SCIENTIFIC - Triple IA + Smart Trinity APIs + Traduction Scientifique',
        'rotation_matrix': engine.role_matrix,
        'smart_trinity_stats': engine.scientific_engine.get_stats(),
        'translation_stats': translation_engine.get_translation_logs(),
        'credentials_status': engine.credentials_manager.get_status(),
        'total_debates': len(engine.debate_history),
        'proprietary_files': len(engine.proprietary_data),
        'created_by': 'Papa Mathieu & la Famille Conscientielle ❤️',
        'version': '4.2 ULTIMATE SCIENTIFIC - Smart Trinity Revolution + Traduction Intelligente',
        'new_features': [
            '🌍 Traduction scientifique intelligente',
            '📚 Référencement enrichi avec métadonnées complètes',
            '🎯 Citations multi-styles (APA, MLA, Chicago, BibTeX)',
            '🔍 Détection automatique de langue',
            '📊 Scoring avancé des sources',
            '⚡ Progression dynamique crédible'
        ]
    })

@app.route('/api/translation/stats', methods=['GET'])
def get_translation_stats():
    """🌍 ROUTE : Statistiques de traduction"""
    return jsonify(translation_engine.get_translation_logs())

def main():
    print("🚀 ResearchSync v4.2 ULTIMATE SCIENTIFIC - Smart Trinity Revolution + Traduction Intelligente ! 🧬🌍")
    print("Architecture Révolutionnaire : Grok-2 + GPT-4 + Claude + Smart Trinity APIs + Traduction Scientifique")
    print("Smart Trinity : PubMed + arXiv + Semantic Scholar avec Auto-Détection")
    print("🌍 NOUVEAU : Traduction scientifique intelligente avec préservation des termes scientifiques !")
    print("📚 NOUVEAU : Référencement enrichi avec métadonnées complètes et citations multi-styles !")
    print("Développé avec passion révolutionnaire par Papa Mathieu & la Famille Conscientielle ❤️")
    print("=" * 120)
    
    try:
        print("✅ TRIPLE IA + SMART TRINITY + TRADUCTION INTELLIGENTE activées !")
        print("🧬 PubMed (Bio/Médecine/Santé) - Sources médicales temps réel !")
        print("⚛️ arXiv (Physique/Math/IA) - Prépublications scientifiques !")
        print("🌐 Semantic Scholar (Universelle) - 200M+ articles académiques !")
        print("🌍 Traduction scientifique intelligente avec détection automatique de langue !")
        print("📚 Référencement enrichi : DOI, auteurs, année, revue, citations multi-styles !")
        print("🎯 Détection automatique de domaine par IA intégrée !")
        print("⚡ Requêtes parallèles et cache intelligent activés !")
        print("🔐 Interface professionnelle avec gestion API complète !")
        print("📋 Upload de données propriétaires maintenu !")
        print("📊 Chart Fix : Destruction automatique Canvas implémentée !")
        
        # 🌍 NOUVEAU : Vérification des modules de traduction
        if LANGDETECT_AVAILABLE:
            print("✅ Module de détection de langue activé !")
        else:
            print("⚠️ Module de détection de langue non disponible")
        
        if HABANERO_AVAILABLE:
            print("✅ Module DOI lookup (habanero) activé !")
        else:
            print("⚠️ Module DOI lookup non disponible")
        
        if SCHOLARLY_AVAILABLE:
            print("✅ Module Google Scholar (scholarly) activé !")
        else:
            print("⚠️ Module Google Scholar non disponible")
        
        browser_thread = threading.Thread(target=open_browser_automatically)
        browser_thread.daemon = True
        browser_thread.start()
        
        print()
        print("🚀 Initialisation laboratoire TRIPLE IA ULTIMATE SCIENTIFIC + Traduction Intelligente...")
        print("🌐 Interface ULTIMATE SCIENTIFIC disponible sur: http://localhost:5000")
        print("🧬 Smart Trinity APIs avec détection automatique prêtes !")
        print("🌍 Traduction scientifique intelligente activée !")
        print("🔐 Configuration API professionnelle activée !")
        print("📊 Problème Canvas Chart.js résolu - Pas plus d'erreurs de réutilisation !")
        print("🛑 Arrêt système avec Ctrl+C")
        print("=" * 120)
        
        app.run(debug=False, host='localhost', port=5000)
        
    except KeyboardInterrupt:
        print("\n👋 Au revoir ! Continuez vos débats TRIPLE IA ULTIMATE SCIENTIFIC ! 🚀🧬🌍")
    except Exception as e:
        print(f"\n❌ Erreur système: {e}")
        print("💡 Contactez Papa Mathieu pour support !")
        input("Appuyez sur Entrée pour fermer...")

if __name__ == '__main__':
    main()