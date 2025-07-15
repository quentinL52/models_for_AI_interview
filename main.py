import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from datetime import datetime
from pymongo import MongoClient
from bson.objectid import ObjectId
import uvicorn
import os
import logging
from celery.result import AsyncResult
from tasks.worker_celery import run_interview_analysis_task
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from src.cv_parsing_agents import CvParserAgent
from src.interview_simulator.entretient_version_prod import InterviewProcessor
from src.scoring_engine import ContextualScoringEngine
from src.rag_handler import RAGHandler

app = FastAPI(
    title="API d'IA pour la RH",
    description="Une API pour le parsing de CV et la simulation d'entretiens.",
    version="1.2.0"
)

# Initialisation des services au démarrage
try:
    logger.info("Initialisation du RAG Handler...")
    rag_handler = RAGHandler()
    if rag_handler.vector_store:
        logger.info(f"Vector store chargé avec {rag_handler.vector_store.index.ntotal} vecteurs.")
    else:
        logger.warning("Le RAG Handler n'a pas pu être initialisé (pas de documents ?). Le feedback contextuel sera désactivé.")
except Exception as e:
    logger.error(f"Erreur critique lors de l'initialisation du RAG Handler: {e}", exc_info=True)
    rag_handler = None

# Configuration MongoDB
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(MONGO_URI)
db = client.hr_ai_system
feedback_collection = db.interview_feedbacks

class InterviewRequest(BaseModel):
    user_id: str = Field(..., example="google_user_12345")
    job_offer_id: str = Field(..., example="job_offer_abcde")
    cv_document: Dict[str, Any] = Field(..., example={"candidat": {"nom": "John Doe", "compétences": {"hard_skills": ["Python", "FastAPI"]}}})
    job_offer: Dict[str, Any] = Field(..., example={"poste": "Développeur Python", "description": "Recherche développeur expérimenté..."})
    messages: List[Dict[str, Any]]
    conversation_history: List[Dict[str, Any]]

class HealthCheck(BaseModel):
    status: str = Field(default="ok", example="ok")


@app.get("/", tags=["Status"], summary="Vérification de l'état de l'API")
def read_root() -> HealthCheck:
    """Vérifie que l'API est en cours d'exécution."""
    return HealthCheck(status="ok")

# --- Endpoint du parser de CV ---
@app.post("/parse-cv/", tags=["CV Parsing"], summary="Analyser un CV au format PDF avec scoring contextuel")
async def parse_cv_endpoint(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Le fichier doit être au format PDF.")
    tmp_path = None  
    try:
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(contents)
            tmp.flush()
            tmp_path = tmp.name
        
        logger.info(f"Début du parsing du CV temporaire : {tmp_path}")
        cv_agent = CvParserAgent(pdf_path=tmp_path)
        parsed_data = await run_in_threadpool(cv_agent.process)
        if not parsed_data:
            raise HTTPException(status_code=500, detail="Échec du parsing du CV.")
        logger.info("Parsing du CV réussi. Lancement du scoring contextuel.")
        scoring_engine = ContextualScoringEngine(parsed_data)
        scored_skills_data = await run_in_threadpool(scoring_engine.calculate_scores)
        if parsed_data.get("candidat"):
            parsed_data["candidat"].update(scored_skills_data)
        else:
            parsed_data.update(scored_skills_data)

        logger.info("Scoring terminé. Retour de la réponse complète.")
        return parsed_data

    except Exception as e:
        logger.error(f"Erreur lors du parsing ou du scoring du CV : {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur interne du serveur : {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                logger.info(f"Fichier temporaire supprimé : {tmp_path}")
            except Exception as cleanup_error:
                logger.warning(f"Erreur lors de la suppression du fichier temporaire : {cleanup_error}")

# --- Endpoint de simulation d'entretien ---
@app.post("/simulate-interview/", tags=["Simulation d'Entretien"], summary="Gérer une conversation d'entretien")
async def simulate_interview_endpoint(request: InterviewRequest):
    try:
        processor = InterviewProcessor(
            cv_document=request.cv_document,
            job_offer=request.job_offer,
            conversation_history=request.conversation_history
        )
        ai_response_object = await run_in_threadpool(processor.run, messages=request.messages)
        
        # On retourne juste la réponse de l'assistant pour le chat
        return {"response": ai_response_object["messages"][-1].content}

    except Exception as e:
        logger.error(f"Erreur interne dans /simulate-interview/: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur interne du serveur : {e}")


# --- Endpoints pour l'analyse asynchrone ---
class AnalysisRequest(BaseModel):
    conversation_history: List[Dict[str, Any]]
    job_description_text: str

@app.post("/trigger-analysis/", tags=["Analyse Asynchrone"], status_code=202)
def trigger_analysis(request: AnalysisRequest):
    """
    Déclenche l'analyse de l'entretien en tâche de fond.
    Retourne immédiatement un ID de tâche.
    """
    task = run_interview_analysis_task.delay(
        request.conversation_history, 
        [request.job_description_text]
    )
    return {"task_id": task.id}


@app.get("/analysis-status/{task_id}", tags=["Analyse Asynchrone"])
def get_analysis_status(task_id: str):
    """
    Vérifie le statut de la tâche d'analyse.
    Si terminée, retourne le résultat.
    """
    task_result = AsyncResult(task_id)
    if task_result.ready():
        if task_result.successful():
            return {
                "status": "SUCCESS",
                "result": task_result.get()
            }
        else:
            return {"status": "FAILURE", "error": str(task_result.info)}
    else:
        return {"status": "PENDING"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


## la bonne version de l'API est celle-ci, avec les imports et la structure de base.