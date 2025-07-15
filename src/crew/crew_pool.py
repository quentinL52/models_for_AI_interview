from crewai import Crew, Process
from langchain_core.tools import tool
import json
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Type
from .agents import report_generator_agent, skills_extractor_agent, experience_extractor_agent, project_extractor_agent, education_extractor_agent, ProfileBuilderAgent, informations_personnelle_agent, reconversion_detector_agent
from .tasks import generate_report_task, task_extract_skills, task_extract_experience, task_extract_projects, task_extract_education, task_build_profile, task_extract_informations, task_detect_reconversion
from src.deep_learning_analyzer import MultiModelInterviewAnalyzer
from src.rag_handler import RAGHandler 
from langchain_core.tools import BaseTool

@tool
def interview_analyser(conversation_history: list, job_description_text: list) -> str:
    """
    Appelle cet outil à la toute fin d'un entretien d'embauche pour analyser
    l'intégralité de la conversation et générer un rapport de feedback.
    Ne l'utilise PAS pour répondre à une question normale, mais seulement pour conclure et analyser l'entretien.
    """
    # 1. Analyse DL de la conversation
    analyzer = MultiModelInterviewAnalyzer()
    structured_analysis = analyzer.run_full_analysis(conversation_history, job_description_text)

    # 2. Enrichissement avec RAG
    rag_handler = RAGHandler()
    rag_feedback = []
    # Extraire les intentions et sentiments pour trouver des conseils pertinents
    if structured_analysis.get("intent_analysis"):
        for intent in structured_analysis["intent_analysis"]:
            # Exemple de requête basée sur l'intention
            query = f"Conseils pour un candidat qui cherche à {intent['labels'][0]}"
            rag_feedback.extend(rag_handler.get_relevant_feedback(query))
    
    if structured_analysis.get("sentiment_analysis"):
        for sentiment_group in structured_analysis["sentiment_analysis"]:
            for sentiment in sentiment_group:
                if sentiment['label'] == 'stress' and sentiment['score'].item() > 0.6:
                    rag_feedback.extend(rag_handler.get_relevant_feedback("gestion du stress en entretien"))
    unique_feedback = list(set(rag_feedback))
    interview_crew = Crew(
        agents=[report_generator_agent],
        tasks=[generate_report_task],
        process=Process.sequential,
        verbose=False,
        telemetry=False
    )

    final_report = interview_crew.kickoff(inputs={
        'structured_analysis_data': json.dumps(structured_analysis, indent=2),
        'rag_contextual_feedback': "\n".join(unique_feedback)
    })
    return final_report


'''
class EmptyInput(BaseModel):
    pass

class InterviewAnalysisTool(BaseTool):
    """
    Appelle cet outil à la toute fin d'un entretien d'embauche pour analyser
    l'intégralité de la conversation et générer un rapport de feedback.
    Ne l'utilise PAS pour répondre à une question normale, mais seulement pour conclure et analyser l'entretien.
    """
    name: str = "interview_analyser"
    description: str = (
        "Appelle cet outil à la toute fin d'un entretien d'embauche pour analyser "
        "l'intégralité de la conversation et générer un rapport de feedback. "
        "Ne l'utilise PAS pour répondre à une question normale, mais seulement pour conclure et analyser l'entretien."
    )
    args_schema: type[BaseModel] = EmptyInput
    job_offer: Dict[str, Any]
    conversation_history: List[Dict[str, Any]]

    def _run(self) -> str:
        """Exécute l'analyse de l'entretien."""
        interview_crew = Crew(
            agents=[report_generator_agent],
            tasks=[generate_report_task],
            process=Process.sequential,
            verbose=False,
            telemetry=False
        )
        analyzer = MultiModelInterviewAnalyzer()
        structured_analysis = analyzer.run_full_analysis(self.conversation_history, self.job_offer)

        final_report = interview_crew.kickoff(inputs={
            'structured_analysis_data': json.dumps(structured_analysis, indent=2)
        })
        return final_report
'''
def analyse_cv(cv_content: str) -> json:
    crew = Crew(
        agents=[            
            informations_personnelle_agent,
            skills_extractor_agent,
            experience_extractor_agent,
            project_extractor_agent,
            education_extractor_agent,
            reconversion_detector_agent,

            ProfileBuilderAgent       
        ],
        tasks=[
            task_extract_informations,
            task_extract_skills,
            task_extract_experience,
            task_extract_projects,
            task_extract_education,
            task_detect_reconversion,
            task_build_profile     
        ],
        process=Process.sequential,
        verbose=False,
        telemetry=False
    )
    result = crew.kickoff(inputs={"cv_content": cv_content})
    return result
 
    