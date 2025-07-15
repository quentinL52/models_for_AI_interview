import json
from datetime import datetime

# Pondérations basées sur la fiche projet
CONTEXT_WEIGHTS = {
    "formations": 0.3,
    "projets": 0.6,
    "expériences": 0.8,
    "multiple": 1.0
}

# Facteurs pour la formule de scoring
ALPHA = 0.5  # Poids du contexte
BETA = 0.3   # Poids de la fréquence
GAMMA = 0.2  # Poids de la profondeur (durée)

class ContextualScoringEngine:
    def __init__(self, cv_data: dict):
        self.cv_data = cv_data.get("candidat", {})
        self.full_text = self._get_full_text_from_cv()

    def _get_full_text_from_cv(self) -> str:
        """Concatène tout le contenu textuel du CV pour le comptage de fréquence."""
        return json.dumps(self.cv_data, ensure_ascii=False).lower()

    def _parse_date(self, date_str: str) -> datetime:
        """Parse une date, en gérant les cas spéciaux comme 'Aujourd'hui'."""
        if not date_str or date_str.lower() == "non spécifié":
            return None
        if date_str.lower() == "aujourd'hui":
            return datetime.now()
        try:
            return datetime.strptime(date_str, "%Y")
        except ValueError:
            return None

    def _calculate_duration_in_years(self, start_date_str: str, end_date_str: str) -> float:
        """Calcule la durée d'une expérience en années."""
        start_date = self._parse_date(start_date_str)
        end_date = self._parse_date(end_date_str)
        if start_date and end_date:
            return abs((end_date - start_date).days / 365.25)
        return 0.5 

    def calculate_scores(self) -> dict:
        """Calcule les scores pondérés pour toutes les hard skills."""
        skills = self.cv_data.get("compétences", {}).get("hard_skills", [])
        if not skills:
            return {}

        scored_skills = []
        for skill in skills:
            skill_lower = skill.lower()
            contexts = []
            if skill_lower in json.dumps(self.cv_data.get("formations", []), ensure_ascii=False).lower():
                contexts.append(CONTEXT_WEIGHTS["formations"])
            if skill_lower in json.dumps(self.cv_data.get("projets", []), ensure_ascii=False).lower():
                contexts.append(CONTEXT_WEIGHTS["projets"])
            if skill_lower in json.dumps(self.cv_data.get("expériences", []), ensure_ascii=False).lower():
                contexts.append(CONTEXT_WEIGHTS["expériences"])
            
            if len(contexts) > 1:
                context_score = CONTEXT_WEIGHTS["multiple"]
            elif contexts:
                context_score = contexts[0]
            else:
                context_score = 0.1 

            # 2. Fréquence de mention
            frequency_score = self.full_text.count(skill_lower)

            # 3. Profondeur d'utilisation (durée max en années)
            max_duration = 0
            for exp in self.cv_data.get("expériences", []):
                if skill_lower in json.dumps(exp, ensure_ascii=False).lower():
                    duration = self._calculate_duration_in_years(exp.get("start_date"), exp.get("end_date"))
                    if duration > max_duration:
                        max_duration = duration
            depth_score = max_duration

            # Normalisation simple (peut être affinée)
            normalized_frequency = 1 - (1 / (1 + frequency_score))
            normalized_depth = 1 - (1 / (1 + depth_score))

            # Calcul du score final
            final_score = (ALPHA * context_score) + \
                          (BETA * normalized_frequency) + \
                          (GAMMA * normalized_depth)
            
            scored_skills.append({
                "skill": skill,
                "score": round(final_score, 2),
                "details": {
                    "context_score": context_score,
                    "frequency": frequency_score,
                    "max_duration_years": round(depth_score, 1)
                }
            })
        
        # Trier par score décroissant
        scored_skills.sort(key=lambda x: x["score"], reverse=True)
        return {"analyse_competences": scored_skills}
