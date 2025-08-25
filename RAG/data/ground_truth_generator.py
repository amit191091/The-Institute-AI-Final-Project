"""
Ground Truth Generator for RAG Evaluation

This script helps create ground truth datasets for evaluating RAG performance
by generating question-answer pairs and allowing manual curation.
"""

import json
from pathlib import Path
from typing import List, Dict, Any

# Domain-specific question templates for gear failure analysis
QUESTION_TEMPLATES = [
    # Factual extraction questions
    "What is the value of {parameter} mentioned in the document?",
    "What sensors were used for {measurement_type}?",
    "What are the specifications of the {component_name}?",
    
    # Analysis questions  
    "What are the key indicators of {failure_type}?",
    "Describe the progression of {condition} over time",
    "What is the likely cause of {symptom}?",
    
    # Table/Figure specific questions
    "What information is shown in Table {number}?",
    "What does Figure {number} demonstrate?",
    "What are the wear depth values for case {case_id}?",
    
    # Timeline questions
    "What happened on {date}?",
    "How did the condition change between {start_date} and {end_date}?",
    
    # Technical questions
    "What is the recommended action for {condition}?",
    "What evidence supports {conclusion}?",
]

# Gear failure domain vocabulary
DOMAIN_TERMS = {
    "parameters": ["transmission ratio", "module", "gear type", "sampling rate", "sensitivity"],
    "measurement_types": ["vibration analysis", "imaging", "wear measurement"],
    "component_names": ["MG-5025A gearbox", "accelerometer", "tachometer", "starboard shaft"],
    "failure_types": ["gear wear", "micropitting", "macropitting", "surface distress"],
    "conditions": ["wear progression", "vibration levels", "signal characteristics"],
    "symptoms": ["harsh tonality", "increased noise", "functional loss"],
    "case_ids": ["W1", "W10", "W25", "W35", "Healthy"],
    "dates": ["April 16", "May 30", "June 13"],
    "conclusions": ["progressive surface distress", "loaded tooth flank failure"]
}

def generate_question_candidates() -> List[str]:
    """Generate candidate questions using templates and domain terms."""
    questions = []
    
    for template in QUESTION_TEMPLATES:
        # Extract placeholders from template
        import re
        placeholders = re.findall(r'\{(\w+)\}', template)
        
        # Generate combinations
        if len(placeholders) == 1:
            placeholder = placeholders[0]
            if placeholder in DOMAIN_TERMS:
                for term in DOMAIN_TERMS[placeholder]:
                    questions.append(template.format(**{placeholder: term}))
        elif len(placeholders) == 2:
            # Handle two-placeholder templates
            if all(p in DOMAIN_TERMS for p in placeholders):
                for term1 in DOMAIN_TERMS[placeholders[0]][:2]:  # Limit combinations
                    for term2 in DOMAIN_TERMS[placeholders[1]][:2]:
                        questions.append(template.format(**{
                            placeholders[0]: term1,
                            placeholders[1]: term2
                        }))
    
    # Add specific questions for this document
    specific_questions = [
        "What is the model number of the gearbox?",
        "What lubricant was used in the system?", 
        "What is the sampling rate for the vibration sensors?",
        "How many cases are shown in the wear depth table?",
        "What brand of accelerometers were used?",
        "What is the module value for the gears?",
        "What happened during the June 13 measurement?",
        "What is the wear depth for case W35?",
        "What type of gears are used in the MG-5025A?",
        "What is the sensitivity of the starboard accelerometer?",
    ]
    
    questions.extend(specific_questions)
    return questions

def create_ground_truth_template(output_path: Path):
    """Create a template file for manual ground truth creation."""
    questions = generate_question_candidates()
    
    # Create template structure
    template = {
        "metadata": {
            "document": "Gear wear Failure.pdf",
            "created_by": "ground_truth_generator.py",
            "total_questions": len(questions),
            "instructions": [
                "Fill in the 'ground_truth' field with accurate answers",
                "Mark 'include': true for questions to include in evaluation",
                "Add difficulty rating: 'easy', 'medium', 'hard'",
                "Specify question type: 'factual', 'analytical', 'table', 'figure'"
            ]
        },
        "questions": []
    }
    
    for i, question in enumerate(questions):
        template["questions"].append({
            "id": f"q_{i+1:03d}",
            "question": question,
            "ground_truth": "",  # To be filled manually
            "include": True,     # To be set manually
            "difficulty": "",    # To be set manually
            "type": "",         # To be set manually
            "notes": ""         # Optional additional notes
        })
    
    # Save template
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(template, f, indent=2, ensure_ascii=False)
    
    print(f"Created ground truth template with {len(questions)} questions: {output_path}")
    return template

def load_and_validate_ground_truth(path: Path) -> Dict[str, Any]:
    """Load and validate completed ground truth dataset."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Validate structure
    included_questions = [q for q in data["questions"] if q.get("include", False)]
    completed_questions = [q for q in included_questions if q.get("ground_truth", "").strip()]
    
    print(f"Total questions: {len(data['questions'])}")
    print(f"Included questions: {len(included_questions)}")
    print(f"Completed questions: {len(completed_questions)}")
    
    # Create evaluation dataset format
    eval_dataset = {
        "questions": [q["question"] for q in completed_questions],
        "ground_truths": [q["ground_truth"] for q in completed_questions],
        "metadata": [
            {
                "id": q["id"],
                "difficulty": q.get("difficulty", ""),
                "type": q.get("type", ""),
                "notes": q.get("notes", "")
            } for q in completed_questions
        ]
    }
    
    return eval_dataset

if __name__ == "__main__":
    # Generate template
    template_path = Path("data/ground_truth_template.json")
    template_path.parent.mkdir(exist_ok=True)
    
    create_ground_truth_template(template_path)
    
    print("\nNext steps:")
    print("1. Open data/ground_truth_template.json")
    print("2. Fill in ground_truth answers for relevant questions")
    print("3. Set include=true for questions you want to evaluate")
    print("4. Run validation to create evaluation dataset")
