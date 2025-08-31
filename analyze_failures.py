#!/usr/bin/env python3

import json

def analyze_failures():
    failed_questions = []
    with open('logs/eval_per_question.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            if data.get('answer_correctness', 1.0) == 0.0:
                failed_questions.append({
                    'question': data['question'],
                    'answer': data['answer'],
                    'reference': data['reference'],
                    'is_table': data.get('is_table_question', False)
                })

    print(f'Found {len(failed_questions)} failed questions (score 0.0)')
    
    # Analyze specific problematic patterns
    wrong_answer_pattern = 0
    for q in failed_questions:
        # Look for cases where the answer is clearly wrong like "15 and 45 RPS" for percentage questions
        if "15 and 45" in q['answer'] and any(word in q['question'].lower() for word in ['percent', '%', 'much', 'rise']):
            wrong_answer_pattern += 1
            print(f"Wrong pattern found:")
            print(f"  Q: {q['question']}")
            print(f"  A: {q['answer']}")
            print(f"  R: {q['reference']}")
            print()
    
    print(f"Found {wrong_answer_pattern} questions with wrong routing/answer pattern")

if __name__ == "__main__":
    analyze_failures()
