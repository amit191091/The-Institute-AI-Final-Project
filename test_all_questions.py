#!/usr/bin/env python3
"""Test script to check all questions in the dataset."""

import json
from pathlib import Path
from RAG.app.pipeline_modules.pipeline_utils import LLM
from RAG.app.rag_service import RAGService
from RAG.app.Agent_Components.agents import answer_needle
from RAG.app.retrieve import query_analyzer, apply_filters, rerank_candidates
from RAG.app.Gradio_apps.ui_data_loader import load_ground_truth_and_qa, _norm_q

def test_all_questions():
    print("Testing all questions in the dataset...")
    
    # Load the RAG system
    print("Loading RAG system...")
    service = RAGService()
    result = service.run_pipeline(use_normalized=False)
    docs = result["docs"]
    hybrid = result["hybrid_retriever"]
    llm = LLM()
    
    # Load ground truth data
    print("Loading ground truth data...")
    gt_map, qa_map = load_ground_truth_and_qa()
    
    # Load questions from the dataset
    qa_file = Path("RAG/data/gear_wear_qa.jsonl")
    questions = []
    
    if qa_file.exists():
        with open(qa_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        questions.append({
                            'id': data.get('id', ''),
                            'question': data.get('question', ''),
                            'expected_answer': data.get('answer', '')
                        })
                    except json.JSONDecodeError:
                        continue
    
    print(f"Found {len(questions)} questions to test")
    
    # Test each question
    results = []
    success_count = 0
    gt_found_count = 0
    
    for i, q_data in enumerate(questions, 1):
        question = q_data['question']
        expected = q_data['expected_answer']
        q_id = q_data['id']
        
        print(f"\n--- Testing Question {i}: {q_id} ---")
        print(f"Question: {question}")
        print(f"Expected: {expected}")
        
        try:
            # Test retrieval
            qa = query_analyzer(question)
            candidates = hybrid.invoke(qa.get("canonical") or question) or []
            filtered = apply_filters(candidates, qa.get("filters") or {})
            top_docs = rerank_candidates(qa.get("canonical") or question, filtered, top_n=8)
            
            # Generate answer
            if top_docs:
                answer = answer_needle(llm, top_docs, question)
            else:
                answer = "No relevant documents found"
            
            # Check ground truth
            nq = _norm_q(question)
            gts = []
            
            if gt_map.get("__loaded__") and nq in gt_map.get("norm", {}):
                gts = gt_map["norm"][nq]
                gt_found_count += 1
            
            # Check if answer contains expected information
            answer_lower = answer.lower()
            expected_lower = expected.lower()
            
            # Simple similarity check
            success = False
            if expected_lower in answer_lower or answer_lower in expected_lower:
                success = True
            elif any(word in answer_lower for word in expected_lower.split() if len(word) > 3):
                success = True
            
            if success:
                success_count += 1
            
            result = {
                'id': q_id,
                'question': question,
                'expected': expected,
                'answer': answer,
                'success': success,
                'gt_found': bool(gts),
                'candidates_count': len(candidates),
                'filtered_count': len(filtered),
                'top_docs_count': len(top_docs)
            }
            
            results.append(result)
            
            print(f"Answer: {answer}")
            print(f"Success: {success}")
            print(f"GT Found: {bool(gts)}")
            print(f"Candidates: {len(candidates)}, Filtered: {len(filtered)}, Top: {len(top_docs)}")
            
        except Exception as e:
            print(f"Error processing question {i}: {e}")
            results.append({
                'id': q_id,
                'question': question,
                'expected': expected,
                'answer': f"ERROR: {e}",
                'success': False,
                'gt_found': False,
                'candidates_count': 0,
                'filtered_count': 0,
                'top_docs_count': 0
            })
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total questions tested: {len(questions)}")
    print(f"Successful answers: {success_count}/{len(questions)} ({success_count/len(questions)*100:.1f}%)")
    print(f"Ground truth found: {gt_found_count}/{len(questions)} ({gt_found_count/len(questions)*100:.1f}%)")
    
    # Show failed questions
    failed = [r for r in results if not r['success']]
    if failed:
        print(f"\nFailed questions ({len(failed)}):")
        for r in failed[:5]:  # Show first 5
            print(f"  - {r['id']}: {r['question'][:50]}...")
            print(f"    Expected: {r['expected']}")
            print(f"    Got: {r['answer'][:100]}...")
    
    # Save detailed results
    output_file = Path("question_test_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed results saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    test_all_questions()
