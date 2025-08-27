#!/usr/bin/env python3
"""
Prompt Optimization Script for RAG System

This script runs RAGAS evaluation on the Q&A dataset to optimize prompts
and achieve better performance scores without launching the Gradio interface.

CONFIGURATION:
    Edit the variables at the top of this file to change defaults:
    - DEFAULT_QUESTIONS: Number of questions to evaluate
    - DEFAULT_ITERATIONS: Number of optimization iterations
    - DEFAULT_TARGET_SCORE: Target overall score to achieve

Usage:
    python prompt_optimizer.py [--questions N] [--iterations N] [--target-score X]
    
    Or just run: python prompt_optimizer.py (uses defaults from configuration)
"""

# =============================================================================
# USER CONFIGURATION - CHANGE THESE VALUES HERE
# =============================================================================
DEFAULT_QUESTIONS = 40          # Number of questions to evaluate
DEFAULT_ITERATIONS = 0          # Number of optimization iterations (single run for testing)
DEFAULT_TARGET_SCORE = 0.85     # Target overall score to achieve (slightly lowered for realism)
# =============================================================================

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import time

# Import the RAG pipeline components
from RAG.app.pipeline import run_evaluation, build_pipeline, _discover_input_paths, _clean_run_outputs
from RAG.app.eval_ragas import run_eval_detailed, pretty_metrics, TARGETS
from RAG.app.prompts import NEEDLE_SYSTEM, NEEDLE_PROMPT, TABLE_SYSTEM, TABLE_PROMPT
from dotenv import load_dotenv


class PromptOptimizer:
    def __init__(self, max_questions: int = DEFAULT_QUESTIONS, target_score: float = DEFAULT_TARGET_SCORE):
        self.max_questions = max_questions
        self.target_score = target_score
        self.best_scores = {}
        self.optimization_history = []
        
        # Load environment variables
        load_dotenv(override=True)
        
        # Set logging to reduce verbose output
        import logging
        logging.getLogger().setLevel(logging.WARNING)
        
        print("🚀 Initializing Prompt Optimizer...")
        
        # Build the RAG pipeline
        self._build_pipeline()
        
    def _build_pipeline(self):
        """Build the RAG pipeline without launching UI."""
        print("📚 Building RAG pipeline...")
        
        # Clean outputs and discover paths
        _clean_run_outputs()
        paths = _discover_input_paths()
        
        if not paths:
            print("❌ No input files found. Place PDFs/DOCs under data/ or the root PDF.")
            return
            
        # Build pipeline (docs, hybrid, llm)
        self.docs, self.hybrid, self.debug = build_pipeline(paths)
        from RAG.app.pipeline import _LLM
        self.llm = _LLM()
        
        print(f"✅ Pipeline built: {len(self.docs)} documents loaded")
        
    def run_baseline_evaluation(self) -> Dict[str, float]:
        """Run baseline evaluation with current prompts."""
        print("\n📊 Running baseline evaluation...")
        
        # Set environment to run evaluation without UI
        os.environ["RAG_EVAL"] = "1"
        os.environ["RAG_HEADLESS"] = "1"
        
        try:
            # Run evaluation using the existing pipeline
            run_evaluation(self.docs, self.hybrid, self.llm)
            
            # Load the evaluation results
            eval_file = Path("logs/eval_ragas_summary.json")
            if eval_file.exists():
                with open(eval_file, 'r', encoding='utf-8') as f:
                    baseline_scores = json.load(f)
                print("✅ Baseline evaluation completed")
                return baseline_scores
            else:
                print("⚠️  No evaluation results found")
                return {}
                
        except Exception as e:
            print(f"❌ Baseline evaluation failed: {e}")
            return {}
    
    def optimize_prompts(self, iterations: int = 3) -> Dict[str, Any]:
        """Run prompt optimization iterations."""
        print(f"\n🎯 Starting prompt optimization ({iterations} iterations)...")
        
        # Get baseline scores
        baseline = self.run_baseline_evaluation()
        print(f"📈 Baseline scores: {baseline}")
        
        best_overall_score = 0.0
        best_prompts = {
            "needle_system": NEEDLE_SYSTEM,
            "needle_prompt": NEEDLE_PROMPT,
            "table_system": TABLE_SYSTEM,
            "table_prompt": TABLE_PROMPT
        }
        
        for iteration in range(iterations):
            print(f"\n🔄 Iteration {iteration + 1}/{iterations}")
            
            # Generate optimized prompts based on current performance
            optimized_prompts = self._generate_optimized_prompts(baseline, iteration)
            
            # Test the optimized prompts
            scores = self._test_prompts(optimized_prompts)
            
            # Calculate overall score (weighted average)
            overall_score = self._calculate_overall_score(scores)
            
            print(f"📊 Iteration {iteration + 1} scores:")
            for metric, score in scores.items():
                if score is not None and not (isinstance(score, float) and score != score):  # Not NaN
                    print(f"   {metric}: {score:.3f}")
            print(f"   Overall Score: {overall_score:.3f}")
            
            # Update best if improved
            if overall_score > best_overall_score:
                best_overall_score = overall_score
                best_prompts = optimized_prompts.copy()
                print(f"🎉 New best score: {overall_score:.3f}")
            
            # Update baseline for next iteration
            baseline = scores
            
            # Save optimization history
            self.optimization_history.append({
                "iteration": iteration + 1,
                "scores": scores,
                "overall_score": overall_score,
                "prompts": optimized_prompts
            })
            
            # Early stopping if target reached
            if overall_score >= self.target_score:
                print(f"🎯 Target score {self.target_score} reached!")
                break
                
            time.sleep(1)  # Brief pause between iterations
        
        # Save final results
        self._save_optimization_results(best_prompts, best_overall_score)
        
        return {
            "best_prompts": best_prompts,
            "best_score": best_overall_score,
            "history": self.optimization_history
        }
    
    def _generate_optimized_prompts(self, current_scores: Dict[str, float], iteration: int) -> Dict[str, str]:
        """Generate optimized prompts based on current performance."""
        
        # Analyze which metrics need improvement
        needs_improvement = {}
        for metric, target in TARGETS.items():
            current = current_scores.get(metric)
            if current is not None and current < target:
                needs_improvement[metric] = target - current
        
        print(f"🔧 Areas needing improvement: {list(needs_improvement.keys())}")
        
        # Generate optimized prompts
        optimized = {
            "needle_system": NEEDLE_SYSTEM,
            "needle_prompt": NEEDLE_PROMPT,
            "table_system": TABLE_SYSTEM,
            "table_prompt": TABLE_PROMPT
        }
        
        # Optimize based on specific weaknesses
        if "faithfulness" in needs_improvement:
            optimized["needle_system"] = self._optimize_faithfulness_prompt(NEEDLE_SYSTEM, iteration)
            optimized["needle_prompt"] = self._optimize_faithfulness_prompt(NEEDLE_PROMPT, iteration)
            
        if "context_precision" in needs_improvement:
            # Use date-specific optimization for date-related questions
            optimized["needle_system"] = self._optimize_date_prompt(optimized["needle_system"], iteration)
            optimized["needle_prompt"] = self._optimize_date_prompt(optimized["needle_prompt"], iteration)
            
        if "context_recall" in needs_improvement:
            optimized["needle_system"] = self._optimize_recall_prompt(optimized["needle_system"], iteration)
            
        if "answer_relevancy" in needs_improvement:
            optimized["needle_system"] = self._optimize_relevancy_prompt(optimized["needle_system"], iteration)
            optimized["needle_prompt"] = self._optimize_relevancy_prompt(optimized["needle_prompt"], iteration)
            
        return optimized
    
    def _optimize_faithfulness_prompt(self, prompt: str, iteration: int) -> str:
        """Optimize prompt for better faithfulness to context."""
        enhancements = [
            "FAITHFULNESS: Base your answer EXCLUSIVELY on the provided context.",
            "MANDATORY: Include specific citations [filename pX] for every piece of information you provide.",
            "VERIFICATION: Double-check each statement against the context before including it in your answer.",
            "PRECISION: Use exact quotes and precise values from the context whenever possible.",
            "BOUNDARIES: Do not infer, extrapolate, or add information not explicitly stated in the context.",
            "CONTEXT ONLY: Use ONLY information from the provided context - no external knowledge.",
            "EXACT MATCHING: Use exact terms, numbers, and values as they appear in the context.",
            "NO INFERENCE: Do not draw conclusions or make assumptions beyond what is explicitly stated.",
            "CONTEXT VERIFICATION: Verify every statement against the provided context before including it.",
            "COMPLETENESS: Ensure all information in your answer can be traced back to the context."
        ]
        
        if iteration < len(enhancements):
            return prompt + "\n\n" + enhancements[iteration]
        return prompt
    
    def _optimize_precision_prompt(self, prompt: str, iteration: int) -> str:
        """Optimize prompt for better context precision."""
        enhancements = [
            "CONTEXT PRECISION FOCUS: Use the most relevant sections of the context for your answer. Prioritize specific information over general content.",
            "PRECISION FILTERING: Focus on context that directly addresses the question. Use the most specific and relevant information available.",
            "RELEVANCE PRIORITY: If multiple contexts are available, choose the most specific and relevant ones while maintaining completeness.",
            "SELECTIVE EXTRACTION: Be selective about which context elements to include, but ensure the answer is complete and accurate.",
            "DIRECT ANSWER FOCUS: Extract the information that directly answers the question while maintaining context relevance."
        ]
        
        if iteration < len(enhancements):
            return prompt + "\n\n" + enhancements[iteration]
        return prompt
    
    def _optimize_date_prompt(self, prompt: str, iteration: int) -> str:
        """Optimize prompt specifically for date questions."""
        date_enhancements = [
            "DATE SEARCH: Thoroughly search all context for date information in various formats (9 April 2023, April 9, 2023, etc.).",
            "DATE PRIORITY: When multiple dates are mentioned, identify which one specifically answers the question about timing.",
            "INITIAL DATE FOCUS: For 'initial' or 'first' questions, find the earliest occurrence of the event in the timeline.",
            "DATE CONTEXT: Ensure the date found is actually related to the specific event mentioned in the question.",
            "DATE VERIFICATION: Double-check that the date answers the exact question asked, not just any related date."
        ]
        
        if iteration < len(date_enhancements):
            return prompt + "\n\n" + date_enhancements[iteration]
        return prompt
    
    def _optimize_recall_prompt(self, prompt: str, iteration: int) -> str:
        """Optimize prompt for better context recall."""
        enhancements = [
            "COMPREHENSIVE: Ensure you cover all relevant aspects mentioned in the context.",
            "THOROUGHNESS: Review the entire context to avoid missing important information.",
            "COVERAGE: Include all relevant details from the context that answer the question.",
            "COMPLETENESS: Make sure your answer reflects the full scope of relevant information available."
        ]
        
        if iteration < len(enhancements):
            return prompt + "\n\n" + enhancements[iteration]
        return prompt
    
    def _optimize_relevancy_prompt(self, prompt: str, iteration: int) -> str:
        """Optimize prompt for better answer relevancy."""
        enhancements = [
            "ANSWER RELEVANCY: Ensure your answer directly addresses the specific question asked.",
            "QUESTION FOCUS: Stay focused on what the question is asking for, avoid tangential information.",
            "DIRECT RESPONSE: Provide a direct answer to the question without unnecessary elaboration.",
            "RELEVANCE CHECK: Before including information, verify it directly answers the question.",
            "CONCISE ANSWERS: Keep answers concise and relevant to the specific question.",
            "SPECIFICITY: Be specific and precise in your answer, avoid vague responses.",
            "QUESTION MATCHING: Ensure your answer matches the exact question format and intent.",
            "FOCUSED CONTENT: Only include information that directly relates to the question.",
            "CLEAR STRUCTURE: Structure your answer to directly address the question components.",
            "PRECISION: Use exact values and terms that match the question requirements."
        ]
        
        if iteration < len(enhancements):
            return prompt + "\n\n" + enhancements[iteration]
        return prompt
    
    def _test_prompts(self, prompts: Dict[str, str]) -> Dict[str, float]:
        """Test the given prompts by temporarily replacing them and running evaluation."""
        
        # Temporarily replace prompts
        original_prompts = {
            "needle_system": NEEDLE_SYSTEM,
            "needle_prompt": NEEDLE_PROMPT,
            "table_system": TABLE_SYSTEM,
            "table_prompt": TABLE_PROMPT
        }
        
        try:
            # Replace prompts in the module
            import RAG.app.prompts as prompts_module
            prompts_module.NEEDLE_SYSTEM = prompts["needle_system"]
            prompts_module.NEEDLE_PROMPT = prompts["needle_prompt"]
            prompts_module.TABLE_SYSTEM = prompts["table_system"]
            prompts_module.TABLE_PROMPT = prompts["table_prompt"]
            
            # Run evaluation with new prompts
            os.environ["RAG_EVAL"] = "1"
            os.environ["RAG_HEADLESS"] = "1"
            
            run_evaluation(self.docs, self.hybrid, self.llm)
            
            # Load results
            eval_file = Path("logs/eval_ragas_summary.json")
            if eval_file.exists():
                with open(eval_file, 'r', encoding='utf-8') as f:
                    scores = json.load(f)
                return scores
            else:
                return {}
                
        finally:
            # Restore original prompts
            prompts_module.NEEDLE_SYSTEM = original_prompts["needle_system"]
            prompts_module.NEEDLE_PROMPT = original_prompts["needle_prompt"]
            prompts_module.TABLE_SYSTEM = original_prompts["table_system"]
            prompts_module.TABLE_PROMPT = original_prompts["table_prompt"]
    
    def _calculate_overall_score(self, scores: Dict[str, float]) -> float:
        """Calculate weighted overall score from individual metrics."""
        weights = {
            "faithfulness": 0.3,
            "answer_relevancy": 0.2,
            "context_precision": 0.25,
            "context_recall": 0.25
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            score = scores.get(metric)
            if score is not None and not (isinstance(score, float) and score != score):  # Not NaN
                total_score += score * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _save_optimization_results(self, best_prompts: Dict[str, str], best_score: float):
        """Save optimization results to files."""
        
        # Create results directory
        results_dir = Path("prompt_optimization_results")
        results_dir.mkdir(exist_ok=True)
        
        # Save best prompts
        prompts_file = results_dir / "best_prompts.py"
        with open(prompts_file, 'w', encoding='utf-8') as f:
            f.write("# Optimized Prompts Generated by Prompt Optimizer\n\n")
            f.write(f"# Best Overall Score: {best_score:.3f}\n")
            f.write(f"# Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f'NEEDLE_SYSTEM = """{best_prompts["needle_system"]}"""\n\n')
            f.write(f'NEEDLE_PROMPT = """{best_prompts["needle_prompt"]}"""\n\n')
            f.write(f'TABLE_SYSTEM = """{best_prompts["table_system"]}"""\n\n')
            f.write(f'TABLE_PROMPT = """{best_prompts["table_prompt"]}"""\n')
        
        # Save optimization history
        history_file = results_dir / "optimization_history.json"
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(self.optimization_history, f, indent=2, ensure_ascii=False)
        
        # Save summary report
        summary_file = results_dir / "optimization_summary.md"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"# Prompt Optimization Summary\n\n")
            f.write(f"**Best Overall Score:** {best_score:.3f}\n\n")
            f.write(f"**Target Score:** {self.target_score}\n\n")
            f.write(f"**Iterations Completed:** {len(self.optimization_history)}\n\n")
            
            f.write("## Optimization History\n\n")
            for i, iteration in enumerate(self.optimization_history):
                f.write(f"### Iteration {iteration['iteration']}\n")
                f.write(f"- Overall Score: {iteration['overall_score']:.3f}\n")
                for metric, score in iteration['scores'].items():
                    if score is not None and not (isinstance(score, float) and score != score):
                        f.write(f"- {metric}: {score:.3f}\n")
                f.write("\n")
        
        print(f"\n💾 Results saved to: {results_dir}")
        print(f"   📄 Best prompts: {prompts_file}")
        print(f"   📊 History: {history_file}")
        print(f"   📋 Summary: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Prompt Optimization for RAG System")
    parser.add_argument("--questions", type=int, default=DEFAULT_QUESTIONS, 
                       help=f"Maximum number of questions to evaluate (default: {DEFAULT_QUESTIONS})")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS,
                       help=f"Number of optimization iterations (default: {DEFAULT_ITERATIONS})")
    parser.add_argument("--target-score", type=float, default=DEFAULT_TARGET_SCORE,
                       help=f"Target overall score to achieve (default: {DEFAULT_TARGET_SCORE})")
    
    args = parser.parse_args()
    
    print("🎯 RAG Prompt Optimizer")
    print("=" * 50)
    print(f"📝 Max questions: {args.questions}")
    print(f"🔄 Iterations: {args.iterations}")
    print(f"🎯 Target score: {args.target_score}")
    print("=" * 50)
    
    # Set environment variables for evaluation
    os.environ["RAG_EVAL"] = "1"
    os.environ["RAG_HEADLESS"] = "1"
    
    # Limit questions if specified
    if args.questions > 0:
        # Create a temporary limited dataset
        qa_file = Path("RAG/data/gear_wear_qa.jsonl")
        if qa_file.exists():
            # Read all questions
            with open(qa_file, 'r', encoding='utf-8') as f:
                all_questions = [json.loads(line) for line in f if line.strip()]
            
            # Take only the first N questions
            limited_questions = all_questions[:args.questions]
            
            # Create temporary file
            temp_file = Path("temp_limited_qa.jsonl")
            with open(temp_file, 'w', encoding='utf-8') as f:
                for q in limited_questions:
                    f.write(json.dumps(q, ensure_ascii=False) + '\n')
            
            # Set environment to use temporary file
            os.environ["RAG_QA_FILE"] = str(temp_file)
            
            print(f"📋 Using first {args.questions} questions for evaluation")
    
    try:
        # Run optimization
        optimizer = PromptOptimizer(max_questions=args.questions, target_score=args.target_score)
        results = optimizer.optimize_prompts(iterations=args.iterations)
        
        print("\n🎉 Optimization completed!")
        print(f"🏆 Best overall score: {results['best_score']:.3f}")
        
        if results['best_score'] >= args.target_score:
            print(f"✅ Target score {args.target_score} achieved!")
        else:
            print(f"⚠️  Target score {args.target_score} not reached. Consider more iterations.")
            
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up temporary file
        temp_file = Path("temp_limited_qa.jsonl")
        if temp_file.exists():
            temp_file.unlink()


if __name__ == "__main__":
    main()
