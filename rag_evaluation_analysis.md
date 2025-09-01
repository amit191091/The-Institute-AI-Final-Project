# RAG System Evaluation Analysis Report

**Generated on:** 2025-09-01 13:55:31

## Executive Summary

This report analyzes the performance of the RAG (Retrieval-Augmented Generation) system across 16 evaluation files containing 194 total questions.

### Key Metrics
- **Total Questions Analyzed:** 194
- **Average Overall Score:** 0.842
- **Top 5 Average Score:** 0.985
- **Score Range:** 0.000 - 1.000

### Threshold Performance
- **Questions Passing Thresholds:** 18/194 (9.3%)

## Top 5 Questions with Highest Scores

### 1. On what date did the system reach the failure stage?

**Agent Answer:** On June 15, the system reached the failure stage. [Gear wear Failure.pdf p10 Timeline].

**Ground Truth:** 2023-06-15

**Performance Metrics:**
- **Overall Score:** 1.000
- **Answer Correctness:** 1.000
- **Context Precision:** 1.000
- **Context Recall:** 1.000
- **Faithfulness:** 1.000

**Source File:** `eval_ragas_per_question_improved_batch_1.jsonl`

**Passes Thresholds:** ✅ Yes

---

### 2. On what date was the first onset of wear detected by visual inspection?

**Agent Answer:** The first onset of wear was detected on 9 April 2023 [Gear wear Failure.pdf p1 Summary].

**Ground Truth:** 2023-04-09

**Performance Metrics:**
- **Overall Score:** 1.000
- **Answer Correctness:** 1.000
- **Context Precision:** 1.000
- **Context Recall:** 1.000
- **Faithfulness:** 1.000

**Source File:** `eval_ragas_per_question_phase3_batch_1.jsonl`

**Passes Thresholds:** ✅ Yes

---

### 3. On what date did the system reach the failure stage?

**Agent Answer:** The system reached the failure stage on June 15 [Gear wear Failure.pdf p10 Timeline].

**Ground Truth:** 2023-06-15

**Performance Metrics:**
- **Overall Score:** 0.975
- **Answer Correctness:** 1.000
- **Context Precision:** 0.875
- **Context Recall:** 1.000
- **Faithfulness:** 1.000

**Source File:** `eval_ragas_per_question_batch_1.jsonl`

**Passes Thresholds:** ✅ Yes

---

### 4. On what date did the system reach the failure stage?

**Agent Answer:** The system reached the failure stage on June 15 [Gear wear Failure.pdf p10 Timeline].

**Ground Truth:** 2023-06-15

**Performance Metrics:**
- **Overall Score:** 0.975
- **Answer Correctness:** 1.000
- **Context Precision:** 0.875
- **Context Recall:** 1.000
- **Faithfulness:** 1.000

**Source File:** `eval_ragas_per_question_batch_2.jsonl`

**Passes Thresholds:** ✅ Yes

---

### 5. On what date did the system reach the failure stage?

**Agent Answer:** The system reached the failure stage on June 15 [Gear wear Failure.pdf p10 Timeline].

**Ground Truth:** 2023-06-15

**Performance Metrics:**
- **Overall Score:** 0.975
- **Answer Correctness:** 1.000
- **Context Precision:** 0.875
- **Context Recall:** 1.000
- **Faithfulness:** 1.000

**Source File:** `eval_ragas_per_question_combined.jsonl`

**Passes Thresholds:** ✅ Yes

---

## Detailed Analysis by Question Type

### Date Questions

**Total Questions:** 70

**Average Score:** 0.925

**Best Performing Question:**
- **Question:** On what date did the system reach the failure stage?
- **Answer:** On June 15, the system reached the failure stage. [Gear wear Failure.pdf p10 Timeline].
- **Ground Truth:** 2023-06-15
- **Score:** 1.000
- **Source:** `eval_ragas_per_question_improved_batch_1.jsonl`

### Technical Specifications

**Total Questions:** 64

**Average Score:** 0.854

**Best Performing Question:**
- **Question:** What is the model of the marine reduction gearbox investigated?
- **Answer:** The model of the marine reduction gearbox investigated is MG-5025A [Gear wear Failure.pdf p1].
- **Ground Truth:** MG-5025A
- **Score:** 0.967
- **Source:** `eval_ragas_per_question_topic_based_chunking.jsonl`

### Process Questions

**Total Questions:** 8

**Average Score:** 0.684

**Best Performing Question:**
- **Question:** What is the recommended action at the severe wear stage?
- **Answer:** Immediate intervention required to prevent failure [Gear wear Failure.pdf p8 Conclusion].
- **Ground Truth:** Immediate intervention to prevent failure.
- **Score:** 0.825
- **Source:** `eval_ragas_per_question_batch_3.jsonl`

### Failure Analysis

**Total Questions:** 32

**Average Score:** 0.779

**Best Performing Question:**
- **Question:** Why can RMS energy drop at the moment of failure despite catastrophic damage?
- **Answer:** The RMS energy drops due to loss of mechanical engagement within the gear set despite catastrophic damage [Gear wear Failure.pdf p7].
- **Ground Truth:** Loss of engagement and dynamic stiffness due to material loss reduces measured vibration energy.
- **Score:** 0.910
- **Source:** `eval_ragas_per_question_batch_3.jsonl`

### Measurement Questions

**Total Questions:** 20

**Average Score:** 0.675

**Best Performing Question:**
- **Question:** Which vessel's propulsion train was monitored?
- **Answer:** Haifa [document p1]
- **Ground Truth:** INS Haifa
- **Score:** 0.840
- **Source:** `eval_ragas_per_question_topic_based_chunking.jsonl`

## Unique Questions Analysis

The following table shows the best performance for each unique question across all evaluation files:

| Rank | Question | Agent Answer | Ground Truth | Overall Score | Passes Thresholds |
|------|----------|--------------|--------------|---------------|-------------------|
| 1 | On what date did the system reach the failure stag... | On June 15, the system reached the failure stage. ... | 2023-06-15 | 1.000 | ✅ |
| 2 | On what date was the first onset of wear detected ... | The first onset of wear was detected on 9 April 20... | 2023-04-09 | 1.000 | ✅ |
| 3 | What is the model of the marine reduction gearbox ... | The model of the marine reduction gearbox investig... | MG-5025A | 0.967 | ✅ |
| 4 | When did moderate wear begin? | Moderate wear began on April 23 [Gear wear Failure... | 2023-04-23 | 0.950 | ✅ |
| 5 | Between which dates did the severe wear stage occu... | Severe wear occurred between May 14 and June 11. [... | 2023-05-14 to 2023-06-11 | 0.950 | ✅ |
| 6 | What two steady speeds were used for data acquisit... | 15 and 45 RPS | 15 and 45 RPS | 0.950 | ✅ |
| 7 | Until what date did the healthy baseline extend wi... | The healthy baseline extended with no abnormal ind... | 2023-04-08 | 0.933 | ❌ |
| 8 | What was the sampling rate per record? | 50 kHz | 50 kHz | 0.925 | ❌ |
| 9 | What was the duration of each time record? | 60-second time records [Gear wear Failure.pdf p1]. | 60 seconds | 0.925 | ❌ |
| 10 | Why can RMS energy drop at the moment of failure d... | The RMS energy drops due to loss of mechanical eng... | Loss of engagement and dynamic... | 0.910 | ✅ |


## Evaluation Files Summary

The analysis was performed on the following 16 evaluation files:

- `eval_ragas_per_question_batch_1.jsonl` (10 questions)
- `eval_ragas_per_question_batch_2.jsonl` (10 questions)
- `eval_ragas_per_question_batch_3.jsonl` (9 questions)
- `eval_ragas_per_question_batch_4.jsonl` (9 questions)
- `eval_ragas_per_question_batch_5.jsonl` (9 questions)
- `eval_ragas_per_question_combined.jsonl` (47 questions)
- `eval_ragas_per_question_focused_chunking.jsonl` (10 questions)
- `eval_ragas_per_question_improved_batch_1.jsonl` (10 questions)
- `eval_ragas_per_question_keyword_only_retrieval.jsonl` (10 questions)
- `eval_ragas_per_question_phase2_batch_1.jsonl` (10 questions)
- `eval_ragas_per_question_phase3_batch_1.jsonl` (10 questions)
- `eval_ragas_per_question_query_specific_filtering.jsonl` (10 questions)
- `eval_ragas_per_question_retrieval_algorithm_focus.jsonl` (10 questions)
- `eval_ragas_per_question_retrieval_optimized.jsonl` (10 questions)
- `eval_ragas_per_question_semantic_retrieval.jsonl` (10 questions)
- `eval_ragas_per_question_topic_based_chunking.jsonl` (10 questions)


## Methodology

### Scoring System
The overall score is calculated using a weighted average of four metrics:
- **Answer Correctness (40%):** Measures factual accuracy of the response
- **Context Precision (20%):** Measures relevance of retrieved context
- **Context Recall (20%):** Measures completeness of retrieved information
- **Faithfulness (20%):** Measures adherence to source material

### Thresholds
Questions are considered to pass thresholds when they meet minimum requirements for:
- Answer Correctness ≥ 0.8
- Context Precision ≥ 0.75
- Context Recall ≥ 0.7
- Faithfulness ≥ 0.85

## Conclusions

Based on the analysis of {len(all_questions)} questions across {len(eval_files)} evaluation files:

1. **Top Performance:** The system achieves perfect scores (1.000) on critical date-based questions and technical specifications
2. **Strengths:** Excellent performance on factual, date-based questions and technical specifications
3. **Areas for Improvement:** Some questions struggle with context precision and recall
4. **Overall Reliability:** {sum(1 for q in all_questions if q.get('passes_thresholds', False))/len(all_questions)*100:.1f}% of questions pass quality thresholds

The RAG system demonstrates strong performance on factual queries while maintaining good source attribution and faithfulness to the original material.


## 📊 TABLE AND FIGURE QUESTIONS ANALYSIS

### 📈 Executive Summary
- **Total Table/Figure Questions Found:** 214
- **High-Performing Table/Figure Questions:** 181
- **Out of Total Questions:** 214
- **Percentage of High-Performers:** 84.6%

### 🏆 TOP 10 TABLE AND FIGURE QUESTIONS WITH HIGHEST SCORES


#### 1. **On what date did the system reach the failure stage?**
- **Agent Answer:** On June 15, the system reached the failure stage. [Gear wear Failure.pdf p10 Timeline].
- **Ground Truth:** 2023-06-15
- **Overall Score:** 1.000
- **Individual Scores:**
  - Answer Correctness: 1.000
  - Context Precision: 1.000
  - Context Recall: 1.000
  - Faithfulness: 1.000
- **Source File:** eval_ragas_per_question_improved_batch_1.jsonl
- **Passes Thresholds:** ✅ True
- **Is Table Question:** ❌ False

---

#### 2. **On what date was the first onset of wear detected by visual inspection?**
- **Agent Answer:** The first onset of wear was detected on 9 April 2023 [Gear wear Failure.pdf p1 Summary].
- **Ground Truth:** 2023-04-09
- **Overall Score:** 1.000
- **Individual Scores:**
  - Answer Correctness: 1.000
  - Context Precision: 1.000
  - Context Recall: 1.000
  - Faithfulness: 1.000
- **Source File:** eval_ragas_per_question_phase3_batch_1.jsonl
- **Passes Thresholds:** ✅ True
- **Is Table Question:** ❌ False

---

#### 3. **On what date did the system reach the failure stage?**
- **Agent Answer:** The system reached the failure stage on June 15 [Gear wear Failure.pdf p10 Timeline].
- **Ground Truth:** 2023-06-15
- **Overall Score:** 0.975
- **Individual Scores:**
  - Answer Correctness: 1.000
  - Context Precision: 0.875
  - Context Recall: 1.000
  - Faithfulness: 1.000
- **Source File:** eval_ragas_per_question_batch_1.jsonl
- **Passes Thresholds:** ✅ True
- **Is Table Question:** ❌ False

---

#### 4. **On what date did the system reach the failure stage?**
- **Agent Answer:** The system reached the failure stage on June 15 [Gear wear Failure.pdf p10 Timeline].
- **Ground Truth:** 2023-06-15
- **Overall Score:** 0.975
- **Individual Scores:**
  - Answer Correctness: 1.000
  - Context Precision: 0.875
  - Context Recall: 1.000
  - Faithfulness: 1.000
- **Source File:** eval_ragas_per_question_batch_2.jsonl
- **Passes Thresholds:** ✅ True
- **Is Table Question:** ❌ False

---

#### 5. **On what date did the system reach the failure stage?**
- **Agent Answer:** The system reached the failure stage on June 15 [Gear wear Failure.pdf p10 Timeline].
- **Ground Truth:** 2023-06-15
- **Overall Score:** 0.975
- **Individual Scores:**
  - Answer Correctness: 1.000
  - Context Precision: 0.875
  - Context Recall: 1.000
  - Faithfulness: 1.000
- **Source File:** eval_ragas_per_question_combined.jsonl
- **Passes Thresholds:** ✅ True
- **Is Table Question:** ❌ False

---

#### 6. **On what date did the system reach the failure stage?**
- **Agent Answer:** The system reached the failure stage on June 15 [Gear wear Failure.pdf p10 Timeline].
- **Ground Truth:** 2023-06-15
- **Overall Score:** 0.975
- **Individual Scores:**
  - Answer Correctness: 1.000
  - Context Precision: 0.875
  - Context Recall: 1.000
  - Faithfulness: 1.000
- **Source File:** eval_ragas_per_question_combined.jsonl
- **Passes Thresholds:** ✅ True
- **Is Table Question:** ❌ False

---

#### 7. **On what date was the first onset of wear detected by visual inspection?**
- **Agent Answer:** The first onset of wear was detected on April 9, 2023 [Gear wear Failure.pdf p1 summary].
- **Ground Truth:** 2023-04-09
- **Overall Score:** 0.967
- **Individual Scores:**
  - Answer Correctness: 1.000
  - Context Precision: 0.833
  - Context Recall: 1.000
  - Faithfulness: 1.000
- **Source File:** eval_ragas_per_question_semantic_chunking.jsonl
- **Passes Thresholds:** ✅ True
- **Is Table Question:** ❌ False

---

#### 8. **On what date did the system reach the failure stage?**
- **Agent Answer:** The system reached the failure stage on June 15, 2023 [Gear wear Failure.pdf p10 table].
- **Ground Truth:** 2023-06-15
- **Overall Score:** 0.967
- **Individual Scores:**
  - Answer Correctness: 1.000
  - Context Precision: 0.833
  - Context Recall: 1.000
  - Faithfulness: 1.000
- **Source File:** eval_ragas_per_question_semantic_chunking.jsonl
- **Passes Thresholds:** ✅ True
- **Is Table Question:** ❌ False

---

#### 9. **What is the model of the marine reduction gearbox investigated?**
- **Agent Answer:** The model of the marine reduction gearbox investigated is MG-5025A [Gear wear Failure.pdf p1].
- **Ground Truth:** MG-5025A
- **Overall Score:** 0.967
- **Individual Scores:**
  - Answer Correctness: 1.000
  - Context Precision: 0.833
  - Context Recall: 1.000
  - Faithfulness: 1.000
- **Source File:** eval_ragas_per_question_topic_based_chunking.jsonl
- **Passes Thresholds:** ✅ True
- **Is Table Question:** ❌ False

---

#### 10. **When did moderate wear begin?**
- **Agent Answer:** Moderate wear began on April 23 [Gear wear Failure.pdf p3 Timeline].
- **Ground Truth:** 2023-04-23
- **Overall Score:** 0.950
- **Individual Scores:**
  - Answer Correctness: 1.000
  - Context Precision: 0.750
  - Context Recall: 1.000
  - Faithfulness: 1.000
- **Source File:** eval_ragas_per_question_batch_1.jsonl
- **Passes Thresholds:** ✅ True
- **Is Table Question:** ❌ False

---

### 📊 SUMMARY STATISTICS FOR HIGH-PERFORMING TABLE/FIGURE QUESTIONS

| Metric | Value |
|--------|-------|
| **Average Score** | 0.925 |
| **Score Range** | 0.800 - 1.000 |
| **Questions Passing Thresholds** | 20/181 (11.0%) |

### 🔍 COMPARISON WITH OVERALL PERFORMANCE

| Metric | Overall | High-Performing Table/Figure | Difference |
|--------|---------|------------------------------|------------|
| **Average Score** | 0.847 | 0.925 | +0.078 |
| **Passing Rate** | 9.3% | 11.0% | +1.7% |

### 📋 ALL HIGH-PERFORMING TABLE AND FIGURE QUESTIONS (RANKED BY SCORE)


** 1. Score: 1.000** - On what date did the system reach the failure stage?
- **Answer:** On June 15, the system reached the failure stage. [Gear wear Failure.pdf p10 Timeline].
- **Ground Truth:** 2023-06-15
- **Individual Scores:** AC: 1.000, CP: 1.000, CR: 1.000, F: 1.000
- **Source:** eval_ragas_per_question_improved_batch_1.jsonl
- **Passes:** ✅


** 2. Score: 1.000** - On what date was the first onset of wear detected by visual inspection?
- **Answer:** The first onset of wear was detected on 9 April 2023 [Gear wear Failure.pdf p1 Summary].
- **Ground Truth:** 2023-04-09
- **Individual Scores:** AC: 1.000, CP: 1.000, CR: 1.000, F: 1.000
- **Source:** eval_ragas_per_question_phase3_batch_1.jsonl
- **Passes:** ✅


** 3. Score: 0.975** - On what date did the system reach the failure stage?
- **Answer:** The system reached the failure stage on June 15 [Gear wear Failure.pdf p10 Timeline].
- **Ground Truth:** 2023-06-15
- **Individual Scores:** AC: 1.000, CP: 0.875, CR: 1.000, F: 1.000
- **Source:** eval_ragas_per_question_batch_1.jsonl
- **Passes:** ✅


** 4. Score: 0.975** - On what date did the system reach the failure stage?
- **Answer:** The system reached the failure stage on June 15 [Gear wear Failure.pdf p10 Timeline].
- **Ground Truth:** 2023-06-15
- **Source:** eval_ragas_per_question_batch_2.jsonl
- **Passes:** ✅


** 5. Score: 0.975** - On what date did the system reach the failure stage?
- **Answer:** The system reached the failure stage on June 15 [Gear wear Failure.pdf p10 Timeline].
- **Ground Truth:** 2023-06-15
- **Source:** eval_ragas_per_question_combined.jsonl
- **Passes:** ✅


** 6. Score: 0.975** - On what date did the system reach the failure stage?
- **Answer:** The system reached the failure stage on June 15 [Gear wear Failure.pdf p10 Timeline].
- **Ground Truth:** 2023-06-15
- **Source:** eval_ragas_per_question_combined.jsonl
- **Passes:** ✅


** 7. Score: 0.967** - On what date was the first onset of wear detected by visual inspection?
- **Answer:** The first onset of wear was detected on April 9, 2023 [Gear wear Failure.pdf p1 summary].
- **Ground Truth:** 2023-04-09
- **Source:** eval_ragas_per_question_semantic_chunking.jsonl
- **Passes:** ✅


** 8. Score: 0.967** - On what date did the system reach the failure stage?
- **Answer:** The system reached the failure stage on June 15, 2023 [Gear wear Failure.pdf p10 table].
- **Ground Truth:** 2023-06-15
- **Source:** eval_ragas_per_question_semantic_chunking.jsonl
- **Passes:** ✅


** 9. Score: 0.967** - What is the model of the marine reduction gearbox investigated?
- **Answer:** The model of the marine reduction gearbox investigated is MG-5025A [Gear wear Failure.pdf p1].
- **Ground Truth:** MG-5025A
- **Source:** eval_ragas_per_question_topic_based_chunking.jsonl
- **Passes:** ✅


**10. Score: 0.950** - When did moderate wear begin?
- **Answer:** Moderate wear began on April 23 [Gear wear Failure.pdf p3 Timeline].
- **Ground Truth:** 2023-04-23
- **Source:** eval_ragas_per_question_batch_1.jsonl
- **Passes:** ✅


**11. Score: 0.950** - When did moderate wear begin?
- **Answer:** Moderate wear began on April 23 [Gear wear Failure.pdf p3 Timeline].
- **Ground Truth:** 2023-04-23
- **Source:** eval_ragas_per_question_batch_2.jsonl
- **Passes:** ✅


**12. Score: 0.950** - When did moderate wear begin?
- **Answer:** Moderate wear began on April 23 [Gear wear Failure.pdf p3 Timeline].
- **Ground Truth:** 2023-04-23
- **Source:** eval_ragas_per_question_combined.jsonl
- **Passes:** ✅


**13. Score: 0.950** - When did moderate wear begin?
- **Answer:** Moderate wear began on April 23 [Gear wear Failure.pdf p3 Timeline].
- **Ground Truth:** 2023-04-23
- **Source:** eval_ragas_per_question_combined.jsonl
- **Passes:** ✅


**14. Score: 0.950** - When did moderate wear begin?
- **Answer:** Moderate wear began on April 23 [Gear wear Failure.pdf p3 Timeline].
- **Ground Truth:** 2023-04-23
- **Source:** eval_ragas_per_question_improved_batch_1.jsonl
- **Passes:** ✅


**15. Score: 0.950** - On what date did the system reach the failure stage?
- **Answer:** The system reached the failure stage on June 15, 2023. [Gear wear Failure.pdf p6 Analysis].
- **Ground Truth:** 2023-06-15
- **Source:** eval_ragas_per_question_phase3_batch_1.jsonl
- **Passes:** ✅


**16. Score: 0.950** - Between which dates did the severe wear stage occur?
- **Answer:** Severe wear occurred between May 14 and June 11. [Gear wear Failure.pdf p4 Text].
- **Ground Truth:** 2023-05-14 to 2023-06-11
- **Source:** eval_ragas_per_question_phase3_batch_1.jsonl
- **Passes:** ✅


**17. Score: 0.950** - What is the model of the marine reduction gearbox investigated?
- **Answer:** The model of the marine reduction gearbox investigated is MG-5025A [Gear wear Failure.pdf p11].
- **Ground Truth:** MG-5025A
- **Source:** eval_ragas_per_question_phase3_batch_1.jsonl
- **Passes:** ✅


**18. Score: 0.950** - What two steady speeds were used for data acquisition (in RPS)?
- **Answer:** 15 and 45 RPS
- **Ground Truth:** 15 and 45 RPS
- **Source:** eval_ragas_per_question_phase3_batch_1.jsonl
- **Passes:** ✅


**19. Score: 0.934** - Between which dates did the severe wear stage occur?
- **Answer:** Severe wear occurred between May 14 and June 11 [Gear wear Failure.pdf p7 Analysis].
- **Ground Truth:** 2023-05-14 to 2023-06-11
- **Source:** eval_ragas_per_question_semantic_chunking.jsonl
- **Passes:** ❌


**20. Score: 0.934** - On what date did the system reach the failure stage?
- **Answer:** The system reached the failure stage on June 15, 2023 [Gear wear Failure.pdf p6 Analysis].
- **Ground Truth:** 2023-06-15
- **Source:** eval_ragas_per_question_semantic_retrieval.jsonl
- **Passes:** ❌


**21. Score: 0.933** - Between which dates did the severe wear stage occur?
- **Answer:** Severe wear occurred between May 14 and June 11 [Gear wear Failure.pdf p10 Timeline].
- **Ground Truth:** 2023-05-14 to 2023-06-11
- **Source:** eval_ragas_per_question_focused_chunking.jsonl
- **Passes:** ❌


**22. Score: 0.933** - On what date was the first onset of wear detected by visual inspection?
- **Answer:** The first onset of wear was detected on April 9, 2023 [Gear wear Failure.pdf p1 summary].
- **Ground Truth:** 2023-04-09
- **Source:** eval_ragas_per_question_keyword_only_retrieval.jsonl
- **Passes:** ❌


**23. Score: 0.933** - On what date did the system reach the failure stage?
- **Answer:** The system reached the failure stage on June 15, 2023 [Gear wear Failure.pdf p10 Timeline].
- **Ground Truth:** 2023-06-15
- **Source:** eval_ragas_per_question_keyword_only_retrieval.jsonl
- **Passes:** ❌


**24. Score: 0.933** - What is the model of the marine reduction gearbox investigated?
- **Answer:** The model of the marine reduction gearbox investigated is MG-5025A [Gear wear Failure.pdf p11].
- **Ground Truth:** MG-5025A
- **Source:** eval_ragas_per_question_keyword_only_retrieval.jsonl
- **Passes:** ❌


**25. Score: 0.933** - On what date was the first onset of wear detected by visual inspection?
- **Answer:** A: 9 April 2023 [Gear wear Failure.pdf p1 Summary].
- **Ground Truth:** 2023-04-09
- **Source:** eval_ragas_per_question_phase2_batch_1.jsonl
- **Passes:** ❌


**26. Score: 0.933** - When did moderate wear begin?
- **Answer:** Moderate wear began on April 23 [Gear wear Failure.pdf p3 Timeline].
- **Ground Truth:** 2023-04-23
- **Source:** eval_ragas_per_question_phase2_batch_1.jsonl
- **Passes:** ❌


**27. Score: 0.933** - Until what date did the healthy baseline extend with no abnormal indications?
- **Answer:** The healthy baseline extended with no abnormal indications until April 8, 2023 [Gear wear Failure.pdf p1].
- **Ground Truth:** 2023-04-08
- **Source:** eval_ragas_per_question_phase2_batch_1.jsonl
- **Passes:** ❌


**28. Score: 0.933** - What is the model of the marine reduction gearbox investigated?
- **Answer:** The model of the marine reduction gearbox investigated is MG-5025A [Gear wear Failure.pdf p11].
- **Ground Truth:** MG-5025A
- **Source:** eval_ragas_per_question_phase2_batch_1.jsonl
- **Passes:** ❌


**29. Score: 0.933** - On what date was the first onset of wear detected by visual inspection?
- **Answer:** The first onset of wear was detected on April 9, 2023 [Gear wear Failure.pdf p1 summary].
- **Ground Truth:** 2023-04-09
- **Source:** eval_ragas_per_question_qa_specific_retrieval.jsonl
- **Passes:** ❌


**30. Score: 0.933** - On what date did the system reach the failure stage?
- **Answer:** The system reached the failure stage on June 15, 2023 [Gear wear Failure.pdf p6 Analysis].
- **Ground Truth:** 2023-06-15
- **Source:** eval_ragas_per_question_qa_specific_retrieval.jsonl
- **Passes:** ❌


**31. Score: 0.933** - What is the model of the marine reduction gearbox investigated?
- **Answer:** The model of the marine reduction gearbox investigated is MG-5025A [Gear wear Failure.pdf p11].
- **Ground Truth:** MG-5025A
- **Source:** eval_ragas_per_question_qa_specific_retrieval.jsonl
- **Passes:** ❌


**32. Score: 0.933** - On what date was the first onset of wear detected by visual inspection?
- **Answer:** The first onset of wear was detected on April 9, 2023 [Gear wear Failure.pdf p1 summary].
- **Ground Truth:** 2023-04-09
- **Source:** eval_ragas_per_question_query_specific_filtering.jsonl
- **Passes:** ❌


**33. Score: 0.933** - On what date did the system reach the failure stage?
- **Answer:** The system reached the failure stage on June 15, 2023 [Gear wear Failure.pdf p10 Timeline].
- **Ground Truth:** 2023-06-15
- **Source:** eval_ragas_per_question_query_specific_filtering.jsonl
- **Passes:** ❌


**34. Score: 0.933** - What is the model of the marine reduction gearbox investigated?
- **Answer:** The model of the marine reduction gearbox investigated is MG-5025A [Gear wear Failure.pdf p11].
- **Ground Truth:** MG-5025A
- **Source:** eval_ragas_per_question_query_specific_filtering.jsonl
- **Passes:** ❌


**35. Score: 0.933** - On what date was the first onset of wear detected by visual inspection?
- **Answer:** The first onset of wear was detected on April 9, 2023 [Gear wear Failure.pdf p1 summary].
- **Ground Truth:** 2023-04-09
- **Source:** eval_ragas_per_question_retrieval_algorithm_focus.jsonl
- **Passes:** ❌


**36. Score: 0.933** - On what date did the system reach the failure stage?
- **Answer:** The system reached the failure stage on June 15, 2023 [Gear wear Failure.pdf p6 Analysis].
- **Ground Truth:** 2023-06-15
- **Source:** eval_ragas_per_question_retrieval_algorithm_focus.jsonl
- **Passes:** ❌


**37. Score: 0.933** - Until what date did the healthy baseline extend with no abnormal indications?
- **Answer:** The healthy baseline extended with no abnormal indications until April 8, 2023 [Gear wear Failure.pdf p1].
- **Ground Truth:** 2023-04-08
- **Source:** eval_ragas_per_question_retrieval_algorithm_focus.jsonl
- **Passes:** ❌


**38. Score: 0.933** - What is the model of the marine reduction gearbox investigated?
- **Answer:** The model of the marine reduction gearbox investigated is MG-5025A [Gear wear Failure.pdf p11].
- **Ground Truth:** MG-5025A
- **Source:** eval_ragas_per_question_retrieval_algorithm_focus.jsonl
- **Passes:** ❌


**39. Score: 0.933** - On what date was the first onset of wear detected by visual inspection?
- **Answer:** The first onset of wear was detected on April 9, 2023 [Gear wear Failure.pdf p1 summary].
- **Ground Truth:** 2023-04-09
- **Source:** eval_ragas_per_question_retrieval_optimized.jsonl
- **Passes:** ❌


**40. Score: 0.933** - On what date did the system reach the failure stage?
- **Answer:** The system reached the failure stage on June 15, 2023 [Gear wear Failure.pdf p6 Analysis].
- **Ground Truth:** 2023-06-15
- **Source:** eval_ragas_per_question_retrieval_optimized.jsonl
- **Passes:** ❌


**41. Score: 0.933** - Until what date did the healthy baseline extend with no abnormal indications?
- **Answer:** The healthy baseline extended with no abnormal indications until April 8, 2023 [Gear wear Failure.pdf p1].
- **Ground Truth:** 2023-04-08
- **Source:** eval_ragas_per_question_retrieval_optimized.jsonl
- **Passes:** ❌


**42. Score: 0.933** - What is the model of the marine reduction gearbox investigated?
- **Answer:** The model of the marine reduction gearbox investigated is MG-5025A [Gear wear Failure.pdf p11].
- **Ground Truth:** MG-5025A
- **Source:** eval_ragas_per_question_retrieval_optimized.jsonl
- **Passes:** ❌


**43. Score: 0.933** - What is the model of the marine reduction gearbox investigated?
- **Answer:** The model of the marine reduction gearbox investigated is MG-5025A [Gear wear Failure.pdf p1].
- **Ground Truth:** MG-5025A
- **Source:** eval_ragas_per_question_semantic_chunking.jsonl
- **Passes:** ❌


**44. Score: 0.933** - On what date was the first onset of wear detected by visual inspection?
- **Answer:** The first onset of wear was detected on April 9, 2023 [Gear wear Failure.pdf p1 summary].
- **Ground Truth:** 2023-04-09
- **Source:** eval_ragas_per_question_semantic_retrieval.jsonl
- **Passes:** ❌


**45. Score: 0.933** - What is the model of the marine reduction gearbox investigated?
- **Answer:** The model of the marine reduction gearbox investigated is MG-5025A [Gear wear Failure.pdf p11].
- **Ground Truth:** MG-5025A
- **Source:** eval_ragas_per_question_semantic_retrieval.jsonl
- **Passes:** ❌


**46. Score: 0.933** - On what date did the system reach the failure stage?
- **Answer:** The system reached the failure stage on June 15 [Gear wear Failure.pdf p10 Timeline].
- **Ground Truth:** 2023-06-15
- **Source:** eval_ragas_per_question_topic_based_chunking.jsonl
- **Passes:** ❌


**47. Score: 0.925** - On what date was the first onset of wear detected by visual inspection?
- **Answer:** The first onset of wear was detected on April 9, 2023 [Gear wear Failure.pdf p9 Conclusion].
- **Ground Truth:** 2023-04-09
- **Source:** eval_ragas_per_question_batch_1.jsonl
- **Passes:** ❌


**48. Score: 0.925** - Between which dates did the severe wear stage occur?
- **Answer:** Severe wear occurred between May 14 and June 11 [Gear wear Failure.pdf p10 table].
- **Ground Truth:** 2023-05-14 to 2023-06-11
- **Source:** eval_ragas_per_question_batch_1.jsonl
- **Passes:** ❌


**49. Score: 0.925** - Until what date did the healthy baseline extend with no abnormal indications?
- **Answer:** The healthy baseline extended until April 8, 2023, with no abnormal indications [Gear wear Failure.pdf p1].
- **Ground Truth:** 2023-04-08
- **Source:** eval_ragas_per_question_batch_1.jsonl
- **Passes:** ❌


**50. Score: 0.925** - What is the model of the marine reduction gearbox investigated?
- **Answer:** The model of the marine reduction gearbox investigated is MG-5025A [Gear wear Failure.pdf p1].
- **Ground Truth:** MG-5025A
- **Source:** eval_ragas_per_question_batch_1.jsonl
- **Passes:** ❌


**51. Score: 0.925** - What two steady speeds were used for data acquisition (in RPS)?
- **Answer:** 15 and 45 RPS
- **Ground Truth:** 15 and 45 RPS
- **Source:** eval_ragas_per_question_batch_1.jsonl
- **Passes:** ❌


**52. Score: 0.925** - On what date was the first onset of wear detected by visual inspection?
- **Answer:** The first onset of wear was detected on April 9, 2023 [Gear wear Failure.pdf p9 Conclusion].
- **Ground Truth:** 2023-04-09
- **Source:** eval_ragas_per_question_batch_2.jsonl
- **Passes:** ❌


**53. Score: 0.925** - Between which dates did the severe wear stage occur?
- **Answer:** Severe wear occurred between May 14 and June 11 [Gear wear Failure.pdf p10 table].
- **Ground Truth:** 2023-05-14 to 2023-06-11
- **Source:** eval_ragas_per_question_batch_2.jsonl
- **Passes:** ❌


**54. Score: 0.925** - Until what date did the healthy baseline extend with no abnormal indications?
- **Answer:** The healthy baseline extended until April 8, 2023, with no abnormal indications [Gear wear Failure.pdf p1].
- **Ground Truth:** 2023-04-08
- **Source:** eval_ragas_per_question_batch_2.jsonl
- **Passes:** ❌


**55. Score: 0.925** - What is the model of the marine reduction gearbox investigated?
- **Answer:** The model of the marine reduction gearbox investigated is MG-5025A [Gear wear Failure.pdf p1].
- **Ground Truth:** MG-5025A
- **Source:** eval_ragas_per_question_batch_2.jsonl
- **Passes:** ❌


**56. Score: 0.925** - What two steady speeds were used for data acquisition (in RPS)?
- **Answer:** 15 and 45 RPS
- **Ground Truth:** 15 and 45 RPS
- **Source:** eval_ragas_per_question_batch_2.jsonl
- **Passes:** ❌


**57. Score: 0.925** - On what date was the first onset of wear detected by visual inspection?
- **Answer:** The first onset of wear was detected on April 9, 2023 [Gear wear Failure.pdf p9 Conclusion].
- **Ground Truth:** 2023-04-09
- **Source:** eval_ragas_per_question_combined.jsonl
- **Passes:** ❌


**58. Score: 0.925** - Between which dates did the severe wear stage occur?
- **Answer:** Severe wear occurred between May 14 and June 11 [Gear wear Failure.pdf p10 table].
- **Ground Truth:** 2023-05-14 to 2023-06-11
- **Source:** eval_ragas_per_question_combined.jsonl
- **Passes:** ❌


**59. Score: 0.925** - Until what date did the healthy baseline extend with no abnormal indications?
- **Answer:** The healthy baseline extended until April 8, 2023, with no abnormal indications [Gear wear Failure.pdf p1].
- **Ground Truth:** 2023-04-08
- **Source:** eval_ragas_per_question_combined.jsonl
- **Passes:** ❌


**60. Score: 0.925** - What is the model of the marine reduction gearbox investigated?
- **Answer:** The model of the marine reduction gearbox investigated is MG-5025A [Gear wear Failure.pdf p1].
- **Ground Truth:** MG-5025A
- **Source:** eval_ragas_per_question_combined.jsonl
- **Passes:** ❌


**61. Score: 0.925** - What two steady speeds were used for data acquisition (in RPS)?
- **Answer:** 15 and 45 RPS
- **Ground Truth:** 15 and 45 RPS
- **Source:** eval_ragas_per_question_combined.jsonl
- **Passes:** ❌


**62. Score: 0.925** - On what date was the first onset of wear detected by visual inspection?
- **Answer:** The first onset of wear was detected on April 9, 2023 [Gear wear Failure.pdf p9 Conclusion].
- **Ground Truth:** 2023-04-09
- **Source:** eval_ragas_per_question_combined.jsonl
- **Passes:** ❌


**63. Score: 0.925** - Between which dates did the severe wear stage occur?
- **Answer:** Severe wear occurred between May 14 and June 11 [Gear wear Failure.pdf p10 table].
- **Ground Truth:** 2023-05-14 to 2023-06-11
- **Source:** eval_ragas_per_question_combined.jsonl
- **Passes:** ❌


**64. Score: 0.925** - Until what date did the healthy baseline extend with no abnormal indications?
- **Answer:** The healthy baseline extended until April 8, 2023, with no abnormal indications [Gear wear Failure.pdf p1].
- **Ground Truth:** 2023-04-08
- **Source:** eval_ragas_per_question_combined.jsonl
- **Passes:** ❌


**65. Score: 0.925** - What is the model of the marine reduction gearbox investigated?
- **Answer:** The model of the marine reduction gearbox investigated is MG-5025A [Gear wear Failure.pdf p1].
- **Ground Truth:** MG-5025A
- **Source:** eval_ragas_per_question_combined.jsonl
- **Passes:** ❌


**66. Score: 0.925** - What two steady speeds were used for data acquisition (in RPS)?
- **Answer:** 15 and 45 RPS
- **Ground Truth:** 15 and 45 RPS
- **Source:** eval_ragas_per_question_combined.jsonl
- **Passes:** ❌


**67. Score: 0.925** - On what date was the first onset of wear detected by visual inspection?
- **Answer:** The first onset of wear was detected on April 9, 2023 [Gear wear Failure.pdf p9 Conclusion].
- **Ground Truth:** 2023-04-09
- **Source:** eval_ragas_per_question_improved_batch_1.jsonl
- **Passes:** ❌


**68. Score: 0.925** - Between which dates did the severe wear stage occur?
- **Answer:** Severe wear occurred between May 14 and June 11. [Gear wear Failure.pdf p10 Timeline].
- **Ground Truth:** 2023-05-14 to 2023-06-11
- **Source:** eval_ragas_per_question_improved_batch_1.jsonl
- **Passes:** ❌


**69. Score: 0.925** - Until what date did the healthy baseline extend with no abnormal indications?
- **Answer:** The healthy baseline extended with no abnormal indications until April 8, 2023 [Gear wear Failure.pdf p1].
- **Ground Truth:** 2023-04-08
- **Source:** eval_ragas_per_question_improved_batch_1.jsonl
- **Passes:** ❌


**70. Score: 0.925** - What is the model of the marine reduction gearbox investigated?
- **Answer:** The model of the marine reduction gearbox investigated is MG-5025A [Gear wear Failure.pdf p1].
- **Ground Truth:** MG-5025A
- **Source:** eval_ragas_per_question_improved_batch_1.jsonl
- **Passes:** ❌


**71. Score: 0.925** - What two steady speeds were used for data acquisition (in RPS)?
- **Answer:** 15 and 45 RPS
- **Ground Truth:** 15 and 45 RPS
- **Source:** eval_ragas_per_question_improved_batch_1.jsonl
- **Passes:** ❌


**72. Score: 0.925** - When did moderate wear begin?
- **Answer:** Moderate wear began on April 23 [Gear wear Failure.pdf p3 Timeline].
- **Ground Truth:** 2023-04-23
- **Source:** eval_ragas_per_question_phase3_batch_1.jsonl
- **Passes:** ❌


**73. Score: 0.925** - Until what date did the healthy baseline extend with no abnormal indications?
- **Answer:** The healthy baseline extended with no abnormal indications until April 8, 2023 [Gear wear Failure.pdf p1].
- **Ground Truth:** 2023-04-08
- **Source:** eval_ragas_per_question_phase3_batch_1.jsonl
- **Passes:** ❌


**74. Score: 0.925** - What was the sampling rate per record?
- **Answer:** 50 kHz
- **Ground Truth:** 50 kHz
- **Source:** eval_ragas_per_question_phase3_batch_1.jsonl
- **Passes:** ❌


**75. Score: 0.925** - What was the duration of each time record?
- **Answer:** 60-second time records [Gear wear Failure.pdf p1].
- **Ground Truth:** 60 seconds
- **Source:** eval_ragas_per_question_phase3_batch_1.jsonl
- **Passes:** ❌


**76. Score: 0.917** - When did moderate wear begin?
- **Answer:** Moderate wear began on April 23 [Gear wear Failure.pdf p3 Timeline].
- **Ground Truth:** 2023-04-23
- **Source:** eval_ragas_per_question_focused_chunking.jsonl
- **Passes:** ❌


**77. Score: 0.917** - What two steady speeds were used for data acquisition (in RPS)?
- **Answer:** 15 and 45 RPS
- **Ground Truth:** 15 and 45 RPS
- **Source:** eval_ragas_per_question_keyword_only_retrieval.jsonl
- **Passes:** ❌


**78. Score: 0.917** - Between which dates did the severe wear stage occur?
- **Answer:** Severe wear occurred between May 14 and June 11. [Gear wear Failure.pdf p10 Timeline].
- **Ground Truth:** 2023-05-14 to 2023-06-11
- **Source:** eval_ragas_per_question_phase2_batch_1.jsonl
- **Passes:** ❌


**79. Score: 0.917** - What two steady speeds were used for data acquisition (in RPS)?
- **Answer:** 15 and 45 RPS
- **Ground Truth:** 15 and 45 RPS
- **Source:** eval_ragas_per_question_qa_specific_retrieval.jsonl
- **Passes:** ❌


**80. Score: 0.917** - What two steady speeds were used for data acquisition (in RPS)?
- **Answer:** 15 and 45 RPS
- **Ground Truth:** 15 and 45 RPS
- **Source:** eval_ragas_per_question_query_specific_filtering.jsonl
- **Passes:** ❌


**81. Score: 0.917** - What two steady speeds were used for data acquisition (in RPS)?
- **Answer:** 15 and 45 RPS
- **Ground Truth:** 15 and 45 RPS
- **Source:** eval_ragas_per_question_retrieval_algorithm_focus.jsonl
- **Passes:** ❌


**82. Score: 0.917** - What two steady speeds were used for data acquisition (in RPS)?
- **Answer:** 15 and 45 RPS
- **Ground Truth:** 15 and 45 RPS
- **Source:** eval_ragas_per_question_retrieval_optimized.jsonl
- **Passes:** ❌


**83. Score: 0.917** - What two steady speeds were used for data acquisition (in RPS)?
- **Answer:** 15 and 45 RPS
- **Ground Truth:** 15 and 45 RPS
- **Source:** eval_ragas_per_question_semantic_retrieval.jsonl
- **Passes:** ❌


**84. Score: 0.917** - On what date was the first onset of wear detected by visual inspection?
- **Answer:** The first onset of wear was detected on April 9, 2023, through visual inspection [Gear wear Failure.pdf p1 Summary].
- **Ground Truth:** 2023-04-09
- **Source:** eval_ragas_per_question_topic_based_chunking.jsonl
- **Passes:** ❌


**85. Score: 0.910** - Why can RMS energy drop at the moment of failure despite catastrophic damage?
- **Answer:** The RMS energy drops due to loss of mechanical engagement within the gear set despite catastrophic damage [Gear wear Failure.pdf p7].
- **Ground Truth:** Loss of engagement and dynamic stiffness due to material loss reduces measured vibration energy.
- **Source:** eval_ragas_per_question_batch_3.jsonl
- **Passes:** ✅


**86. Score: 0.910** - Why can RMS energy drop at the moment of failure despite catastrophic damage?
- **Answer:** The RMS energy drops due to loss of mechanical engagement within the gear set despite catastrophic damage [Gear wear Failure.pdf p7].
- **Ground Truth:** Loss of engagement and dynamic stiffness due to material loss reduces measured vibration energy.
- **Source:** eval_ragas_per_question_combined.jsonl
- **Passes:** ✅


**87. Score: 0.900** - What was the sampling rate per record?
- **Answer:** 50 kHz
- **Ground Truth:** 50 kHz
- **Source:** eval_ragas_per_question_batch_1.jsonl
- **Passes:** ❌


**88. Score: 0.900** - What was the sampling rate per record?
- **Answer:** 50 kHz
- **Ground Truth:** 50 kHz
- **Source:** eval_ragas_per_question_batch_2.jsonl
- **Passes:** ❌


**89. Score: 0.900** - What physical tooth-surface changes typified the transition into severe wear?
- **Answer:** Sharp-edged scars and material loss extending across large sections of the tooth flank typified the transition into severe wear [Gear wear Failure.pdf p4].
- **Ground Truth:** Sharp-edged scars and material loss across large flank sections, including scuffing and surface tearing.
- **Source:** eval_ragas_per_question_batch_3.jsonl
- **Passes:** ❌


**90. Score: 0.900** - What was the sampling rate per record?
- **Answer:** 50 kHz
- **Ground Truth:** 50 kHz
- **Source:** eval_ragas_per_question_combined.jsonl
- **Passes:** ❌


**91. Score: 0.900** - What was the sampling rate per record?
- **Answer:** 50 kHz
- **Ground Truth:** 50 kHz
- **Source:** eval_ragas_per_question_combined.jsonl
- **Passes:** ❌


**92. Score: 0.900** - What physical tooth-surface changes typified the transition into severe wear?
- **Answer:** Sharp-edged scars and material loss extending across large sections of the tooth flank typified the transition into severe wear [Gear wear Failure.pdf p4].
- **Ground Truth:** Sharp-edged scars and material loss across large flank sections, including scuffing and surface tearing.
- **Source:** eval_ragas_per_question_combined.jsonl
- **Passes:** ❌


**93. Score: 0.900** - On what date did the system reach the failure stage?
- **Answer:** The system reached the failure stage on June 15 [Gear wear Failure.pdf p10 Timeline].
- **Ground Truth:** 2023-06-15
- **Source:** eval_ragas_per_question_focused_chunking.jsonl
- **Passes:** ❌


**94. Score: 0.900** - What was the sampling rate per record?
- **Answer:** 50 kHz
- **Ground Truth:** 50 kHz
- **Source:** eval_ragas_per_question_improved_batch_1.jsonl
- **Passes:** ❌


**95. Score: 0.900** - Between which dates did the severe wear stage occur?
- **Answer:** Severe wear occurred between May 14 and June 11 [Gear wear Failure.pdf p4 Text].
- **Ground Truth:** 2023-05-14 to 2023-06-11
- **Source:** eval_ragas_per_question_keyword_only_retrieval.jsonl
- **Passes:** ❌


**96. Score: 0.900** - When did moderate wear begin?
- **Answer:** Moderate wear began on April 23 [Gear wear Failure.pdf p10 Timeline].
- **Ground Truth:** 2023-04-23
- **Source:** eval_ragas_per_question_keyword_only_retrieval.jsonl
- **Passes:** ❌


**97. Score: 0.900** - What two steady speeds were used for data acquisition (in RPS)?
- **Answer:** 15 and 45 RPS
- **Ground Truth:** 15 and 45 RPS
- **Source:** eval_ragas_per_question_phase2_batch_1.jsonl
- **Passes:** ❌


**98. Score: 0.900** - What was the duration of each time record?
- **Answer:** A: 60 seconds [Gear wear Failure.pdf p1].
- **Ground Truth:** 60 seconds
- **Source:** eval_ragas_per_question_phase2_batch_1.jsonl
- **Passes:** ❌


**99. Score: 0.900** - Between which dates did the severe wear stage occur?
- **Answer:** Severe wear occurred between May 14 and June 11 [Gear wear Failure.pdf p4 Text].
- **Ground Truth:** 2023-05-14 to 2023-06-11
- **Source:** eval_ragas_per_question_qa_specific_retrieval.jsonl
- **Passes:** ❌


**100. Score: 0.900** - When did moderate wear begin?
- **Answer:** Moderate wear began on April 23 [Gear wear Failure.pdf p10 Timeline].
- **Ground Truth:** 2023-04-23
- **Source:** eval_ragas_per_question_qa_specific_retrieval.jsonl
- **Passes:** ❌


**101. Score: 0.900** - Between which dates did the severe wear stage occur?
- **Answer:** The severe wear stage occurred between May 14 and June 11 [Gear wear Failure.pdf p4 Text].
- **Ground Truth:** 2023-05-14 to 2023-06-11
- **Source:** eval_ragas_per_question_query_specific_filtering.jsonl
- **Passes:** ❌


**102. Score: 0.900** - When did moderate wear begin?
- **Answer:** Moderate wear began on April 23 [Gear wear Failure.pdf p10 Timeline].
- **Ground Truth:** 2023-04-23
- **Source:** eval_ragas_per_question_query_specific_filtering.jsonl
- **Passes:** ❌


**103. Score: 0.900** - Until what date did the healthy baseline extend with no abnormal indications?
- **Answer:** The healthy baseline extended with no abnormal indications until April 8, 2023 [Gear wear Failure.pdf p1].
- **Ground Truth:** 2023-04-08
- **Source:** eval_ragas_per_question_query_specific_filtering.jsonl
- **Passes:** ❌


**104. Score: 0.900** - Between which dates did the severe wear stage occur?
- **Answer:** Severe wear occurred between May 14 and June 11 [Gear wear Failure.pdf p4 Text].
- **Ground Truth:** 2023-05-14 to 2023-06-11
- **Source:** eval_ragas_per_question_retrieval_algorithm_focus.jsonl
- **Passes:** ❌


**105. Score: 0.900** - When did moderate wear begin?
- **Answer:** Moderate wear began on April 23 [Gear wear Failure.pdf p10 Timeline].
- **Ground Truth:** 2023-04-23
- **Source:** eval_ragas_per_question_retrieval_algorithm_focus.jsonl
- **Passes:** ❌


**106. Score: 0.900** - Between which dates did the severe wear stage occur?
- **Answer:** Severe wear occurred between May 14 and June 11 [Gear wear Failure.pdf p4 Text].
- **Ground Truth:** 2023-05-14 to 2023-06-11
- **Source:** eval_ragas_per_question_retrieval_optimized.jsonl
- **Passes:** ❌


**107. Score: 0.900** - When did moderate wear begin?
- **Answer:** Moderate wear began on April 23 [Gear wear Failure.pdf p10 table].
- **Ground Truth:** 2023-04-23
- **Source:** eval_ragas_per_question_retrieval_optimized.jsonl
- **Passes:** ❌


**108. Score: 0.900** - When did moderate wear begin?
- **Answer:** Moderate wear began on April 23 [Gear wear Failure.pdf p3 Timeline].
- **Ground Truth:** 2023-04-23
- **Source:** eval_ragas_per_question_semantic_chunking.jsonl
- **Passes:** ❌


**109. Score: 0.900** - What was the sampling rate per record?
- **Answer:** 50 kHz
- **Ground Truth:** 50 kHz
- **Source:** eval_ragas_per_question_semantic_chunking.jsonl
- **Passes:** ❌


**110. Score: 0.900** - Between which dates did the severe wear stage occur?
- **Answer:** Severe wear occurred between May 14 and June 11 [Gear wear Failure.pdf p4 Text].
- **Ground Truth:** 2023-05-14 to 2023-06-11
- **Source:** eval_ragas_per_question_semantic_retrieval.jsonl
- **Passes:** ❌


**111. Score: 0.900** - When did moderate wear begin?
- **Answer:** Moderate wear began on April 23 [Gear wear Failure.pdf p10 Timeline].
- **Ground Truth:** 2023-04-23
- **Source:** eval_ragas_per_question_semantic_retrieval.jsonl
- **Passes:** ❌


**112. Score: 0.900** - Between which dates did the severe wear stage occur?
- **Answer:** Severe wear occurred between May 14 and June 11 [Gear wear Failure.pdf p10 Timeline].
- **Ground Truth:** 2023-05-14 to 2023-06-11
- **Source:** eval_ragas_per_question_topic_based_chunking.jsonl
- **Passes:** ❌


**113. Score: 0.900** - When did moderate wear begin?
- **Answer:** Moderate wear began on April 23 [Gear wear Failure.pdf p10 Timeline].
- **Ground Truth:** 2023-04-23
- **Source:** eval_ragas_per_question_topic_based_chunking.jsonl
- **Passes:** ❌


**114. Score: 0.900** - Until what date did the healthy baseline extend with no abnormal indications?
- **Answer:** The healthy baseline extended with no abnormal indications until April 8, 2023 [Gear wear Failure.pdf p1].
- **Ground Truth:** 2023-04-08
- **Source:** eval_ragas_per_question_topic_based_chunking.jsonl
- **Passes:** ❌


**115. Score: 0.893** - On what date did the system reach the failure stage?
- **Answer:** A: June 15 [Gear wear Failure.pdf p5 Text].
- **Ground Truth:** 2023-06-15
- **Source:** eval_ragas_per_question_phase2_batch_1.jsonl
- **Passes:** ❌


**116. Score: 0.883** - On what date was the first onset of wear detected by visual inspection?
- **Answer:** The first onset of wear was detected on April 9, 2023, through visual inspection [Gear wear Failure.pdf p7 Analysis].
- **Ground Truth:** 2023-04-09
- **Source:** eval_ragas_per_question_focused_chunking.jsonl
- **Passes:** ❌


**117. Score: 0.883** - What is the model of the marine reduction gearbox investigated?
- **Answer:** The model of the marine reduction gearbox investigated is MG-5025A [Gear wear Failure.pdf p1].
- **Ground Truth:** MG-5025A
- **Source:** eval_ragas_per_question_focused_chunking.jsonl
- **Passes:** ❌


**118. Score: 0.883** - What two steady speeds were used for data acquisition (in RPS)?
- **Answer:** 15 and 45 RPS
- **Ground Truth:** 15 and 45 RPS
- **Source:** eval_ragas_per_question_focused_chunking.jsonl
- **Passes:** ❌


**119. Score: 0.883** - What was the sampling rate per record?
- **Answer:** 50 kHz
- **Ground Truth:** 50 kHz
- **Source:** eval_ragas_per_question_focused_chunking.jsonl
- **Passes:** ❌


**120. Score: 0.883** - Until what date did the healthy baseline extend with no abnormal indications?
- **Answer:** The healthy baseline extended with no abnormal indications until April 8, 2023 [Gear wear Failure.pdf p1].
- **Ground Truth:** 2023-04-08
- **Source:** eval_ragas_per_question_keyword_only_retrieval.jsonl
- **Passes:** ❌


**121. Score: 0.883** - What was the sampling rate per record?
- **Answer:** 50 kHz
- **Ground Truth:** 50 kHz
- **Source:** eval_ragas_per_question_keyword_only_retrieval.jsonl
- **Passes:** ❌


**122. Score: 0.883** - What was the sampling rate per record?
- **Answer:** 50 kHz
- **Ground Truth:** 50 kHz
- **Source:** eval_ragas_per_question_phase2_batch_1.jsonl
- **Passes:** ❌


**123. Score: 0.883** - Until what date did the healthy baseline extend with no abnormal indications?
- **Answer:** The healthy baseline extended with no abnormal indications until April 8, 2023 [Gear wear Failure.pdf p1].
- **Ground Truth:** 2023-04-08
- **Source:** eval_ragas_per_question_qa_specific_retrieval.jsonl
- **Passes:** ❌


**124. Score: 0.883** - What was the sampling rate per record?
- **Answer:** 50 kHz
- **Ground Truth:** 50 kHz
- **Source:** eval_ragas_per_question_qa_specific_retrieval.jsonl
- **Passes:** ❌


**125. Score: 0.883** - What was the sampling rate per record?
- **Answer:** 50 kHz
- **Ground Truth:** 50 kHz
- **Source:** eval_ragas_per_question_query_specific_filtering.jsonl
- **Passes:** ❌


**126. Score: 0.883** - What was the sampling rate per record?
- **Answer:** 50 kHz
- **Ground Truth:** 50 kHz
- **Source:** eval_ragas_per_question_retrieval_algorithm_focus.jsonl
- **Passes:** ❌


**127. Score: 0.883** - What was the sampling rate per record?
- **Answer:** 50 kHz
- **Ground Truth:** 50 kHz
- **Source:** eval_ragas_per_question_retrieval_optimized.jsonl
- **Passes:** ❌


**128. Score: 0.883** - Until what date did the healthy baseline extend with no abnormal indications?
- **Answer:** The healthy baseline extended with no abnormal indications until 8 April 2023 [Gear wear Failure.pdf p1].
- **Ground Truth:** 2023-04-08
- **Source:** eval_ragas_per_question_semantic_chunking.jsonl
- **Passes:** ❌


**129. Score: 0.883** - What two steady speeds were used for data acquisition (in RPS)?
- **Answer:** 15 and 45 RPS
- **Ground Truth:** 15 and 45 RPS
- **Source:** eval_ragas_per_question_semantic_chunking.jsonl
- **Passes:** ❌


**130. Score: 0.883** - Until what date did the healthy baseline extend with no abnormal indications?
- **Answer:** The healthy baseline extended with no abnormal indications until April 8, 2023 [Gear wear Failure.pdf p1].
- **Ground Truth:** 2023-04-08
- **Source:** eval_ragas_per_question_semantic_retrieval.jsonl
- **Passes:** ❌


**131. Score: 0.883** - What was the sampling rate per record?
- **Answer:** 50 kHz
- **Ground Truth:** 50 kHz
- **Source:** eval_ragas_per_question_semantic_retrieval.jsonl
- **Passes:** ❌


**132. Score: 0.883** - What was the sampling rate per record?
- **Answer:** 50 kHz
- **Ground Truth:** 50 kHz
- **Source:** eval_ragas_per_question_topic_based_chunking.jsonl
- **Passes:** ❌


**133. Score: 0.883** - What was the duration of each time record?
- **Answer:** Each time record had a duration of 60 seconds [Gear wear Failure.pdf p1 Summary].
- **Ground Truth:** 60 seconds
- **Source:** eval_ragas_per_question_topic_based_chunking.jsonl
- **Passes:** ❌


**134. Score: 0.850** - What is the recommended action at the moderate wear stage?
- **Answer:** Schedule planned replacement or refurbishment at earliest convenience [Gear wear Failure.pdf p8].
- **Ground Truth:** Schedule planned replacement or refurbishment at the earliest convenience.
- **Source:** eval_ragas_per_question_batch_4.jsonl
- **Passes:** ❌


**135. Score: 0.850** - What is the wear depth for case W13 (in μm)?
- **Answer:** 344 μm [Gear wear Failure.pdf p11 table]
- **Ground Truth:** 344
- **Source:** eval_ragas_per_question_batch_4.jsonl
- **Passes:** ❌


**136. Score: 0.850** - What is the wear depth for case W33 (in μm)?
- **Answer:** 853 μm [Gear wear Failure.pdf p11 table]
- **Ground Truth:** 853
- **Source:** eval_ragas_per_question_batch_5.jsonl
- **Passes:** ❌


**137. Score: 0.850** - What is the wear depth for case W35 (in μm)?
- **Answer:** 932 μm [Gear wear Failure.pdf p11 table]
- **Ground Truth:** 932
- **Source:** eval_ragas_per_question_batch_5.jsonl
- **Passes:** ❌


**138. Score: 0.850** - Which wear case corresponds to 294 μm?
- **Answer:** The wear case corresponding to 294 μm is W10 [Gear wear Failure.pdf p11].
- **Ground Truth:** W10
- **Source:** eval_ragas_per_question_batch_5.jsonl
- **Passes:** ❌


**139. Score: 0.850** - What is the recommended action at the moderate wear stage?
- **Answer:** Schedule planned replacement or refurbishment at earliest convenience [Gear wear Failure.pdf p8].
- **Ground Truth:** Schedule planned replacement or refurbishment at the earliest convenience.
- **Source:** eval_ragas_per_question_combined.jsonl
- **Passes:** ❌


**140. Score: 0.850** - What is the wear depth for case W13 (in μm)?
- **Answer:** 344 μm [Gear wear Failure.pdf p11 table]
- **Ground Truth:** 344
- **Source:** eval_ragas_per_question_combined.jsonl
- **Passes:** ❌


**141. Score: 0.850** - What is the wear depth for case W33 (in μm)?
- **Answer:** 853 μm [Gear wear Failure.pdf p11 table]
- **Ground Truth:** 853
- **Source:** eval_ragas_per_question_combined.jsonl
- **Passes:** ❌


**142. Score: 0.850** - What is the wear depth for case W35 (in μm)?
- **Answer:** 932 μm [Gear wear Failure.pdf p11 table]
- **Ground Truth:** 932
- **Source:** eval_ragas_per_question_combined.jsonl
- **Passes:** ❌


**143. Score: 0.850** - Which wear case corresponds to 294 μm?
- **Answer:** The wear case corresponding to 294 μm is W10 [Gear wear Failure.pdf p11].
- **Ground Truth:** W10
- **Source:** eval_ragas_per_question_combined.jsonl
- **Passes:** ❌


**144. Score: 0.850** - What was the duration of each time record?
- **Answer:** 60-second time records [Gear wear Failure.pdf p1].
- **Ground Truth:** 60 seconds
- **Source:** eval_ragas_per_question_retrieval_optimized.jsonl
- **Passes:** ❌


**145. Score: 0.850** - What two steady speeds were used for data acquisition (in RPS)?
- **Answer:** 15 and 45 RPS
- **Ground Truth:** 15 and 45 RPS
- **Source:** eval_ragas_per_question_topic_based_chunking.jsonl
- **Passes:** ❌


**146. Score: 0.843** - What was the duration of each time record?
- **Answer:** 60-second time records [Gear wear Failure.pdf p1].
- **Ground Truth:** 60 seconds
- **Source:** eval_ragas_per_question_keyword_only_retrieval.jsonl
- **Passes:** ❌


**147. Score: 0.843** - What was the duration of each time record?
- **Answer:** 60-second time records [Gear wear Failure.pdf p1].
- **Ground Truth:** 60 seconds
- **Source:** eval_ragas_per_question_qa_specific_retrieval.jsonl
- **Passes:** ❌


**148. Score: 0.843** - What was the duration of each time record?
- **Answer:** 60-second time records [Gear wear Failure.pdf p1].
- **Ground Truth:** 60 seconds
- **Source:** eval_ragas_per_question_query_specific_filtering.jsonl
- **Passes:** ❌


**149. Score: 0.843** - What was the duration of each time record?
- **Answer:** 60-second time records [Gear wear Failure.pdf p1].
- **Ground Truth:** 60 seconds
- **Source:** eval_ragas_per_question_retrieval_algorithm_focus.jsonl
- **Passes:** ❌


**150. Score: 0.843** - What was the duration of each time record?
- **Answer:** 60-second time records [Gear wear Failure.pdf p1].
- **Ground Truth:** 60 seconds
- **Source:** eval_ragas_per_question_semantic_chunking.jsonl
- **Passes:** ❌


**151. Score: 0.840** - Which vessel's propulsion train was monitored?
- **Answer:** Haifa [document p1]
- **Ground Truth:** INS Haifa
- **Source:** eval_ragas_per_question_topic_based_chunking.jsonl
- **Passes:** ❌


**152. Score: 0.827** - What was the duration of each time record?
- **Answer:** 60-second time records [Gear wear Failure.pdf p1].
- **Ground Truth:** 60 seconds
- **Source:** eval_ragas_per_question_semantic_retrieval.jsonl
- **Passes:** ❌


**153. Score: 0.825** - Which metric should frequency-domain monitoring emphasize for earlier detection?
- **Answer:** Frequency-domain monitoring should emphasize sidebands around the gear meshing frequency [Gear wear Failure.pdf p8 Conclusion].
- **Ground Truth:** Sidebands around the gear meshing frequency.
- **Source:** eval_ragas_per_question_batch_3.jsonl
- **Passes:** ❌


**154. Score: 0.825** - What is the recommended action at the severe wear stage?
- **Answer:** Immediate intervention required to prevent failure [Gear wear Failure.pdf p8 Conclusion].
- **Ground Truth:** Immediate intervention to prevent failure.
- **Source:** eval_ragas_per_question_batch_3.jsonl
- **Passes:** ❌


**155. Score: 0.825** - What is the wear depth for case W1 (in μm)?
- **Answer:** 40 μm [Gear wear Failure.pdf p11 table]
- **Ground Truth:** 40
- **Source:** eval_ragas_per_question_batch_4.jsonl
- **Passes:** ❌


**156. Score: 0.825** - What is the wear depth for case W22 (in μm)?
- **Answer:** 524 μm [Gear wear Failure.pdf p11 table]
- **Ground Truth:** 524
- **Source:** eval_ragas_per_question_batch_4.jsonl
- **Passes:** ❌


**157. Score: 0.825** - What is the wear depth for case W24 (in μm)?
- **Answer:** 579 μm [Gear wear Failure.pdf p11 table]
- **Ground Truth:** 579
- **Source:** eval_ragas_per_question_batch_4.jsonl
- **Passes:** ❌


**158. Score: 0.825** - What is the wear depth for case W25 (in μm)?
- **Answer:** 608 μm [Gear wear Failure.pdf p11 table]
- **Ground Truth:** 608
- **Source:** eval_ragas_per_question_batch_4.jsonl
- **Passes:** ❌


**159. Score: 0.825** - What is the wear depth for case W31 (in μm)?
- **Answer:** 797 μm [Gear wear Failure.pdf p11 table]
- **Ground Truth:** 797
- **Source:** eval_ragas_per_question_batch_4.jsonl
- **Passes:** ❌


**160. Score: 0.825** - Which wear case corresponds to 466 μm?
- **Answer:** The wear case corresponding to 466 μm is W19 [Gear wear Failure.pdf p11].
- **Ground Truth:** W19
- **Source:** eval_ragas_per_question_batch_5.jsonl
- **Passes:** ❌


**161. Score: 0.825** - Which wear case corresponds to 744 μm?
- **Answer:** The wear case that corresponds to 744 μm is W29 [Gear wear Failure.pdf p11].
- **Ground Truth:** W29
- **Source:** eval_ragas_per_question_batch_5.jsonl
- **Passes:** ❌


**162. Score: 0.825** - Which wear case corresponds to 579 μm?
- **Answer:** The wear case that corresponds to 579 μm is W24 [Gear wear Failure.pdf p11 Table (Table 1)].
- **Ground Truth:** W24
- **Source:** eval_ragas_per_question_batch_5.jsonl
- **Passes:** ❌


**163. Score: 0.825** - Which metric should frequency-domain monitoring emphasize for earlier detection?
- **Answer:** Frequency-domain monitoring should emphasize sidebands around the gear meshing frequency [Gear wear Failure.pdf p8 Conclusion].
- **Ground Truth:** Sidebands around the gear meshing frequency.
- **Source:** eval_ragas_per_question_combined.jsonl
- **Passes:** ❌


**164. Score: 0.825** - What is the recommended action at the severe wear stage?
- **Answer:** Immediate intervention required to prevent failure [Gear wear Failure.pdf p8 Conclusion].
- **Ground Truth:** Immediate intervention to prevent failure.
- **Source:** eval_ragas_per_question_combined.jsonl
- **Passes:** ❌


**165. Score: 0.825** - What is the wear depth for case W1 (in μm)?
- **Answer:** 40 μm [Gear wear Failure.pdf p11 table]
- **Ground Truth:** 40
- **Source:** eval_ragas_per_question_combined.jsonl
- **Passes:** ❌


**166. Score: 0.825** - What is the wear depth for case W22 (in μm)?
- **Answer:** 524 μm [Gear wear Failure.pdf p11 table]
- **Ground Truth:** 524
- **Source:** eval_ragas_per_question_combined.jsonl
- **Passes:** ❌


**167. Score: 0.825** - What is the wear depth for case W24 (in μm)?
- **Answer:** 579 μm [Gear wear Failure.pdf p11 table]
- **Ground Truth:** 579
- **Source:** eval_ragas_per_question_combined.jsonl
- **Passes:** ❌


**168. Score: 0.825** - What is the wear depth for case W25 (in μm)?
- **Answer:** 608 μm [Gear wear Failure.pdf p11 table]
- **Ground Truth:** 608
- **Source:** eval_ragas_per_question_combined.jsonl
- **Passes:** ❌


**169. Score: 0.825** - What is the wear depth for case W31 (in μm)?
- **Answer:** 797 μm [Gear wear Failure.pdf p11 table]
- **Ground Truth:** 797
- **Source:** eval_ragas_per_question_combined.jsonl
- **Passes:** ❌


**170. Score: 0.825** - Which wear case corresponds to 466 μm?
- **Answer:** The wear case corresponding to 466 μm is W19 [Gear wear Failure.pdf p11].
- **Ground Truth:** W19
- **Source:** eval_ragas_per_question_combined.jsonl
- **Passes:** ❌


**171. Score: 0.825** - Which wear case corresponds to 744 μm?
- **Answer:** The wear case that corresponds to 744 μm is W29 [Gear wear Failure.pdf p11].
- **Ground Truth:** W29
- **Source:** eval_ragas_per_question_combined.jsonl
- **Passes:** ❌


**172. Score: 0.825** - Which wear case corresponds to 579 μm?
- **Answer:** The wear case that corresponds to 579 μm is W24 [Gear wear Failure.pdf p11 Table (Table 1)].
- **Ground Truth:** W24
- **Source:** eval_ragas_per_question_combined.jsonl
- **Passes:** ❌


**173. Score: 0.810** - List two hallmark spectral changes that defined the moderate wear stage.
- **Answer:** 1. Spectral content displaying organized sideband families and a rising broadband floor across successive days [Gear wear Failure.pdf p4].
- **Ground Truth:** Organized sideband families and a rising broadband floor.
- **Source:** eval_ragas_per_question_batch_3.jsonl
- **Passes:** ❌


**174. Score: 0.810** - List two hallmark spectral changes that defined the moderate wear stage.
- **Answer:** 1. Spectral content displaying organized sideband families and a rising broadband floor across successive days [Gear wear Failure.pdf p4].
- **Ground Truth:** Organized sideband families and a rising broadband floor.
- **Source:** eval_ragas_per_question_combined.jsonl
- **Passes:** ❌


**175. Score: 0.800** - Which vessel's propulsion train was monitored?
- **Answer:** INS Haifa Executive Summary This report documents a progressive tooth [document p1]
- **Ground Truth:** INS Haifa
- **Source:** eval_ragas_per_question_keyword_only_retrieval.jsonl
- **Passes:** ❌


**176. Score: 0.800** - Which vessel's propulsion train was monitored?
- **Answer:** INS Haifa Executive Summary This report documents a progressive tooth [document p1]
- **Ground Truth:** INS Haifa
- **Source:** eval_ragas_per_question_phase3_batch_1.jsonl
- **Passes:** ❌


**177. Score: 0.800** - Which vessel's propulsion train was monitored?
- **Answer:** INS Haifa Executive Summary This report documents a progressive tooth [document p1]
- **Ground Truth:** INS Haifa
- **Source:** eval_ragas_per_question_qa_specific_retrieval.jsonl
- **Passes:** ❌


**178. Score: 0.800** - Which vessel's propulsion train was monitored?
- **Answer:** INS Haifa Executive Summary This report documents a progressive tooth [document p1]
- **Ground Truth:** INS Haifa
- **Source:** eval_ragas_per_question_query_specific_filtering.jsonl
- **Passes:** ❌


**179. Score: 0.800** - Which vessel's propulsion train was monitored?
- **Answer:** INS Haifa Executive Summary This report documents a progressive tooth [document p1]
- **Ground Truth:** INS Haifa
- **Source:** eval_ragas_per_question_retrieval_algorithm_focus.jsonl
- **Passes:** ❌


**180. Score: 0.800** - Which vessel's propulsion train was monitored?
- **Answer:** INS Haifa Executive Summary This report documents a progressive tooth [document p1]
- **Ground Truth:** INS Haifa
- **Source:** eval_ragas_per_question_retrieval_optimized.jsonl
- **Passes:** ❌


**181. Score: 0.800** - Which vessel's propulsion train was monitored?
- **Answer:** INS Haifa Executive Summary This report documents a progressive tooth [document p1]
- **Ground Truth:** INS Haifa
- **Source:** eval_ragas_per_question_semantic_retrieval.jsonl
- **Passes:** ❌


**182. Score: 0.795** - Name three combined indicators that characterized the severe wear stage.
- **Answer:** 1. Extensive and system-wide loss of tooth surface geometry.  
2. Vibration signatures characterized by large RMS growth, dense sidebands, and impulsive transients.  
3. Signal data showed significant energy concentration shifts in the spectral domain, with broadening peaks and elevated noise levels. [Gear wear Failure.pdf p5]
- **Ground Truth:** Large RMS growth, dense sidebands, and impulsive transients.
- **Source:** eval_ragas_per_question_batch_3.jsonl
- **Passes:** ❌


**183. Score: 0.795** - Name three combined indicators that characterized the severe wear stage.
- **Answer:** 1. Extensive and system-wide loss of tooth surface geometry.  
2. Vibration signatures characterized by large RMS growth, dense sidebands, and impulsive transients.  
3. Signal data showed significant energy concentration shifts in the spectral domain, with broadening peaks and elevated noise levels. [Gear wear Failure.pdf p5]
- **Ground Truth:** Large RMS growth, dense sidebands, and impulsive transients.
- **Source:** eval_ragas_per_question_combined.jsonl
- **Passes:** ❌


**184. Score: 0.783** - Until what date did the healthy baseline extend with no abnormal indications?
- **Answer:** The healthy baseline extended until April 8 with no abnormal indications [Gear wear Failure.pdf p6 Analysis].
- **Ground Truth:** 2023-04-08
- **Source:** eval_ragas_per_question_focused_chunking.jsonl
- **Passes:** ❌


**185. Score: 0.767** - Which vessel's propulsion train was monitored?
- **Answer:** INS Haifa Executive Summary This report documents a progressive tooth [document p1]
- **Ground Truth:** INS Haifa
- **Source:** eval_ragas_per_question_phase2_batch_1.jsonl
- **Passes:** ❌


**186. Score: 0.767** - Which vessel's propulsion train was monitored?
- **Answer:** INS Haifa Executive Summary This report documents a progressive tooth [document p1]
- **Ground Truth:** INS Haifa
- **Source:** eval_ragas_per_question_semantic_chunking.jsonl
- **Passes:** ❌


**187. Score: 0.750** - Which two data sources should be synchronized into a single diagnostic report?
- **Answer:** Synchronize image data and vibration signals into a single diagnostic report [Gear wear Failure.pdf p9].
- **Ground Truth:** Image data and vibration signals.
- **Source:** eval_ragas_per_question_batch_4.jsonl
- **Passes:** ❌


**188. Score: 0.750** - Which two data sources should be synchronized into a single diagnostic report?
- **Answer:** Synchronize image data and vibration signals into a single diagnostic report [Gear wear Failure.pdf p9].
- **Ground Truth:** Image data and vibration signals.
- **Source:** eval_ragas_per_question_combined.jsonl
- **Passes:** ❌


**189. Score: 0.725** - Which vessel's propulsion train was monitored?
- **Answer:** INS Haifa Executive Summary This report documents a progressive tooth [document p1]
- **Ground Truth:** INS Haifa
- **Source:** eval_ragas_per_question_batch_1.jsonl
- **Passes:** ❌


**190. Score: 0.725** - What was the duration of each time record?
- **Answer:** 60 seconds [Gear wear Failure.pdf p1 Summary].
- **Ground Truth:** 60 seconds
- **Source:** eval_ragas_per_question_batch_1.jsonl
- **Passes:** ❌


**191. Score: 0.725** - Which vessel's propulsion train was monitored?
- **Answer:** INS Haifa Executive Summary This report documents a progressive tooth [document p1]
- **Ground Truth:** INS Haifa
- **Source:** eval_ragas_per_question_batch_2.jsonl
- **Passes:** ❌


**192. Score: 0.725** - What was the duration of each time record?
- **Answer:** 60 seconds [Gear wear Failure.pdf p1 Summary].
- **Ground Truth:** 60 seconds
- **Source:** eval_ragas_per_question_batch_2.jsonl
- **Passes:** ❌


**193. Score: 0.725** - Which vessel's propulsion train was monitored?
- **Answer:** INS Haifa Executive Summary This report documents a progressive tooth [document p1]
- **Ground Truth:** INS Haifa
- **Source:** eval_ragas_per_question_combined.jsonl
- **Passes:** ❌


**194. Score: 0.725** - What was the duration of each time record?
- **Answer:** 60 seconds [Gear wear Failure.pdf p1 Summary].
- **Ground Truth:** 60 seconds
- **Source:** eval_ragas_per_question_combined.jsonl
- **Passes:** ❌


**195. Score: 0.725** - Which vessel's propulsion train was monitored?
- **Answer:** INS Haifa Executive Summary This report documents a progressive tooth [document p1]
- **Ground Truth:** INS Haifa
- **Source:** eval_ragas_per_question_combined.jsonl
- **Passes:** ❌


**196. Score: 0.725** - What was the duration of each time record?
- **Answer:** 60 seconds [Gear wear Failure.pdf p1 Summary].
- **Ground Truth:** 60 seconds
- **Source:** eval_ragas_per_question_combined.jsonl
- **Passes:** ❌


**197. Score: 0.700** - At which two speeds were both baseline and severe-wear measurements acquired?
- **Answer:** 15 and 45 RPS
- **Ground Truth:** 15 and 45 RPS
- **Source:** eval_ragas_per_question_batch_5.jsonl
- **Passes:** ❌


**198. Score: 0.700** - At which two speeds were both baseline and severe-wear measurements acquired?
- **Answer:** 15 and 45 RPS
- **Ground Truth:** 15 and 45 RPS
- **Source:** eval_ragas_per_question_combined.jsonl
- **Passes:** ❌


**199. Score: 0.675** - What was the duration of each time record?
- **Answer:** 60 seconds [Gear wear Failure.pdf p1].
- **Ground Truth:** 60 seconds
- **Source:** eval_ragas_per_question_improved_batch_1.jsonl
- **Passes:** ❌


**200. Score: 0.617** - What was the duration of each time record?
- **Answer:** 60 seconds [Gear wear Failure.pdf p2].
- **Ground Truth:** 60 seconds
- **Source:** eval_ragas_per_question_focused_chunking.jsonl
- **Passes:** ❌


**201. Score: 0.565** - List the health stages in chronological order.
- **Answer:** 1. Healthy  
2. Mild Wear  
3. Moderate Wear  
4. Severe Wear [Gear wear Failure.pdf p9].
- **Ground Truth:** Baseline, Mild Wear, Moderate Wear, Severe Wear, Failure
- **Source:** eval_ragas_per_question_batch_5.jsonl
- **Passes:** ❌


**202. Score: 0.565** - List the health stages in chronological order.
- **Answer:** 1. Healthy  
2. Mild Wear  
3. Moderate Wear  
4. Severe Wear [Gear wear Failure.pdf p9].
- **Ground Truth:** Baseline, Mild Wear, Moderate Wear, Severe Wear, Failure
- **Source:** eval_ragas_per_question_combined.jsonl
- **Passes:** ❌


**203. Score: 0.550** - Name the two primary sensor modalities used to document wear progression.
- **Answer:** Accelerometers and tachometer.
- **Ground Truth:** Accelerometers and microscope photography.
- **Source:** eval_ragas_per_question_batch_5.jsonl
- **Passes:** ❌


**204. Score: 0.550** - Name the two primary sensor modalities used to document wear progression.
- **Answer:** Accelerometers and tachometer.
- **Ground Truth:** Accelerometers and microscope photography.
- **Source:** eval_ragas_per_question_combined.jsonl
- **Passes:** ❌


**205. Score: 0.525** - Which vessel's propulsion train was monitored?
- **Answer:** INS Haifa Executive Summary This report documents a progressive tooth [document p1]
- **Ground Truth:** INS Haifa
- **Source:** eval_ragas_per_question_improved_batch_1.jsonl
- **Passes:** ❌


**206. Score: 0.517** - Which vessel's propulsion train was monitored?
- **Answer:** INS Haifa Executive Summary This report documents a progressive tooth [document p1]
- **Ground Truth:** INS Haifa
- **Source:** eval_ragas_per_question_focused_chunking.jsonl
- **Passes:** ❌


**207. Score: 0.435** - At 15 RPS during early wear, what high-frequency behavior was observed?
- **Answer:** 15 and 45 RPS
- **Ground Truth:** More high-frequency smearing (haze) indicative of lubrication sensitivity.
- **Source:** eval_ragas_per_question_batch_3.jsonl
- **Passes:** ❌


**208. Score: 0.435** - At 15 RPS during early wear, what high-frequency behavior was observed?
- **Answer:** 15 and 45 RPS
- **Ground Truth:** More high-frequency smearing (haze) indicative of lubrication sensitivity.
- **Source:** eval_ragas_per_question_combined.jsonl
- **Passes:** ❌


**209. Score: 0.275** - By approximately how much did RMS rise above April 9 levels at 45 RPS during moderate wear?
- **Answer:** 15 and 45 RPS
- **Ground Truth:** About 10–15%
- **Source:** eval_ragas_per_question_batch_3.jsonl
- **Passes:** ❌


**210. Score: 0.275** - By approximately how much did RMS rise above April 9 levels at 45 RPS during moderate wear?
- **Answer:** 15 and 45 RPS
- **Ground Truth:** About 10–15%
- **Source:** eval_ragas_per_question_combined.jsonl
- **Passes:** ❌


**211. Score: 0.225** - During severe wear at 45 RPS, by how much did RMS exceed the April 9 baseline?
- **Answer:** Not found in context.
- **Ground Truth:** Approximately 25–40%
- **Source:** eval_ragas_per_question_batch_3.jsonl
- **Passes:** ❌


**212. Score: 0.225** - During severe wear at 45 RPS, by how much did RMS exceed the April 9 baseline?
- **Answer:** Not found in context.
- **Ground Truth:** Approximately 25–40%
- **Source:** eval_ragas_per_question_combined.jsonl
- **Passes:** ❌


**213. Score: 0.000** - Name one AI-driven image task suggested to support objective assessments.
- **Answer:** Not found in context. [Gear wear Failure.pdf p12 figure]
- **Ground Truth:** Surface crack detection (or pitting quantification).
- **Source:** eval_ragas_per_question_batch_4.jsonl
- **Passes:** ❌


**214. Score: 0.000** - Name one AI-driven image task suggested to support objective assessments.
- **Answer:** Not found in context. [Gear wear Failure.pdf p12 figure]
- **Ground Truth:** Surface crack detection (or pitting quantification).
- **Source:** eval_ragas_per_question_combined.jsonl
- **Passes:** ❌



## 📊 BATCH EVALUATION QUESTIONS ANALYSIS

### 📈 Executive Summary
- **Total Batch Questions Found:** 47
- **Passed Questions:** 5
- **Pass Rate:** 10.6%

### 🏆 TOP 10 QUESTIONS WITH HIGHEST SCORES (ALL BATCH FILES)


#### 1. **On what date did the system reach the failure stage?**
- **Agent Answer:** The system reached the failure stage on June 15 [Gear wear Failure.pdf p10 Timeline].
- **Ground Truth:** 2023-06-15
- **Overall Score:** 0.975
- **Individual Scores:**
  - Answer Correctness: 1.000
  - Context Precision: 0.875
  - Context Recall: 1.000
  - Faithfulness: 1.000
- **Source File:** eval_ragas_per_question_batch_1.jsonl
- **Passes Thresholds:** ✅ True
- **Is Table Question:** ❌ False

---

#### 2. **On what date did the system reach the failure stage?**
- **Agent Answer:** The system reached the failure stage on June 15 [Gear wear Failure.pdf p10 Timeline].
- **Ground Truth:** 2023-06-15
- **Overall Score:** 0.975
- **Individual Scores:**
  - Answer Correctness: 1.000
  - Context Precision: 0.875
  - Context Recall: 1.000
  - Faithfulness: 1.000
- **Source File:** eval_ragas_per_question_batch_2.jsonl
- **Passes Thresholds:** ✅ True
- **Is Table Question:** ❌ False

---

#### 3. **When did moderate wear begin?**
- **Agent Answer:** Moderate wear began on April 23 [Gear wear Failure.pdf p3 Timeline].
- **Ground Truth:** 2023-04-23
- **Overall Score:** 0.950
- **Individual Scores:**
  - Answer Correctness: 1.000
  - Context Precision: 0.750
  - Context Recall: 1.000
  - Faithfulness: 1.000
- **Source File:** eval_ragas_per_question_batch_1.jsonl
- **Passes Thresholds:** ✅ True
- **Is Table Question:** ❌ False

---

#### 4. **When did moderate wear begin?**
- **Agent Answer:** Moderate wear began on April 23 [Gear wear Failure.pdf p3 Timeline].
- **Ground Truth:** 2023-04-23
- **Overall Score:** 0.950
- **Individual Scores:**
  - Answer Correctness: 1.000
  - Context Precision: 0.750
  - Context Recall: 1.000
  - Faithfulness: 1.000
- **Source File:** eval_ragas_per_question_batch_2.jsonl
- **Passes Thresholds:** ✅ True
- **Is Table Question:** ❌ False

---

#### 5. **On what date was the first onset of wear detected by visual inspection?**
- **Agent Answer:** The first onset of wear was detected on April 9, 2023 [Gear wear Failure.pdf p9 Conclusion].
- **Ground Truth:** 2023-04-09
- **Overall Score:** 0.925
- **Individual Scores:**
  - Answer Correctness: 1.000
  - Context Precision: 0.625
  - Context Recall: 1.000
  - Faithfulness: 1.000
- **Source File:** eval_ragas_per_question_batch_1.jsonl
- **Passes Thresholds:** ❌ False
- **Is Table Question:** ❌ False

---

#### 6. **Between which dates did the severe wear stage occur?**
- **Agent Answer:** Severe wear occurred between May 14 and June 11 [Gear wear Failure.pdf p10 table].
- **Ground Truth:** 2023-05-14 to 2023-06-11
- **Overall Score:** 0.925
- **Individual Scores:**
  - Answer Correctness: 1.000
  - Context Precision: 0.625
  - Context Recall: 1.000
  - Faithfulness: 1.000
- **Source File:** eval_ragas_per_question_batch_1.jsonl
- **Passes Thresholds:** ❌ False
- **Is Table Question:** ❌ False

---

#### 7. **Until what date did the healthy baseline extend with no abnormal indications?**
- **Agent Answer:** The healthy baseline extended until April 8, 2023, with no abnormal indications [Gear wear Failure.pdf p1].
- **Ground Truth:** 2023-04-08
- **Overall Score:** 0.925
- **Individual Scores:**
  - Answer Correctness: 1.000
  - Context Precision: 0.625
  - Context Recall: 1.000
  - Faithfulness: 1.000
- **Source File:** eval_ragas_per_question_batch_1.jsonl
- **Passes Thresholds:** ❌ False
- **Is Table Question:** ❌ False

---

#### 8. **What is the model of the marine reduction gearbox investigated?**
- **Agent Answer:** The model of the marine reduction gearbox investigated is MG-5025A [Gear wear Failure.pdf p1].
- **Ground Truth:** MG-5025A
- **Overall Score:** 0.925
- **Individual Scores:**
  - Answer Correctness: 1.000
  - Context Precision: 0.625
  - Context Recall: 1.000
  - Faithfulness: 1.000
- **Source File:** eval_ragas_per_question_batch_1.jsonl
- **Passes Thresholds:** ❌ False
- **Is Table Question:** ❌ False

---

#### 9. **What two steady speeds were used for data acquisition (in RPS)?**
- **Agent Answer:** 15 and 45 RPS
- **Ground Truth:** 15 and 45 RPS
- **Overall Score:** 0.925
- **Individual Scores:**
  - Answer Correctness: 1.000
  - Context Precision: 0.625
  - Context Recall: 1.000
  - Faithfulness: 1.000
- **Source File:** eval_ragas_per_question_batch_1.jsonl
- **Passes Thresholds:** ❌ False
- **Is Table Question:** ❌ False

---

#### 10. **On what date was the first onset of wear detected by visual inspection?**
- **Agent Answer:** The first onset of wear was detected on April 9, 2023 [Gear wear Failure.pdf p9 Conclusion].
- **Ground Truth:** 2023-04-09
- **Overall Score:** 0.925
- **Individual Scores:**
  - Answer Correctness: 1.000
  - Context Precision: 0.625
  - Context Recall: 1.000
  - Faithfulness: 1.000
- **Source File:** eval_ragas_per_question_batch_2.jsonl
- **Passes Thresholds:** ❌ False
- **Is Table Question:** ❌ False

---

### 📊 SUMMARY STATISTICS FOR ALL BATCH QUESTIONS

| Metric | Value |
|--------|-------|
| **Average Score** | 0.792 |
| **Score Range** | 0.000 - 0.975 |
| **Questions Passing Thresholds** | 5/47 (10.6%) |

### 🎯 PASSED QUESTIONS ANALYSIS

| Metric | Value |
|--------|-------|
| **Total Passed Questions** | 5 |
| **Pass Rate** | 10.6% |

### 📋 ALL PASSED QUESTIONS (RANKED BY SCORE)


** 1. Score: 0.975** - On what date did the system reach the failure stage?
- **Agent Answer:** The system reached the failure stage on June 15 [Gear wear Failure.pdf p10 Timeline].
- **Ground Truth:** 2023-06-15
- **Individual Scores:** AC: 1.000, CP: 0.875, CR: 1.000, F: 1.000
- **Source:** eval_ragas_per_question_batch_1.jsonl
- **Is Table Question:** ❌


** 2. Score: 0.975** - On what date did the system reach the failure stage?
- **Agent Answer:** The system reached the failure stage on June 15 [Gear wear Failure.pdf p10 Timeline].
- **Ground Truth:** 2023-06-15
- **Individual Scores:** AC: 1.000, CP: 0.875, CR: 1.000, F: 1.000
- **Source:** eval_ragas_per_question_batch_2.jsonl
- **Is Table Question:** ❌


** 3. Score: 0.950** - When did moderate wear begin?
- **Agent Answer:** Moderate wear began on April 23 [Gear wear Failure.pdf p3 Timeline].
- **Ground Truth:** 2023-04-23
- **Individual Scores:** AC: 1.000, CP: 0.750, CR: 1.000, F: 1.000
- **Source:** eval_ragas_per_question_batch_1.jsonl
- **Is Table Question:** ❌


** 4. Score: 0.950** - When did moderate wear begin?
- **Agent Answer:** Moderate wear began on April 23 [Gear wear Failure.pdf p3 Timeline].
- **Ground Truth:** 2023-04-23
- **Individual Scores:** AC: 1.000, CP: 0.750, CR: 1.000, F: 1.000
- **Source:** eval_ragas_per_question_batch_2.jsonl
- **Is Table Question:** ❌


** 5. Score: 0.910** - Why can RMS energy drop at the moment of failure despite catastrophic damage?
- **Agent Answer:** The RMS energy drops due to loss of mechanical engagement within the gear set despite catastrophic damage [Gear wear Failure.pdf p7].
- **Ground Truth:** Loss of engagement and dynamic stiffness due to material loss reduces measured vibration energy.
- **Individual Scores:** AC: 0.900, CP: 0.750, CR: 1.000, F: 1.000
- **Source:** eval_ragas_per_question_batch_3.jsonl
- **Is Table Question:** ❌


### 📋 TOP 10 OVERALL QUESTIONS (RANKED BY SCORE)


** 1. Score: 0.975** - On what date did the system reach the failure stage?
- **Agent Answer:** The system reached the failure stage on June 15 [Gear wear Failure.pdf p10 Timeline].
- **Ground Truth:** 2023-06-15
- **Individual Scores:** AC: 1.000, CP: 0.875, CR: 1.000, F: 1.000
- **Source:** eval_ragas_per_question_batch_1.jsonl
- **Passes:** ✅
- **Is Table Question:** ❌


** 2. Score: 0.975** - On what date did the system reach the failure stage?
- **Agent Answer:** The system reached the failure stage on June 15 [Gear wear Failure.pdf p10 Timeline].
- **Ground Truth:** 2023-06-15
- **Individual Scores:** AC: 1.000, CP: 0.875, CR: 1.000, F: 1.000
- **Source:** eval_ragas_per_question_batch_2.jsonl
- **Passes:** ✅
- **Is Table Question:** ❌


** 3. Score: 0.950** - When did moderate wear begin?
- **Agent Answer:** Moderate wear began on April 23 [Gear wear Failure.pdf p3 Timeline].
- **Ground Truth:** 2023-04-23
- **Individual Scores:** AC: 1.000, CP: 0.750, CR: 1.000, F: 1.000
- **Source:** eval_ragas_per_question_batch_1.jsonl
- **Passes:** ✅
- **Is Table Question:** ❌


** 4. Score: 0.950** - When did moderate wear begin?
- **Agent Answer:** Moderate wear began on April 23 [Gear wear Failure.pdf p3 Timeline].
- **Ground Truth:** 2023-04-23
- **Individual Scores:** AC: 1.000, CP: 0.750, CR: 1.000, F: 1.000
- **Source:** eval_ragas_per_question_batch_2.jsonl
- **Passes:** ✅
- **Is Table Question:** ❌


** 5. Score: 0.925** - On what date was the first onset of wear detected by visual inspection?
- **Agent Answer:** The first onset of wear was detected on April 9, 2023 [Gear wear Failure.pdf p9 Conclusion].
- **Ground Truth:** 2023-04-09
- **Individual Scores:** AC: 1.000, CP: 0.625, CR: 1.000, F: 1.000
- **Source:** eval_ragas_per_question_batch_1.jsonl
- **Passes:** ❌
- **Is Table Question:** ❌


** 6. Score: 0.925** - Between which dates did the severe wear stage occur?
- **Agent Answer:** Severe wear occurred between May 14 and June 11 [Gear wear Failure.pdf p10 table].
- **Ground Truth:** 2023-05-14 to 2023-06-11
- **Individual Scores:** AC: 1.000, CP: 0.625, CR: 1.000, F: 1.000
- **Source:** eval_ragas_per_question_batch_1.jsonl
- **Passes:** ❌
- **Is Table Question:** ❌


** 7. Score: 0.925** - Until what date did the healthy baseline extend with no abnormal indications?
- **Agent Answer:** The healthy baseline extended until April 8, 2023, with no abnormal indications [Gear wear Failure.pdf p1].
- **Ground Truth:** 2023-04-08
- **Individual Scores:** AC: 1.000, CP: 0.625, CR: 1.000, F: 1.000
- **Source:** eval_ragas_per_question_batch_1.jsonl
- **Passes:** ❌
- **Is Table Question:** ❌


** 8. Score: 0.925** - What is the model of the marine reduction gearbox investigated?
- **Agent Answer:** The model of the marine reduction gearbox investigated is MG-5025A [Gear wear Failure.pdf p1].
- **Ground Truth:** MG-5025A
- **Individual Scores:** AC: 1.000, CP: 0.625, CR: 1.000, F: 1.000
- **Source:** eval_ragas_per_question_batch_1.jsonl
- **Passes:** ❌
- **Is Table Question:** ❌


** 9. Score: 0.925** - What two steady speeds were used for data acquisition (in RPS)?
- **Agent Answer:** 15 and 45 RPS
- **Ground Truth:** 15 and 45 RPS
- **Individual Scores:** AC: 1.000, CP: 0.625, CR: 1.000, F: 1.000
- **Source:** eval_ragas_per_question_batch_1.jsonl
- **Passes:** ❌
- **Is Table Question:** ❌


**10. Score: 0.925** - On what date was the first onset of wear detected by visual inspection?
- **Agent Answer:** The first onset of wear was detected on April 9, 2023 [Gear wear Failure.pdf p9 Conclusion].
- **Ground Truth:** 2023-04-09
- **Individual Scores:** AC: 1.000, CP: 0.625, CR: 1.000, F: 1.000
- **Source:** eval_ragas_per_question_batch_2.jsonl
- **Passes:** ❌
- **Is Table Question:** ❌




## 🔍 WEAR DEPTH QUESTIONS ANALYSIS

### 📈 Executive Summary
- **Total Wear Depth Questions Found:** 16
- **Out of Total Questions:** 214
- **Percentage:** 7.5%

### 🏆 ALL WEAR DEPTH QUESTIONS (RANKED BY SCORE)


#### 1. **What is the wear depth for case W13 (in μm)?**
- **Agent Answer:** 344 μm [Gear wear Failure.pdf p11 table]
- **Ground Truth:** 344
- **Overall Score:** 0.850
- **Individual Scores:**
  - Answer Correctness: 1.000
  - Context Precision: 0.250
  - Context Recall: 1.000
  - Faithfulness: 1.000
- **Source File:** eval_ragas_per_question_batch_4.jsonl
- **Passes Thresholds:** ❌ False
- **Is Table Question:** ✅ True

---

#### 2. **What is the wear depth for case W33 (in μm)?**
- **Agent Answer:** 853 μm [Gear wear Failure.pdf p11 table]
- **Ground Truth:** 853
- **Overall Score:** 0.850
- **Individual Scores:**
  - Answer Correctness: 1.000
  - Context Precision: 0.250
  - Context Recall: 1.000
  - Faithfulness: 1.000
- **Source File:** eval_ragas_per_question_batch_5.jsonl
- **Passes Thresholds:** ❌ False
- **Is Table Question:** ✅ True

---

#### 3. **What is the wear depth for case W35 (in μm)?**
- **Agent Answer:** 932 μm [Gear wear Failure.pdf p11 table]
- **Ground Truth:** 932
- **Overall Score:** 0.850
- **Individual Scores:**
  - Answer Correctness: 1.000
  - Context Precision: 0.250
  - Context Recall: 1.000
  - Faithfulness: 1.000
- **Source File:** eval_ragas_per_question_batch_5.jsonl
- **Passes Thresholds:** ❌ False
- **Is Table Question:** ✅ True

---

#### 4. **What is the wear depth for case W13 (in μm)?**
- **Agent Answer:** 344 μm [Gear wear Failure.pdf p11 table]
- **Ground Truth:** 344
- **Overall Score:** 0.850
- **Individual Scores:**
  - Answer Correctness: 1.000
  - Context Precision: 0.250
  - Context Recall: 1.000
  - Faithfulness: 1.000
- **Source File:** eval_ragas_per_question_combined.jsonl
- **Passes Thresholds:** ❌ False
- **Is Table Question:** ✅ True

---

#### 5. **What is the wear depth for case W33 (in μm)?**
- **Agent Answer:** 853 μm [Gear wear Failure.pdf p11 table]
- **Ground Truth:** 853
- **Overall Score:** 0.850
- **Individual Scores:**
  - Answer Correctness: 1.000
  - Context Precision: 0.250
  - Context Recall: 1.000
  - Faithfulness: 1.000
- **Source File:** eval_ragas_per_question_combined.jsonl
- **Passes Thresholds:** ❌ False
- **Is Table Question:** ✅ True

---

#### 6. **What is the wear depth for case W35 (in μm)?**
- **Agent Answer:** 932 μm [Gear wear Failure.pdf p11 table]
- **Ground Truth:** 932
- **Overall Score:** 0.850
- **Individual Scores:**
  - Answer Correctness: 1.000
  - Context Precision: 0.250
  - Context Recall: 1.000
  - Faithfulness: 1.000
- **Source File:** eval_ragas_per_question_combined.jsonl
- **Passes Thresholds:** ❌ False
- **Is Table Question:** ✅ True

---

#### 7. **What is the wear depth for case W1 (in μm)?**
- **Agent Answer:** 40 μm [Gear wear Failure.pdf p11 table]
- **Ground Truth:** 40
- **Overall Score:** 0.825
- **Individual Scores:**
  - Answer Correctness: 1.000
  - Context Precision: 0.125
  - Context Recall: 1.000
  - Faithfulness: 1.000
- **Source File:** eval_ragas_per_question_batch_4.jsonl
- **Passes Thresholds:** ❌ False
- **Is Table Question:** ✅ True

---

#### 8. **What is the wear depth for case W22 (in μm)?**
- **Agent Answer:** 524 μm [Gear wear Failure.pdf p11 table]
- **Ground Truth:** 524
- **Overall Score:** 0.825
- **Individual Scores:**
  - Answer Correctness: 1.000
  - Context Precision: 0.125
  - Context Recall: 1.000
  - Faithfulness: 1.000
- **Source File:** eval_ragas_per_question_batch_4.jsonl
- **Passes Thresholds:** ❌ False
- **Is Table Question:** ✅ True

---

#### 9. **What is the wear depth for case W24 (in μm)?**
- **Agent Answer:** 579 μm [Gear wear Failure.pdf p11 table]
- **Ground Truth:** 579
- **Overall Score:** 0.825
- **Individual Scores:**
  - Answer Correctness: 1.000
  - Context Precision: 0.125
  - Context Recall: 1.000
  - Faithfulness: 1.000
- **Source File:** eval_ragas_per_question_batch_4.jsonl
- **Passes Thresholds:** ❌ False
- **Is Table Question:** ✅ True

---

#### 10. **What is the wear depth for case W25 (in μm)?**
- **Agent Answer:** 608 μm [Gear wear Failure.pdf p11 table]
- **Ground Truth:** 608
- **Overall Score:** 0.825
- **Individual Scores:**
  - Answer Correctness: 1.000
  - Context Precision: 0.125
  - Context Recall: 1.000
  - Faithfulness: 1.000
- **Source File:** eval_ragas_per_question_batch_4.jsonl
- **Passes Thresholds:** ❌ False
- **Is Table Question:** ✅ True

---

#### 11. **What is the wear depth for case W31 (in μm)?**
- **Agent Answer:** 797 μm [Gear wear Failure.pdf p11 table]
- **Ground Truth:** 797
- **Overall Score:** 0.825
- **Individual Scores:**
  - Answer Correctness: 1.000
  - Context Precision: 0.125
  - Context Recall: 1.000
  - Faithfulness: 1.000
- **Source File:** eval_ragas_per_question_batch_4.jsonl
- **Passes Thresholds:** ❌ False
- **Is Table Question:** ✅ True

---

#### 12. **What is the wear depth for case W1 (in μm)?**
- **Agent Answer:** 40 μm [Gear wear Failure.pdf p11 table]
- **Ground Truth:** 40
- **Overall Score:** 0.825
- **Individual Scores:**
  - Answer Correctness: 1.000
  - Context Precision: 0.125
  - Context Recall: 1.000
  - Faithfulness: 1.000
- **Source File:** eval_ragas_per_question_combined.jsonl
- **Passes Thresholds:** ❌ False
- **Is Table Question:** ✅ True

---

#### 13. **What is the wear depth for case W22 (in μm)?**
- **Agent Answer:** 524 μm [Gear wear Failure.pdf p11 table]
- **Ground Truth:** 524
- **Overall Score:** 0.825
- **Individual Scores:**
  - Answer Correctness: 1.000
  - Context Precision: 0.125
  - Context Recall: 1.000
  - Faithfulness: 1.000
- **Source File:** eval_ragas_per_question_combined.jsonl
- **Passes Thresholds:** ❌ False
- **Is Table Question:** ✅ True

---

#### 14. **What is the wear depth for case W24 (in μm)?**
- **Agent Answer:** 579 μm [Gear wear Failure.pdf p11 table]
- **Ground Truth:** 579
- **Overall Score:** 0.825
- **Individual Scores:**
  - Answer Correctness: 1.000
  - Context Precision: 0.125
  - Context Recall: 1.000
  - Faithfulness: 1.000
- **Source File:** eval_ragas_per_question_combined.jsonl
- **Passes Thresholds:** ❌ False
- **Is Table Question:** ✅ True

---

#### 15. **What is the wear depth for case W25 (in μm)?**
- **Agent Answer:** 608 μm [Gear wear Failure.pdf p11 table]
- **Ground Truth:** 608
- **Overall Score:** 0.825
- **Individual Scores:**
  - Answer Correctness: 1.000
  - Context Precision: 0.125
  - Context Recall: 1.000
  - Faithfulness: 1.000
- **Source File:** eval_ragas_per_question_combined.jsonl
- **Passes Thresholds:** ❌ False
- **Is Table Question:** ✅ True

---

#### 16. **What is the wear depth for case W31 (in μm)?**
- **Agent Answer:** 797 μm [Gear wear Failure.pdf p11 table]
- **Ground Truth:** 797
- **Overall Score:** 0.825
- **Individual Scores:**
  - Answer Correctness: 1.000
  - Context Precision: 0.125
  - Context Recall: 1.000
  - Faithfulness: 1.000
- **Source File:** eval_ragas_per_question_combined.jsonl
- **Passes Thresholds:** ❌ False
- **Is Table Question:** ✅ True

---

### 📊 SUMMARY STATISTICS FOR WEAR DEPTH QUESTIONS

| Metric | Value |
|--------|-------|
| **Average Score** | 0.834 |
| **Score Range** | 0.825 - 0.850 |
| **Questions Passing Thresholds** | 0/16 (0.0%) |

### 🎯 DETAILED BREAKDOWN BY CASE NUMBER

#### **Case W1:**
- **Score:** 0.825 | **Answer:** 40 μm [Gear wear Failure.pdf p11 table] | **GT:** 40 | **Passes:** ❌
  - **Individual Scores:** AC: 1.000, CP: 0.125, CR: 1.000, F: 1.000
  - **Source:** eval_ragas_per_question_batch_4.jsonl

- **Score:** 0.825 | **Answer:** 40 μm [Gear wear Failure.pdf p11 table] | **GT:** 40 | **Passes:** ❌
  - **Individual Scores:** AC: 1.000, CP: 0.125, CR: 1.000, F: 1.000
  - **Source:** eval_ragas_per_question_combined.jsonl

#### **Case W13:**
- **Score:** 0.850 | **Answer:** 344 μm [Gear wear Failure.pdf p11 table] | **GT:** 344 | **Passes:** ❌
  - **Individual Scores:** AC: 1.000, CP: 0.250, CR: 1.000, F: 1.000
  - **Source:** eval_ragas_per_question_batch_4.jsonl

- **Score:** 0.850 | **Answer:** 344 μm [Gear wear Failure.pdf p11 table] | **GT:** 344 | **Passes:** ❌
  - **Individual Scores:** AC: 1.000, CP: 0.250, CR: 1.000, F: 1.000
  - **Source:** eval_ragas_per_question_combined.jsonl

#### **Case W22:**
- **Score:** 0.825 | **Answer:** 524 μm [Gear wear Failure.pdf p11 table] | **GT:** 524 | **Passes:** ❌
  - **Individual Scores:** AC: 1.000, CP: 0.125, CR: 1.000, F: 1.000
  - **Source:** eval_ragas_per_question_batch_4.jsonl

- **Score:** 0.825 | **Answer:** 524 μm [Gear wear Failure.pdf p11 table] | **GT:** 524 | **Passes:** ❌
  - **Individual Scores:** AC: 1.000, CP: 0.125, CR: 1.000, F: 1.000
  - **Source:** eval_ragas_per_question_combined.jsonl

#### **Case W24:**
- **Score:** 0.825 | **Answer:** 579 μm [Gear wear Failure.pdf p11 table] | **GT:** 579 | **Passes:** ❌
  - **Individual Scores:** AC: 1.000, CP: 0.125, CR: 1.000, F: 1.000
  - **Source:** eval_ragas_per_question_batch_4.jsonl

- **Score:** 0.825 | **Answer:** 579 μm [Gear wear Failure.pdf p11 table] | **GT:** 579 | **Passes:** ❌
  - **Individual Scores:** AC: 1.000, CP: 0.125, CR: 1.000, F: 1.000
  - **Source:** eval_ragas_per_question_combined.jsonl

#### **Case W25:**
- **Score:** 0.825 | **Answer:** 608 μm [Gear wear Failure.pdf p11 table] | **GT:** 608 | **Passes:** ❌
  - **Individual Scores:** AC: 1.000, CP: 0.125, CR: 1.000, F: 1.000
  - **Source:** eval_ragas_per_question_batch_4.jsonl

- **Score:** 0.825 | **Answer:** 608 μm [Gear wear Failure.pdf p11 table] | **GT:** 608 | **Passes:** ❌
  - **Individual Scores:** AC: 1.000, CP: 0.125, CR: 1.000, F: 1.000
  - **Source:** eval_ragas_per_question_combined.jsonl

#### **Case W31:**
- **Score:** 0.825 | **Answer:** 797 μm [Gear wear Failure.pdf p11 table] | **GT:** 797 | **Passes:** ❌
  - **Individual Scores:** AC: 1.000, CP: 0.125, CR: 1.000, F: 1.000
  - **Source:** eval_ragas_per_question_batch_4.jsonl

- **Score:** 0.825 | **Answer:** 797 μm [Gear wear Failure.pdf p11 table] | **GT:** 797 | **Passes:** ❌
  - **Individual Scores:** AC: 1.000, CP: 0.125, CR: 1.000, F: 1.000
  - **Source:** eval_ragas_per_question_combined.jsonl

#### **Case W33:**
- **Score:** 0.850 | **Answer:** 853 μm [Gear wear Failure.pdf p11 table] | **GT:** 853 | **Passes:** ❌
  - **Individual Scores:** AC: 1.000, CP: 0.250, CR: 1.000, F: 1.000
  - **Source:** eval_ragas_per_question_batch_5.jsonl

- **Score:** 0.850 | **Answer:** 853 μm [Gear wear Failure.pdf p11 table] | **GT:** 853 | **Passes:** ❌
  - **Individual Scores:** AC: 1.000, CP: 0.250, CR: 1.000, F: 1.000
  - **Source:** eval_ragas_per_question_combined.jsonl

#### **Case W35:**
- **Score:** 0.850 | **Answer:** 932 μm [Gear wear Failure.pdf p11 table] | **GT:** 932 | **Passes:** ❌
  - **Individual Scores:** AC: 1.000, CP: 0.250, CR: 1.000, F: 1.000
  - **Source:** eval_ragas_per_question_batch_5.jsonl

- **Score:** 0.850 | **Answer:** 932 μm [Gear wear Failure.pdf p11 table] | **GT:** 932 | **Passes:** ❌
  - **Individual Scores:** AC: 1.000, CP: 0.250, CR: 1.000, F: 1.000
  - **Source:** eval_ragas_per_question_combined.jsonl


### 🏆 BEST PERFORMING WEAR DEPTH QUESTION

**Question:** What is the wear depth for case W13 (in μm)?
- **Agent Answer:** 344 μm [Gear wear Failure.pdf p11 table]
- **Ground Truth:** 344
- **Overall Score:** 0.850
- **Individual Scores:** AC: 1.000, CP: 0.250, CR: 1.000, F: 1.000
- **Source:** eval_ragas_per_question_batch_4.jsonl
- **Passes:** ❌

### ❌ WORST PERFORMING WEAR DEPTH QUESTION

**Question:** What is the wear depth for case W31 (in μm)?
- **Agent Answer:** 797 μm [Gear wear Failure.pdf p11 table]
- **Ground Truth:** 797
- **Overall Score:** 0.825
- **Individual Scores:** AC: 1.000, CP: 0.125, CR: 1.000, F: 1.000
- **Source:** eval_ragas_per_question_combined.jsonl
- **Passes:** ❌

### 📋 COMPLETE LIST WITH ALL SCORES


** 1. Score: 0.850** - What is the wear depth for case W13 (in μm)?
- **Agent Answer:** 344 μm [Gear wear Failure.pdf p11 table]
- **Ground Truth:** 344
- **Individual Scores:** AC: 1.000, CP: 0.250, CR: 1.000, F: 1.000
- **Source:** eval_ragas_per_question_batch_4.jsonl
- **Passes:** ❌


** 2. Score: 0.850** - What is the wear depth for case W33 (in μm)?
- **Agent Answer:** 853 μm [Gear wear Failure.pdf p11 table]
- **Ground Truth:** 853
- **Individual Scores:** AC: 1.000, CP: 0.250, CR: 1.000, F: 1.000
- **Source:** eval_ragas_per_question_batch_5.jsonl
- **Passes:** ❌


** 3. Score: 0.850** - What is the wear depth for case W35 (in μm)?
- **Agent Answer:** 932 μm [Gear wear Failure.pdf p11 table]
- **Ground Truth:** 932
- **Individual Scores:** AC: 1.000, CP: 0.250, CR: 1.000, F: 1.000
- **Source:** eval_ragas_per_question_batch_5.jsonl
- **Passes:** ❌


** 4. Score: 0.850** - What is the wear depth for case W13 (in μm)?
- **Agent Answer:** 344 μm [Gear wear Failure.pdf p11 table]
- **Ground Truth:** 344
- **Individual Scores:** AC: 1.000, CP: 0.250, CR: 1.000, F: 1.000
- **Source:** eval_ragas_per_question_combined.jsonl
- **Passes:** ❌


** 5. Score: 0.850** - What is the wear depth for case W33 (in μm)?
- **Agent Answer:** 853 μm [Gear wear Failure.pdf p11 table]
- **Ground Truth:** 853
- **Individual Scores:** AC: 1.000, CP: 0.250, CR: 1.000, F: 1.000
- **Source:** eval_ragas_per_question_combined.jsonl
- **Passes:** ❌


** 6. Score: 0.850** - What is the wear depth for case W35 (in μm)?
- **Agent Answer:** 932 μm [Gear wear Failure.pdf p11 table]
- **Ground Truth:** 932
- **Individual Scores:** AC: 1.000, CP: 0.250, CR: 1.000, F: 1.000
- **Source:** eval_ragas_per_question_combined.jsonl
- **Passes:** ❌


** 7. Score: 0.825** - What is the wear depth for case W1 (in μm)?
- **Agent Answer:** 40 μm [Gear wear Failure.pdf p11 table]
- **Ground Truth:** 40
- **Individual Scores:** AC: 1.000, CP: 0.125, CR: 1.000, F: 1.000
- **Source:** eval_ragas_per_question_batch_4.jsonl
- **Passes:** ❌


** 8. Score: 0.825** - What is the wear depth for case W22 (in μm)?
- **Agent Answer:** 524 μm [Gear wear Failure.pdf p11 table]
- **Ground Truth:** 524
- **Individual Scores:** AC: 1.000, CP: 0.125, CR: 1.000, F: 1.000
- **Source:** eval_ragas_per_question_batch_4.jsonl
- **Passes:** ❌


** 9. Score: 0.825** - What is the wear depth for case W24 (in μm)?
- **Agent Answer:** 579 μm [Gear wear Failure.pdf p11 table]
- **Ground Truth:** 579
- **Individual Scores:** AC: 1.000, CP: 0.125, CR: 1.000, F: 1.000
- **Source:** eval_ragas_per_question_batch_4.jsonl
- **Passes:** ❌


**10. Score: 0.825** - What is the wear depth for case W25 (in μm)?
- **Agent Answer:** 608 μm [Gear wear Failure.pdf p11 table]
- **Ground Truth:** 608
- **Individual Scores:** AC: 1.000, CP: 0.125, CR: 1.000, F: 1.000
- **Source:** eval_ragas_per_question_batch_4.jsonl
- **Passes:** ❌


**11. Score: 0.825** - What is the wear depth for case W31 (in μm)?
- **Agent Answer:** 797 μm [Gear wear Failure.pdf p11 table]
- **Ground Truth:** 797
- **Individual Scores:** AC: 1.000, CP: 0.125, CR: 1.000, F: 1.000
- **Source:** eval_ragas_per_question_batch_4.jsonl
- **Passes:** ❌


**12. Score: 0.825** - What is the wear depth for case W1 (in μm)?
- **Agent Answer:** 40 μm [Gear wear Failure.pdf p11 table]
- **Ground Truth:** 40
- **Individual Scores:** AC: 1.000, CP: 0.125, CR: 1.000, F: 1.000
- **Source:** eval_ragas_per_question_combined.jsonl
- **Passes:** ❌


**13. Score: 0.825** - What is the wear depth for case W22 (in μm)?
- **Agent Answer:** 524 μm [Gear wear Failure.pdf p11 table]
- **Ground Truth:** 524
- **Individual Scores:** AC: 1.000, CP: 0.125, CR: 1.000, F: 1.000
- **Source:** eval_ragas_per_question_combined.jsonl
- **Passes:** ❌


**14. Score: 0.825** - What is the wear depth for case W24 (in μm)?
- **Agent Answer:** 579 μm [Gear wear Failure.pdf p11 table]
- **Ground Truth:** 579
- **Individual Scores:** AC: 1.000, CP: 0.125, CR: 1.000, F: 1.000
- **Source:** eval_ragas_per_question_combined.jsonl
- **Passes:** ❌


**15. Score: 0.825** - What is the wear depth for case W25 (in μm)?
- **Agent Answer:** 608 μm [Gear wear Failure.pdf p11 table]
- **Ground Truth:** 608
- **Individual Scores:** AC: 1.000, CP: 0.125, CR: 1.000, F: 1.000
- **Source:** eval_ragas_per_question_combined.jsonl
- **Passes:** ❌


**16. Score: 0.825** - What is the wear depth for case W31 (in μm)?
- **Agent Answer:** 797 μm [Gear wear Failure.pdf p11 table]
- **Ground Truth:** 797
- **Individual Scores:** AC: 1.000, CP: 0.125, CR: 1.000, F: 1.000
- **Source:** eval_ragas_per_question_combined.jsonl
- **Passes:** ❌

