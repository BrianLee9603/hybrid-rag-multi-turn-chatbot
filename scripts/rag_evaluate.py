import os
import sys
import json
import asyncio
from typing import Dict, List
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Add project root to path
sys.path.append(os.getcwd())
load_dotenv()

from app.services.rag_service import rag_service
from app.core.config import settings

client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

async def judge_rag(question: str, context: str, answer: str, ground_truth: str) -> Dict:
    """Uses LLM-as-a-judge to score the RAG triad."""
    prompt = f"""
    You are an expert AI evaluator. Rate the RAG performance based on the following:
    
    QUESTION: {question}
    RETRIVED CONTEXT: {context}
    AI ANSWER: {answer}
    GROUND TRUTH: {ground_truth}
    
    Return a JSON object with scores (0.0 to 1.0) and brief reasoning:
    {{
        "context_precision": <Is the ground truth information present in the retrieved context?>,
        "faithfulness": <Is the AI answer derived ONLY from the retrieved context?>,
        "answer_relevance": <Does the AI answer correctly and fully address the question?>,
        "reasoning": "<Concise explanation>"
    }}
    """
    
    response = await client.chat.completions.create(
        model=settings.EVAL_MODEL,
        messages=[{"role": "system", "content": "You are a precise JSON evaluator."},
                  {"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

async def evaluate_rag():
    with open("tests/eval/rag_golden_set.json", "r") as f:
        golden_set = json.load(f)
    
    print(f"--- Starting RAG Evaluation ({len(golden_set)} cases) ---")
    
    results = []
    
    # 1. Re-index to ensure fresh state
    rag_service.reindex_docs()

    for case in golden_set:
        print(f"Evaluating: {case['id']}...")
        
        # Step A: Retrieve
        context = await rag_service.search(case['question'], n_results=3)
        
        # Step B: Generate (Simulated call to the service logic)
        ans_response = await client.chat.completions.create(
            model=settings.AGENT_MODEL,
            messages=[
                {"role": "system", "content": f"Answer based ONLY on context:\n{context}"},
                {"role": "user", "content": case['question']}
            ]
        )
        answer = ans_response.choices[0].message.content
        
        # Step C: Judge
        score = await judge_rag(case['question'], context, answer, case['ground_truth'])
        
        results.append({
            "id": case['id'],
            "scores": score
        })

    # Summary Statistics
    avg_precision = sum(r['scores'].get('context_precision', 0) for r in results) / len(results)
    avg_faithfulness = sum(r['scores'].get('faithfulness', 0) for r in results) / len(results)
    avg_relevance = sum(r['scores'].get('answer_relevance', 0) for r in results) / len(results)
    
    print("\n" + "="*30)
    print("      RAG BASELINE REPORT")
    print("="*30)
    print(f"Context Precision: {avg_precision:.2f}")
    print(f"Faithfulness:      {avg_faithfulness:.2f}")
    print(f"Answer Relevance:   {avg_relevance:.2f}")
    print("="*30)

if __name__ == "__main__":
    asyncio.run(evaluate_rag())
