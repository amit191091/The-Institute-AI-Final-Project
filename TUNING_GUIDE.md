

## 10. Testing the LLM-based Agent System

The pipeline includes an advanced agent system with an LLM-based router that directs questions to the most appropriate sub-agent (Summary, Needle, or Table). It is crucial to test this system to ensure that the routing is accurate and that each sub-agent is performing as expected.

### Step 1: Enable the LLM-based Router

To use the LLM-based router, you need to enable it via the following environment variable in your `.env` file:

```
RAG_USE_LLM_ROUTER=1
```

By default, the router uses a heuristic-based approach. Enabling this flag will switch to the more advanced LLM-based router defined in `app/router_chain.py`.

### Step 2: Test the Router

The primary goal is to ensure that the router correctly identifies the user's intent and forwards the question to the appropriate sub-agent.

**How to test:**

Use the Gradio UI to ask different types of questions and observe the `route=...` part of the log output in the console.

*   **Summary Questions**: Ask broad, open-ended questions that require a summary of information. The route should be `summary`.
    *   *Example*: "Summarize the main failure modes discussed in the document."
*   **Needle Questions**: Ask for specific facts, figures, or details. The route should be `needle`.
    *   *Example*: "What was the viscosity grade of the lubricant?"
*   **Table Questions**: Ask questions that specifically refer to tables or require data that is likely to be in a tabular format. The route should be `table`.
    *   *Example*: "What is the wear depth for case W24 in table 3?"

**What to look for:**

*   Check the `route` in the log output to see if it matches your expectation for the type of question you asked.
*   If the routing is incorrect, you may need to adjust the prompts in `app/prompts.py` or the logic in `app/router_chain.py` to better guide the router.

### Step 3: Test the Sub-Agents

Once you have verified that the routing is working correctly, you should test the performance of each sub-agent.

*   **Summary Agent (`answer_summary`)**: Check if the summaries are concise, accurate, and relevant to the question.
*   **Needle Agent (`answer_needle`)**: Verify that the agent extracts the correct and precise information from the context. It should also include the citation.
*   **Table Agent (`answer_table`)**: Test this agent with questions about specific values in the tables. Ensure that it can correctly parse the tables and extract the requested data.

By systematically testing the router and the sub-agents, you can ensure that the entire LLM-based agent system is robust and provides accurate answers to a wide range of questions.