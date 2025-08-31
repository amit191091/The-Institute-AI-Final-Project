

# **Instruction Manual: Refactoring a Hybrid RAG System for Multi-Document Scalability**

### **Directive for AI Coding Assistant**

This document provides a comprehensive, step-by-step architectural specification for refactoring an existing single-file, metadata-driven hybrid Retrieval-Augmented Generation (RAG) system. The objective is to evolve this prototype into a robust, scalable, and production-ready application capable of managing and querying multiple, distinct document sources.

The core technology stack consists of Python, ChromaDB for vector storage, Neo4j for graph-based data representation, and Gradio for the user interface. Orchestration is assumed to be handled by a library such as LangChain or a custom implementation.

The guiding principle throughout this refactoring process is to prioritize modification and adaptation of the existing codebase over the creation of entirely new components. The goal is to achieve the desired functionality with minimal, targeted, and elegant code changes. All generated or modified code must adhere to modern software engineering best practices, including comprehensive docstrings, Python type hinting, and clear, modular design.

For optimal performance, it is assumed that the user will have all relevant existing Python scripts (e.g., for ingestion, retrieval, and the application interface) open within the VS Code editor. This provides the necessary context for generating accurate and relevant code modifications, aligning with established best practices for interacting with AI coding assistants.1

## **Section 1: Foundational Schema and Data Model Adjustments for Multi-Tenancy**

The first and most critical phase of this refactoring process is to establish a robust data model that supports multi-tenancy at the document level. This architectural foundation will ensure that data from different source files are logically isolated within the shared databases, preventing context leakage and enabling precise, scoped queries. All subsequent logic for ingestion, retrieval, and evaluation will depend on the successful implementation of this schema.

### **1.1 ChromaDB: Implementing Document-Level Scoping via Metadata**

The vector store must be adapted to distinguish between chunks originating from different source documents. This is achieved by enriching the metadata associated with each vector embedding.

Instruction:  
Modify the data ingestion and chunking process. For every text chunk generated from a source document, a new key-value pair must be added to its associated metadata dictionary. This new field shall be named source\_document\_id. The value for this key must be a unique and consistent identifier for the source document from which the chunk was derived. This identifier must be identical for all chunks originating from the same source file.  
Architectural Rationale:  
The power of ChromaDB in a multi-document RAG system is unlocked through its sophisticated metadata filtering capabilities.3 The  
where clause in a query operation is the primary mechanism for constraining a semantic search to a specific subset of the entire vector space. By systematically embedding a source\_document\_id into the metadata of every chunk, the system effectively creates "virtual collections" or logical partitions for each document within a single physical ChromaDB collection. This is a highly efficient and scalable pattern for implementing multi-tenancy without the overhead of managing numerous separate collections.6

The selection of a suitable value for source\_document\_id is a critical design decision. A simple filename is insufficient, as it is not guaranteed to be unique across different directories or projects (e.g., chapter1.md could exist in multiple document sets). A more robust approach is to generate a deterministic identifier from a unique property of the file. While a content-based hash (e.g., MD5 or SHA256) offers benefits like automatic deduplication, for this system's architecture—where each source document is paired with its own ground truth and Q\&A files—the file's canonical, absolute path is the most stable and intuitive source for the identifier. This ensures that a document is uniquely identified within the system's operational context. This practice of using computed hashes for unique identification is also seen in graph data modeling for ensuring entity uniqueness.7 The instruction is therefore to generate the

source\_document\_id by normalizing and hashing the absolute file path of the source document.

### **1.2 Neo4j: Establishing Provenance with a Document Node**

Parallel to the changes in the vector store, the knowledge graph schema must be evolved to explicitly model the provenance of all extracted information. This creates an anchor point for each document within the graph.

Instruction:  
Introduce a new node label into the Neo4j graph schema: Document. Each Document node must have the following properties:

* document\_id: A string property that stores the unique identifier for the source document. This value **must** be identical to the source\_document\_id used in the ChromaDB metadata for the same source file.  
* file\_path: A string property storing the absolute path to the source file on disk.  
* ingestion\_timestamp: A datetime property recording when the document was last ingested or updated.

Next, modify the graph construction logic. For every Chunk node created, an additional relationship must be created. This relationship should be of type PART\_OF and should originate from the Chunk node and point to the corresponding Document node.

Architectural Rationale:  
A knowledge graph without clear data provenance can quickly become an unmanageable and untrustworthy web of interconnected facts. The Document node serves as a fundamental anchor, a root entity from which all information extracted from a specific file originates. This is a core principle of well-modeled, enterprise-grade knowledge graphs and is central to the GraphRAG paradigm.8 The official  
neo4j-graphrag library documentation explicitly defines this "lexical graph" structure, which consists of Document nodes, Chunk nodes, and the relationships connecting them, such as FROM\_DOCUMENT (analogous to our PART\_OF) and NEXT\_CHUNK.10 This structure enables powerful and contextually relevant queries, such as "Find all entities mentioned in document X" or "Show me the chunks immediately following this one within the same document," which are impossible in a flat vector store.12

This dual-provenance mechanism—implicit via metadata in ChromaDB and explicit via a node and relationship in Neo4j—creates a highly resilient and debuggable system. It forms a bridge between the unstructured, semantic world of vectors and the structured, relational world of the graph. If a vector search returns an anomalous chunk, its source\_document\_id can be used to immediately pivot to the Neo4j graph. From there, one can visualize the corresponding Document node, its neighboring Chunk nodes, and all associated Entity nodes. This provides a complete, structured view of the context surrounding the problematic chunk, dramatically simplifying troubleshooting. This interconnectedness is fundamental to delivering the enhanced explainability and trustworthiness that are the primary advantages of a hybrid GraphRAG architecture.8

## **Section 2: Engineering a Scalable Multi-File Ingestion Pipeline**

With the foundational data models established, the next step is to refactor the existing single-file processing logic into a scalable, automated ingestion pipeline. This pipeline will be responsible for discovering, processing, and managing a directory containing multiple document sets, ensuring that the databases remain synchronized with the source files.

### **2.1 File Discovery and State Management**

A robust system requires an automated way to handle collections of documents rather than relying on manual, one-off script executions.

Instruction:  
Create a new standalone Python script, ingest\_corpus.py. This script will serve as the main entry point for the ingestion pipeline and must perform the following actions:

1. Accept a single command-line argument: the path to a root directory containing the document corpus.  
2. Implement a file discovery mechanism that recursively scans the provided directory. The scanner should identify valid "document sets." A valid set is defined as a group of associated files: a primary source document (e.g., .md, .pdf), a corresponding ground truth file (e.g., source\_filename\_gt.txt), and a question-and-answer file for evaluation (e.g., source\_filename\_qa.json). The script should be able to intelligently group these files based on a clear naming convention.  
3. Implement a state management mechanism to ensure the pipeline is incremental and avoids redundant processing. This can be achieved using a simple JSON file or a lightweight SQLite database located in the root directory. For each source file, this state manager should track its path, its last modification timestamp, and the source\_document\_id assigned to it. Before processing a file, the script must check the state manager. If the file has not been modified since its last successful ingestion, it should be skipped.

Architectural Rationale:  
Production-grade RAG applications demand a systematic and automated approach to data ingestion.15 The pipeline must be designed to be both scalable and incremental, capable of handling updates to the corpus without reprocessing the entire dataset from scratch.15 While large-scale enterprise systems might employ distributed processing frameworks like Ray for massive datasets 17, a well-designed, single-threaded scanner with robust state management provides the necessary efficiency and reliability for this use case and represents a minimal yet powerful evolution of the existing code.

### **2.2 Refactoring the Core Processing Logic**

The existing logic for handling a single file must be modularized to be reusable within the new pipeline.

Instruction:  
Encapsulate the entire existing single-file processing logic within a new function or class method. This processing unit should accept the file path of a source document as its primary input. Its responsibilities are as follows:

1. Generate the unique source\_document\_id from the provided file path using the deterministic method defined in Section 1.1.  
2. Load the document content from the file path. Implement appropriate document loaders for different file types (e.g., Markdown, PDF).16  
3. Chunk the loaded text into smaller, manageable segments. The chunking strategy is a critical optimization layer. While fixed-size chunking is simple, it often breaks semantic boundaries. A more robust default is a format-aware or recursive strategy. For Markdown files, MarkdownHeaderTextSplitter is ideal; for general text, RecursiveCharacterTextSplitter is a strong choice as it attempts to split on natural boundaries like paragraphs and sentences.15  
4. Iterate through the list of generated text chunks. For each chunk, inject the source\_document\_id into its metadata dictionary. This step is the practical implementation of the architectural decision from Section 1.1 and is non-negotiable.  
5. Connect to the ChromaDB client and use the collection.upsert() method to add the chunks to the vector store. Using upsert ensures that re-ingesting a document updates existing chunks rather than creating duplicates.5  
6. Invoke the graph construction logic, passing the list of processed chunks and the source\_document\_id to ensure proper linking in Neo4j.

Architectural Rationale:  
This modularization transforms a monolithic script into a reusable component, a cornerstone of scalable software design. The core processing algorithm remains largely unchanged, but it is now parameterized by the source file, allowing it to be called iteratively by the pipeline controller. The explicit instruction to use a sophisticated chunking strategy is based on the understanding that the quality of retrieved context is highly dependent on the coherence of the chunks themselves. Poor chunking is a common failure point in RAG systems.15

### **2.3 Multi-Document Graph Construction**

The Neo4j ingestion logic must be updated to correctly build the graph structure defined in Section 1.2, ensuring each document's data is properly anchored.

Instruction:  
Modify the function responsible for writing data to Neo4j. This function should now accept the source\_document\_id as an argument. Its execution flow must be as follows:

1. First, execute a Cypher MERGE statement to create the Document node if it does not already exist. The MERGE should be based on the document\_id property to ensure idempotency. Example: MERGE (d:Document {document\_id: $doc\_id}) ON CREATE SET d.file\_path \= $path, d.ingestion\_timestamp \= datetime().  
2. Next, iterate through the list of chunks. For each chunk, create its corresponding Chunk node and then create the (Chunk)--\>(Document) relationship, linking it back to the Document node that was just merged.  
3. The pre-existing logic for entity extraction and relationship creation should proceed as before, but with one key difference: all extracted Entity nodes should now be linked to the Chunk node from which they were extracted, not just exist as free-floating nodes in the graph. This creates a clear, traceable path from an entity back to its precise location in the source text.

Architectural Rationale:  
The use of MERGE is fundamental to creating a robust and idempotent ingestion pipeline. It prevents the creation of duplicate Document nodes if a file is processed multiple times, which is essential for maintaining a clean and accurate graph state.15 This entire process aligns perfectly with the documented patterns of the  
neo4j-graphrag library, which first establishes the lexical graph (Documents and Chunks) before layering on the extracted semantic graph (Entities and their relationships).10 This layered approach ensures the graph is well-structured, queryable, and maintains a clear link to the original source material.

## **Section 3: Refactoring the Hybrid Retrieval Logic for Scoped Queries**

With the data correctly modeled and ingested, the retrieval engine must be updated to leverage the new multi-document schema. This ensures that when a user selects a specific document, the RAG system's response is generated exclusively from the content of that document, preventing context contamination from other sources.

### **3.1 ChromaDB: Activating the Metadata Filter**

The vector search component must be constrained to only search within the vectors belonging to the selected document.

Instruction:  
Locate the collection.query() call within the existing retrieval codebase. This function call must be modified to accept an optional source\_document\_id string parameter.

* If a source\_document\_id is provided, a where filter must be constructed and passed to the collection.query() call. The structure of this filter dictionary should be: where={"source\_document\_id": "the\_provided\_document\_id"}.  
* If the source\_document\_id parameter is None or not provided, the where filter should be omitted, allowing the query to perform a global search across all documents in the collection.

Architectural Rationale:  
This modification is the primary lever for enabling scoped retrieval. It directly employs ChromaDB's powerful metadata filtering feature, which is designed for precisely this type of use case.3 The syntax is straightforward, yet its impact on the system's behavior is profound. By applying this filter, the semantic search is surgically confined to the subset of vectors associated with the chosen document. This is the most effective way to prevent "context bleed"—where relevant chunks from other, unselected documents are retrieved, leading to confusing or incorrect answers. This is a common and critical failure mode in naive multi-document RAG implementations. While the primary user-facing application will always provide a document ID via the UI, designing the backend retrieval function to handle an optional filter makes the system's API more flexible and extensible for potential future use cases, such as a global search feature.

### **3.2 Neo4j: Context-Aware Graph Traversal**

Similarly, the graph-based retrieval component must be constrained to ensure its traversals and entity lookups are limited to the selected document's subgraph.

Instruction:  
Modify the function(s) responsible for generating and executing Cypher queries against the Neo4j database. These functions must also be updated to accept the optional source\_document\_id parameter. All Cypher queries must be rewritten to incorporate this ID as either a starting point for the graph traversal or as a filtering condition within the MATCH clause.  
For example, a previous query that might have looked for an entity directly, such as:  
MATCH (e:Entity {name: $entity\_name}) RETURN e  
Must be refactored to anchor the search to the specified document:  
MATCH (d:Document {document\_id: $doc\_id})\<--(c:Chunk)--\>(e:Entity {name: $entity\_name}) RETURN e  
This revised query ensures that the returned entity is guaranteed to be part of the selected document's context.

Architectural Rationale:  
This change ensures that both prongs of the hybrid retrieval strategy—semantic vector search and structured graph traversal—are operating within the same contextual boundary. By initiating all graph traversals from the specific Document node identified by the source\_document\_id, the system guarantees that any retrieved entities, relationships, or subgraphs are verifiably connected to the selected source material.8 This practice dramatically enhances the factual grounding of the generated response and makes the LLM's reasoning path fully traceable through the graph, which is a key differentiator and advantage of the GraphRAG methodology.12

## **Section 4: Developing the Multi-Document Gradio Interface**

To complete the user-facing functionality, a dynamic and intuitive Gradio interface must be developed. This interface will allow users to select which document they wish to query, thereby controlling the context for the RAG pipeline.

### **4.1 Dynamic Document Selection Dropdown**

The UI must not rely on a hardcoded list of documents. It needs to dynamically reflect the current state of the ingested corpus.

Instruction:  
In the main application script (e.g., app.py), create a gradio.Dropdown component that will serve as the document selector. This dropdown must be populated dynamically when the Gradio application loads. To achieve this, create a new helper function named get\_ingested\_documents(). This function will be responsible for querying the backend databases to retrieve a list of all available documents. It can do this in one of two ways:

1. **Neo4j Query:** Execute a Cypher query like MATCH (d:Document) RETURN d.document\_id AS id, d.file\_path AS path ORDER BY d.file\_path to get a list of all Document nodes.  
2. **ChromaDB Query:** Programmatically query the ChromaDB collection to get all unique values for the source\_document\_id metadata field.

The list of choices for the gr.Dropdown should be populated with the document\_id values returned by this function. For a better user experience, the displayed text for each option in the dropdown could be a more human-readable format, like the base filename extracted from the file\_path.

Architectural Rationale:  
A static, hardcoded list of documents in the UI is not a scalable solution. The interface must be data-driven, automatically updating as new documents are processed by the ingestion pipeline. The gradio.Dropdown component is the ideal UI element for this selection task.21 By populating it dynamically at application startup, the system ensures that the user always sees an up-to-date list of queryable documents without requiring any code changes or application restarts when the corpus is updated.

### **4.2 State Management and UI Event Handling**

The user's selection in the dropdown needs to be communicated to the backend to influence the retrieval logic. This requires proper state management within the Gradio application.

**Instruction:**

1. Instantiate a gradio.State object within the Gradio application scope. This object will be used to store the source\_document\_id of the currently selected document.  
2. Implement an event listener for the document selection dropdown. Specifically, wire the .change() event of the gr.Dropdown component to a Python callback function. This function will receive the newly selected value from the dropdown and its sole responsibility is to update the gr.State object with this new value. The gradio.SelectData event data object can be used to access the selected value efficiently.22  
3. Modify the main query-handling function (the function that is triggered when the user submits a question). This function must now accept the gr.State object as an additional input. Inside this function, the value of the state object (the current source\_document\_id) will be retrieved and passed directly to the refactored hybrid retrieval functions from Section 3\.

Architectural Rationale:  
Gradio applications that involve user-specific context or multi-step interactions require an explicit state management mechanism. The gr.State component is designed for this purpose, acting as a server-side session variable that persists across user interactions. The .change() event listener is the critical piece of "glue" logic that connects the user's action (selecting a document) to the application's internal state.21 This creates a reactive UI where the context for the RAG pipeline is immediately and correctly set as soon as the user makes a selection, ensuring that all subsequent queries are properly scoped.

## **Section 5: Establishing a Comprehensive Evaluation Framework**

To ensure the refactored system is not only functional but also accurate and reliable, a comprehensive, automated evaluation framework is required. This framework will leverage the provided ground truth (GT) and question-answer (Q\&A) files to quantitatively measure the performance of the RAG pipeline on a per-document basis.

### **5.1 Evaluation Data Loader**

A utility is needed to load the evaluation datasets in a structured format.

Instruction:  
Create a new directory named evaluation in the project root. Inside this directory, create a script named utils.py. This script will contain a function load\_eval\_data(document\_id: str). This function will:

1. Take a document\_id as input.  
2. Use this ID to locate the corresponding Q\&A file (e.g., \_qa.json) and ground truth context file (e.g., \_gt.txt) on the file system.  
3. Parse these files and load their contents.  
4. Return a list of structured evaluation cases. Each case should be a dictionary or a data class instance containing keys such as question, ground\_truth\_answer (from the Q\&A file), and ground\_truth\_context (from the GT file).

### **5.2 Implementation of Core RAG Metrics**

The core of the evaluation framework will be a set of functions that measure key aspects of RAG performance using an LLM-as-a-judge pattern.

Instruction:  
In the evaluation directory, create a new script named metrics.py. This script will contain Python functions for calculating the core RAG evaluation metrics. These functions will use a powerful LLM (e.g., GPT-4, Claude 3\) to score the pipeline's output against the ground truth data. Implement the following four functions, providing detailed prompts for the LLM judge within each:

1. calculate\_faithfulness(question: str, generated\_answer: str, retrieved\_context: list\[str\]) \-\> float: This metric measures whether the generated\_answer is factually supported by the retrieved\_context. The LLM judge should be prompted to identify any claims in the answer that are not directly backed by the provided context. The score should reflect the proportion of factual claims.  
2. calculate\_answer\_relevancy(question: str, generated\_answer: str) \-\> float: This metric assesses how well the generated\_answer addresses the original question. The LLM judge should determine if the answer is on-topic, direct, and complete.  
3. calculate\_context\_precision(question: str, ground\_truth\_answer: str, retrieved\_context: list\[str\]) \-\> float: This metric evaluates the signal-to-noise ratio of the retrieved context. The LLM judge should be prompted to determine how many of the retrieved chunks are actually relevant and necessary for answering the question. The score represents the ratio of relevant chunks to the total number of retrieved chunks.  
4. calculate\_context\_recall(question: str, ground\_truth\_answer: str, retrieved\_context: list\[str\]) \-\> float: This metric measures whether the retrieved\_context contains all the information necessary to formulate the ground\_truth\_answer. The LLM judge should compare the ground truth answer with the retrieved context to see if any critical pieces of information are missing.

Architectural Rationale:  
This approach operationalizes the state-of-the-art in RAG evaluation.23 Frameworks like Ragas and DeepEval have popularized these component-wise metrics.26 By implementing them, the system's performance can be diagnosed with high precision.  
**Faithfulness** and **Answer Relevancy** specifically test the performance of the *generator* (the LLM and its prompt), measuring its tendency to hallucinate and its ability to follow instructions, respectively. In contrast, **Context Precision** and **Context Recall** test the performance of the *retriever* (the vector search and graph traversal), measuring its ability to find relevant information without including noise, and its ability to find all necessary information.24 This separation is crucial for identifying bottlenecks and performing targeted optimizations.

### **5.3 Automated Evaluation Script (run\_evaluation.py)**

A master script is needed to orchestrate the entire evaluation process for a given document.

Instruction:  
Create a top-level script named run\_evaluation.py. This script will be the main entry point for running tests and must perform the following steps:

1. Accept a source\_document\_id as a command-line argument. This allows for targeted evaluation of a single document.  
2. Invoke the load\_eval\_data() function from evaluation/utils.py to load the corresponding Q\&A and ground truth dataset.  
3. Iterate through each evaluation case (question) in the loaded dataset.  
4. For each question, execute the full, end-to-end RAG pipeline (retrieval and generation), ensuring that the query is scoped to the provided source\_document\_id. This will produce a generated\_answer and the retrieved\_context.  
5. Call the four metric functions from evaluation/metrics.py to calculate the scores for Faithfulness, Answer Relevancy, Context Precision, and Context Recall for the current question.  
6. Store the results for each question. After processing all questions, aggregate the scores (e.g., calculate the average for each metric).  
7. Print a final summary report to the console in a clean, human-readable Markdown table format.

The design of this evaluation script as a modular, command-line-driven tool provides significant long-term value. It can be integrated into a Continuous Integration (CI) pipeline to automatically test the RAG system's performance after every code change. Furthermore, it can be used as the core component of a hyperparameter tuning workflow. By wrapping this script in an outer loop that modifies RAG parameters (such as chunk size, chunk overlap, or the number of retrieved documents top\_k), one can systematically run evaluations for different configurations and compare the resulting summary tables. This transforms evaluation from a simple, one-off validation check into a powerful engine for continuous, data-driven improvement and optimization—a hallmark of a mature MLOps practice.

**Evaluation Summary Report Table:**

| Question ID | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
| :---- | :---- | :---- | :---- | :---- |
| Q1 | 1.00 | 1.00 | 0.75 | 0.90 |
| Q2 | 0.85 | 0.90 | 1.00 | 0.80 |
| Q3 | 1.00 | 1.00 | 0.50 | 1.00 |
| **Average** | **0.95** | **0.97** | **0.75** | **0.90** |

## **Section 6: Execution Guide for VS Code Copilot**

To ensure the successful application of these instructions, a strategic approach to interacting with the AI coding assistant is recommended. This section provides a meta-guide for the developer orchestrating the refactoring process.

### **6.1 Setting the Context**

The quality of the AI assistant's output is directly proportional to the quality and quantity of the context it is given.

Instruction:  
Before beginning the refactoring process, open all relevant existing project files in the VS Code editor. This includes the current scripts for ingestion, querying, data modeling, and the application interface. By having these files open, the AI assistant can analyze the existing code, understand its structure and style, and generate modifications that are consistent and well-integrated. This is a fundamental best practice for maximizing the effectiveness of in-editor AI assistants.1

### **6.2 Section-by-Section Execution**

Attempting to execute this entire set of instructions in a single prompt will likely lead to suboptimal or incomplete results. A methodical, section-by-section approach is far more effective.

Instruction:  
Do not paste this entire Markdown document into the chat interface at once. Instead, proceed through the document sequentially, section by section.

1. Copy the complete text of "Section 1: Foundational Schema and Data Model Adjustments for Multi-Tenancy."  
2. Paste this text into the Copilot chat prompt.  
3. Carefully review the code modifications and new code generated by the assistant.  
4. Once satisfied with the changes for Section 1, commit them to version control.  
5. Proceed to Section 2 and repeat the process.  
   This iterative workflow breaks down the complex overall task into a series of manageable, logically isolated sub-tasks. This approach is a core tenet of effective prompt engineering, making the process easier to manage, review, and debug.2

### **6.3 Iterative Refinement and Validation**

The first response from the AI assistant is a starting point, not necessarily the final product. Iterative refinement is key to achieving a high-quality outcome.

Instruction:  
After the AI assistant generates code for any given section, critically review it. If it does not fully meet the requirements or could be improved, engage in a conversational feedback loop. Use follow-up prompts to guide the assistant toward the desired result. Examples of such prompts include:

* "Please add Python type hints to the function you just created."  
* "Can you refactor this logic to be more resilient to file-not-found errors by using a try-except block?"  
* "Explain the purpose of this specific Cypher clause to me."  
* "Ensure all new functions include a comprehensive docstring explaining their parameters and return values."

This iterative dialogue leverages the conversational context of the AI assistant to refine the code progressively.1 After completing the implementation of Section 5, it is crucial to run the

run\_evaluation.py script. This provides an empirical validation that the entire refactored system is functioning correctly and meeting the desired performance benchmarks before considering the task complete. This continuous cycle of implementation, testing, and refinement is essential for building robust and reliable software.

#### **Works cited**

1. Prompt engineering for Copilot Chat \- Visual Studio Code, accessed August 31, 2025, [https://code.visualstudio.com/docs/copilot/chat/prompt-crafting](https://code.visualstudio.com/docs/copilot/chat/prompt-crafting)  
2. Best practices for using GitHub Copilot, accessed August 31, 2025, [https://docs.github.com/en/copilot/get-started/best-practices](https://docs.github.com/en/copilot/get-started/best-practices)  
3. Metadata Filtering \- Chroma Docs, accessed August 31, 2025, [https://docs.trychroma.com/docs/querying-collections/metadata-filtering](https://docs.trychroma.com/docs/querying-collections/metadata-filtering)  
4. Filters \- Chroma Cookbook, accessed August 31, 2025, [https://cookbook.chromadb.dev/core/filters/](https://cookbook.chromadb.dev/core/filters/)  
5. ChromaDB: Semantic Search with Metadata Filters Using Python | by Sachin Sangal, accessed August 31, 2025, [https://medium.com/@sangal.sachin/chromadb-semantic-search-with-metadata-filters-using-python-456887e5e0cd](https://medium.com/@sangal.sachin/chromadb-semantic-search-with-metadata-filters-using-python-456887e5e0cd)  
6. Metadata Filtering \- LlamaIndex.TS, accessed August 31, 2025, [https://next.ts.llamaindex.ai/docs/llamaindex/modules/rag/query\_engines/metadata\_filtering](https://next.ts.llamaindex.ai/docs/llamaindex/modules/rag/query_engines/metadata_filtering)  
7. Under the Covers With LightRAG: Extraction \- Graph Database & Analytics \- Neo4j, accessed August 31, 2025, [https://neo4j.com/blog/developer/under-the-covers-with-lightrag-extraction/](https://neo4j.com/blog/developer/under-the-covers-with-lightrag-extraction/)  
8. How to Improve Multi-Hop Reasoning With Knowledge Graphs and LLMs \- Neo4j, accessed August 31, 2025, [https://neo4j.com/blog/genai/knowledge-graph-llm-multi-hop-reasoning/](https://neo4j.com/blog/genai/knowledge-graph-llm-multi-hop-reasoning/)  
9. Implementing Advanced Retrieval RAG Strategies With Neo4j, accessed August 31, 2025, [https://neo4j.com/blog/developer/advanced-rag-strategies-neo4j/](https://neo4j.com/blog/developer/advanced-rag-strategies-neo4j/)  
10. API Documentation — neo4j-graphrag-python documentation, accessed August 31, 2025, [https://neo4j.com/docs/neo4j-graphrag-python/current/api.html](https://neo4j.com/docs/neo4j-graphrag-python/current/api.html)  
11. User Guide: Knowledge Graph Builder — neo4j-graphrag-python documentation, accessed August 31, 2025, [https://neo4j.com/docs/neo4j-graphrag-python/current/user\_guide\_kg\_builder.html](https://neo4j.com/docs/neo4j-graphrag-python/current/user_guide_kg_builder.html)  
12. RAG Tutorial: How to Build a RAG System on a Knowledge Graph \- Neo4j, accessed August 31, 2025, [https://neo4j.com/blog/developer/rag-tutorial/](https://neo4j.com/blog/developer/rag-tutorial/)  
13. Building an Agentic GraphRAG System with LangGraph and Neo4j, accessed August 31, 2025, [https://ai.plainenglish.io/building-a-graphrag-multi-agent-system-with-langgraph-and-neo4j-08fc2e2cb64c](https://ai.plainenglish.io/building-a-graphrag-multi-agent-system-with-langgraph-and-neo4j-08fc2e2cb64c)  
14. Enhancing the Accuracy of RAG Applications With Knowledge Graphs | by Tomaz Bratanic | Neo4j Developer Blog | Medium, accessed August 31, 2025, [https://medium.com/neo4j/enhancing-the-accuracy-of-rag-applications-with-knowledge-graphs-ad5e2ffab663](https://medium.com/neo4j/enhancing-the-accuracy-of-rag-applications-with-knowledge-graphs-ad5e2ffab663)  
15. Build an unstructured data pipeline for RAG | Databricks on AWS, accessed August 31, 2025, [https://docs.databricks.com/aws/en/generative-ai/tutorials/ai-cookbook/quality-data-pipeline-rag](https://docs.databricks.com/aws/en/generative-ai/tutorials/ai-cookbook/quality-data-pipeline-rag)  
16. RAG Pipeline: Example, Tools & How to Build It \- lakeFS, accessed August 31, 2025, [https://lakefs.io/blog/what-is-rag-pipeline/](https://lakefs.io/blog/what-is-rag-pipeline/)  
17. Build a RAG data ingestion pipeline for large-scale ML workloads \- AWS, accessed August 31, 2025, [https://aws.amazon.com/blogs/big-data/build-a-rag-data-ingestion-pipeline-for-large-scale-ml-workloads/](https://aws.amazon.com/blogs/big-data/build-a-rag-data-ingestion-pipeline-for-large-scale-ml-workloads/)  
18. Getting Started \- Chroma Docs, accessed August 31, 2025, [https://docs.trychroma.com/getting-started](https://docs.trychroma.com/getting-started)  
19. Multimodal Data Ingestion in RAG: A Practical Guide \- Reddit, accessed August 31, 2025, [https://www.reddit.com/r/Rag/comments/1m5ev9g/multimodal\_data\_ingestion\_in\_rag\_a\_practical\_guide/](https://www.reddit.com/r/Rag/comments/1m5ev9g/multimodal_data_ingestion_in_rag_a_practical_guide/)  
20. Knowledge Graph Extraction and Challenges \- Graph Database & Analytics \- Neo4j, accessed August 31, 2025, [https://neo4j.com/blog/developer/knowledge-graph-extraction-challenges/](https://neo4j.com/blog/developer/knowledge-graph-extraction-challenges/)  
21. Gradio docs – Dropdown, accessed August 31, 2025, [https://www.gradio.app/docs/gradio/dropdown](https://www.gradio.app/docs/gradio/dropdown)  
22. SelectData \- Gradio Docs, accessed August 31, 2025, [https://www.gradio.app/docs/gradio/selectdata](https://www.gradio.app/docs/gradio/selectdata)  
23. every LLM metric you need to know : r/LangChain \- Reddit, accessed August 31, 2025, [https://www.reddit.com/r/LangChain/comments/1j3gllj/every\_llm\_metric\_you\_need\_to\_know/](https://www.reddit.com/r/LangChain/comments/1j3gllj/every_llm_metric_you_need_to_know/)  
24. RAG Evaluation Metrics: Assessing Answer ... \- Confident AI, accessed August 31, 2025, [https://www.confident-ai.com/blog/rag-evaluation-metrics-answer-relevancy-faithfulness-and-more](https://www.confident-ai.com/blog/rag-evaluation-metrics-answer-relevancy-faithfulness-and-more)  
25. RAG Evaluation Metrics: Best Practices for Evaluating RAG Systems \- Patronus AI, accessed August 31, 2025, [https://www.patronus.ai/llm-testing/rag-evaluation-metrics](https://www.patronus.ai/llm-testing/rag-evaluation-metrics)  
26. List of available metrics \- Ragas, accessed August 31, 2025, [https://docs.ragas.io/en/stable/concepts/metrics/available\_metrics/](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/)  
27. Metrics \- Ragas, accessed August 31, 2025, [https://docs.ragas.io/en/v0.1.21/concepts/metrics/](https://docs.ragas.io/en/v0.1.21/concepts/metrics/)