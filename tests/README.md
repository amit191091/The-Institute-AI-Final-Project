# RAG System Tests

This directory contains comprehensive unit tests for the RAG (Retrieval Augmented Generation) system.

## Test Structure

```
RAG/tests/
├── conftest.py                    # Pytest configuration and common fixtures
├── test_basic.py                  # Basic infrastructure tests
├── test_rag_service.py           # RAG service layer tests
├── test_rag_cli.py               # RAG CLI tests
├── test_core_components.py       # Core RAG component tests
├── run_tests.py                  # Test runner script
├── run_smoke.py                  # Smoke test runner (existing)
├── smoke_chunking_test.py        # Chunking functionality test (existing)
├── test_normalize_snapshot.py    # Normalization script test (existing)
└── README.md                     # Comprehensive test documentation
```

## Test Coverage

### 1. RAG Service Tests (`test_rag_service.py`)
- **RAGService Class**: Complete service layer functionality
- **Document Loading**: Loading from various sources
- **Document Processing**: Chunking and metadata attachment
- **Index Building**: Dense and sparse index creation
- **Pipeline Execution**: Full RAG pipeline workflow
- **Query Processing**: Question answering with and without agents
- **System Evaluation**: RAGAS evaluation functionality
- **Status Checking**: System status and health monitoring
- **Error Handling**: Comprehensive error scenarios
- **Edge Cases**: Empty documents, large datasets, special characters
- **Concurrent Access**: Multi-threading safety

### 2. RAG CLI Tests (`test_rag_cli.py`)
- **CLI Commands**: All command-line interface functions
- **Build Command**: Pipeline building with various options
- **Query Command**: Question processing and response formatting
- **Evaluate Command**: System evaluation with sample data
- **Status Command**: System status reporting
- **Clean Command**: Output cleanup functionality
- **Error Handling**: Invalid inputs and error scenarios
- **Integration**: Full CLI workflow testing
- **Concurrent CLI**: Multiple CLI calls simultaneously

### 3. Core Components Tests (`test_core_components.py`)
- **Indexing**: Dense and sparse index creation
- **Agents**: Question routing and agent selection
- **Retrieval**: Document retrieval and filtering
- **Chunking**: Document chunking strategies
- **Metadata**: Metadata attachment and management
- **Validation**: Input validation and error checking
- **Integration**: Component interaction testing

### 4. Basic Infrastructure Tests (`test_basic.py`)
- **Import Testing**: Verify all imports work correctly
- **Project Structure**: Check essential directories and files
- **Fixture Testing**: Verify test fixtures work properly
- **Environment Setup**: Confirm test environment configuration

### 5. Existing Tests (Legacy/Integration)

#### Smoke Tests (`run_smoke.py`, `smoke_chunking_test.py`)
- **Chunking Functionality**: Tests document chunking structure
- **Figure/Table Processing**: Verifies proper anchor generation
- **ID Assignment**: Checks chunk_id, doc_id, content_hash creation
- **Associated Text**: Tests figure-text linking
- **Quick Health Check**: Fast validation of core chunking

#### Normalization Script Test (`test_normalize_snapshot.py`)
- **Script Integration**: Tests normalize_snapshot.py as subprocess
- **Output Validation**: Verifies chunks.jsonl and graph.json creation
- **Content Verification**: Checks for expected figures (1-4) and tables (1-3)
- **Field Validation**: Ensures chunks have required fields (id, document_id, text, type)
- **Graph Structure**: Validates document graph structure

## Running Tests

### Prerequisites
```bash
pip install pytest pytest-cov
```

### Quick Start
```bash
# Run all comprehensive tests
python RAG/tests/run_tests.py

# Run specific test suite
python RAG/tests/run_tests.py basic
python RAG/tests/run_tests.py service
python RAG/tests/run_tests.py cli
python RAG/tests/run_tests.py components

# Run existing smoke tests
python RAG/tests/run_smoke.py

# Run existing normalization test
python -m pytest RAG/tests/test_normalize_snapshot.py -v

# Run with pytest directly
python -m pytest RAG/tests/ -v

# Run with coverage
python -m pytest RAG/tests/ --cov=RAG --cov-report=html
```

### Test Runner Options
```bash
# Run all tests with summary
python RAG/tests/run_tests.py all

# Run specific test quietly
python RAG/tests/run_tests.py service --quiet

# Help
python RAG/tests/run_tests.py --help
```

## Test Fixtures

### Common Fixtures (in `conftest.py`)
- **`temp_project_dir`**: Temporary project directory for testing
- **`sample_documents`**: Tiny sample documents for testing
- **`sample_eval_data`**: Sample evaluation dataset
- **`mock_embedding_function`**: Mock embedding function
- **`mock_llm`**: Mock language model
- **`setup_test_environment`**: Test environment configuration

### Usage in Tests
```python
def test_example(sample_documents, mock_embedding_function):
    """Example test using fixtures."""
    service = RAGService()
    result = service.process_documents(sample_documents)
    assert len(result) == 3
```

## Test Data

### Sample Documents
- **test_doc1.pdf**: Basic gear wear analysis document
- **test_doc2.pdf**: Document with wear depth measurements
- **test_doc3.pdf**: Document with figures and tables

### Sample Evaluation Data
- **Question**: "What is the wear depth for case W15?"
- **Ground Truth**: ["400μm", "400 microns"]
- **Expected Answer**: "The wear depth for case W15 is 400μm."

## Test Categories

### Comprehensive Unit Tests (New)
- **Individual function testing** with mocked dependencies
- **Isolated functionality** for fast execution
- **Complete coverage** of RAG service and CLI
- **Professional quality** with proper error handling

### Basic Infrastructure Tests (New)
- **Import validation** and project structure verification
- **Test environment setup** and fixture validation
- **Foundation testing** to ensure test system works

### Existing Integration Tests (Legacy)
- **Smoke Tests**: Quick chunking functionality validation
- **Script Integration**: Testing normalize_snapshot.py as subprocess
- **Snapshot Testing**: Comparing output to expected results
- **Real Data Processing**: Using actual document structures

### Test Types by Purpose
- **Unit Tests**: Individual component testing (fast, isolated)
- **Integration Tests**: Component interaction testing (medium speed)
- **Smoke Tests**: Quick health checks (fastest)
- **Snapshot Tests**: Output validation (medium speed)
- **Edge Case Tests**: Error conditions and boundary testing

## Best Practices

### Test Design
- **Tiny Fixtures**: Use minimal test data for fast execution
- **Mocking**: Mock heavy dependencies (LLMs, embeddings)
- **Isolation**: Each test should be independent
- **Descriptive Names**: Clear test function names
- **Documentation**: Docstrings for complex tests

### Test Organization
- **Group Related Tests**: Use test classes for related functionality
- **Setup/Teardown**: Use fixtures for common setup
- **Error Testing**: Test both success and failure scenarios
- **Coverage**: Aim for high test coverage

### Performance
- **Fast Execution**: Tests should run quickly
- **Minimal Dependencies**: Avoid heavy imports in tests
- **Efficient Fixtures**: Reuse fixtures when possible
- **Parallel Execution**: Tests should be parallelizable

## Continuous Integration

### GitHub Actions (Example)
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: python RAG/tests/run_tests.py all
```

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure project root is in `sys.path`
2. **Missing Dependencies**: Install `pytest` and `pytest-cov`
3. **Path Issues**: Use absolute paths in test configuration
4. **Environment Variables**: Set test environment in fixtures

### Debug Mode
```bash
# Run with debug output
python -m pytest RAG/tests/ -v -s

# Run specific test with debug
python -m pytest RAG/tests/test_rag_service.py::TestRAGService::test_initialization -v -s
```

## Contributing

### Adding New Tests
1. Create test file in `RAG/tests/`
2. Use existing fixtures from `conftest.py`
3. Follow naming conventions
4. Add to test runner if needed
5. Update this README

### Test Guidelines
- Write tests for new functionality
- Maintain test coverage above 80%
- Use descriptive test names
- Include both positive and negative tests
- Mock external dependencies
- Keep tests fast and reliable
