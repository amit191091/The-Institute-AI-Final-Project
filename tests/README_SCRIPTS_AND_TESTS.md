# 📜 **SCRIPTS & TESTS**

## 📋 **OVERVIEW**

This document covers the utility scripts for development and debugging, as well as the comprehensive testing suite that ensures system quality and reliability.

---

## 📜 **SCRIPTS**

### **Purpose**
Utility scripts for development, testing, and debugging the RAG system

### **Location**
`scripts/`

### **Key Scripts**

#### **📄 query_smoke.py**
**Purpose**: Basic query testing for the RAG system

**Features**:
- **Simple Testing**: Basic query functionality testing
- **Quick Validation**: Fast validation of system components
- **Error Detection**: Identify basic system issues
- **Performance Check**: Basic performance validation

#### **📄 minimal_extraction.py**
**Purpose**: Minimal document extraction testing

**Features**:
- **Core Extraction**: Test basic document extraction
- **Format Validation**: Validate extracted content
- **Error Handling**: Test error handling mechanisms
- **Performance Baseline**: Establish performance baselines

#### **📄 lean_extraction.py**
**Purpose**: Lean extraction testing with minimal dependencies

**Features**:
- **Minimal Dependencies**: Test with minimal library requirements
- **Fast Execution**: Quick extraction testing
- **Resource Efficiency**: Low resource usage testing
- **Isolation Testing**: Isolated component testing

#### **📄 inspect_pdfplumber_simple.py**
**Purpose**: PDFPlumber inspection and testing

**Features**:
- **PDFPlumber Testing**: Test PDFPlumber functionality
- **Content Extraction**: Validate content extraction
- **Table Detection**: Test table detection capabilities
- **Error Analysis**: Analyze PDFPlumber errors

#### **📄 inspect_pdfplumber_only.py**
**Purpose**: PDFPlumber-only inspection without other dependencies

**Features**:
- **Pure PDFPlumber**: Test PDFPlumber in isolation
- **Dependency Isolation**: Test without other libraries
- **Core Functionality**: Validate core PDFPlumber features
- **Performance Analysis**: Analyze PDFPlumber performance

#### **📄 debug_table_pipeline.py**
**Purpose**: Table pipeline debugging and testing

**Features**:
- **Pipeline Debugging**: Debug table extraction pipeline
- **Error Identification**: Identify pipeline issues
- **Performance Analysis**: Analyze pipeline performance
- **Optimization Testing**: Test pipeline optimizations

### **Usage**
```bash
# Navigate to scripts folder
cd scripts

# Run basic query testing
python query_smoke.py

# Run minimal extraction
python minimal_extraction.py

# Run lean extraction
python lean_extraction.py

# Test PDFPlumber
python inspect_pdfplumber_simple.py
python inspect_pdfplumber_only.py

# Debug table pipeline
python debug_table_pipeline.py
```

---

## 🧪 **TESTS**

### **Purpose**
Comprehensive testing suite for all system components

### **Location**
`TESTS/`

### **Key Test Categories**

#### **📁 Core Component Tests**
- **`test_core_components.py`**: Core system component testing
- **`test_rag_service.py`**: RAG service testing
- **`test_rag_pipeline.py`**: Pipeline testing
- **`test_rag_cli.py`**: CLI interface testing
- **`test_modular_architecture.py`**: Architecture testing

#### **📁 Integration Tests**
- **`test_integrated_rag_pipeline.py`**: Integrated pipeline testing
- **`test_mcp_client.py`**: MCP client testing
- **`test_google_ragas.py`**: RAGAS evaluation testing

#### **📁 Utility Tests**
- **`test_extractors.py`**: Data extraction testing
- **`test_source_filtering.py`**: Source filtering testing
- **`test_evaluation_targets.py`**: Evaluation target testing
- **`test_normalize_snapshot.py`**: Data normalization testing

#### **📁 End-to-End Tests**
- **`e2e/`**: End-to-end testing scenarios
- **`playwright_smoke.py`**: UI smoke testing
- **`smoke_chunking_test.py`**: Chunking smoke testing

#### **📁 Test Runners**
- **`run_all_tests.py`**: Complete test suite execution
- **`run_core_tests.py`**: Core component test execution
- **`run_smoke.py`**: Smoke test execution
- **`run_tests.py`**: General test execution

### **Test Features**
- ✅ **Comprehensive Coverage**: All system components tested
- ✅ **Automated Testing**: Automated test execution
- ✅ **Performance Testing**: System performance validation
- ✅ **Integration Testing**: Component integration validation
- ✅ **End-to-End Testing**: Complete workflow testing

### **Usage**
```bash
# Navigate to TESTS folder
cd TESTS

# Run all tests
python run_all_tests.py

# Run core tests
python run_core_tests.py

# Run smoke tests
python run_smoke.py

# Run specific test
python test_rag_service.py
```

---

## 🔧 **DEVELOPMENT WORKFLOW**

### **Scripts Workflow**
1. **Development**: Use scripts for quick testing during development
2. **Debugging**: Use scripts to identify and fix issues
3. **Validation**: Use scripts to validate system changes
4. **Performance**: Use scripts to test performance improvements

### **Testing Workflow**
1. **Unit Testing**: Test individual components
2. **Integration Testing**: Test component interactions
3. **System Testing**: Test complete system functionality
4. **Performance Testing**: Test system performance
5. **End-to-End Testing**: Test complete user workflows

---

## 📊 **TESTING STRATEGY**

### **Test Types**
- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test component interactions
- **System Tests**: Test complete system functionality
- **Performance Tests**: Test system performance and scalability
- **End-to-End Tests**: Test complete user workflows

### **Test Coverage**
- **Code Coverage**: Ensure all code paths are tested
- **Feature Coverage**: Test all system features
- **Error Coverage**: Test error handling and edge cases
- **Performance Coverage**: Test performance under various conditions

### **Test Automation**
- **Continuous Integration**: Automated testing on code changes
- **Regression Testing**: Ensure new changes don't break existing functionality
- **Performance Monitoring**: Continuous performance monitoring
- **Quality Gates**: Automated quality checks

---

## 🎯 **USE CASES**

### **Development**
- **Quick Testing**: Fast validation of changes
- **Debugging**: Identify and fix issues quickly
- **Prototyping**: Test new features and ideas
- **Performance Tuning**: Optimize system performance

### **Quality Assurance**
- **Regression Testing**: Ensure system stability
- **Feature Validation**: Validate new features
- **Performance Validation**: Ensure performance requirements
- **Integration Validation**: Validate system integration

### **Deployment**
- **Pre-deployment Testing**: Test before deployment
- **Post-deployment Validation**: Validate after deployment
- **Rollback Testing**: Test rollback procedures
- **Monitoring Setup**: Set up monitoring and alerting

---

## 📈 **PERFORMANCE TESTING**

### **Performance Metrics**
- **Response Time**: Measure system response times
- **Throughput**: Measure system throughput
- **Resource Usage**: Monitor CPU, memory, and disk usage
- **Scalability**: Test system scalability

### **Performance Tools**
- **Profiling**: Use profiling tools to identify bottlenecks
- **Load Testing**: Test system under load
- **Stress Testing**: Test system under stress conditions
- **Benchmarking**: Compare performance against benchmarks

---

## 🔍 **DEBUGGING**

### **Debugging Tools**
- **Logging**: Use comprehensive logging for debugging
- **Error Tracking**: Track and analyze errors
- **Performance Monitoring**: Monitor system performance
- **Resource Monitoring**: Monitor system resources

### **Debugging Techniques**
- **Step-by-Step Debugging**: Debug issues step by step
- **Isolation Testing**: Test components in isolation
- **Comparison Testing**: Compare with working versions
- **Root Cause Analysis**: Analyze root causes of issues

---

## 📚 **DOCUMENTATION**

### **Script Documentation**
- **Inline Comments**: Code documentation and examples
- **Usage Examples**: Example usage scenarios
- **Error Handling**: Error handling documentation
- **Performance Notes**: Performance considerations

### **Test Documentation**
- **Test Descriptions**: Detailed test descriptions
- **Test Data**: Test data and fixtures
- **Expected Results**: Expected test results
- **Troubleshooting**: Test troubleshooting guide

---

## 🚀 **BEST PRACTICES**

### **Script Development**
- **Modular Design**: Create modular and reusable scripts
- **Error Handling**: Implement robust error handling
- **Logging**: Use comprehensive logging
- **Documentation**: Document scripts thoroughly

### **Test Development**
- **Test Isolation**: Ensure tests are isolated
- **Test Data**: Use appropriate test data
- **Test Maintenance**: Maintain tests regularly
- **Test Automation**: Automate test execution

---

## 🎯 **CONCLUSION**

The Scripts and Tests provide **comprehensive development and quality assurance** with:

- **Development Tools**: Quick testing and debugging scripts
- **Quality Assurance**: Comprehensive testing suite
- **Performance Validation**: Performance testing and monitoring
- **Automated Workflows**: Automated testing and validation

**🏆 This ensures high-quality, reliable system development and deployment!**
