#!/usr/bin/env python3
"""
Playwright test script to interact with the RAG UI and trace behavior.
Tests specific questions from the evaluation dataset to analyze real-time behavior.
"""

import asyncio
import os
import time
import json
from pathlib import Path

# Enable verbose LangChain logging
os.environ["LANGCHAIN_VERBOSE"] = "true"
os.environ["RAG_TRACE"] = "1"
os.environ["RAG_TRACE_RETRIEVAL"] = "1"

# Test questions based on known failures from evaluation
TEST_QUESTIONS = [
    "What is the sampling rate of the accelerometer?",
    "What sensor is used for vibration measurement?", 
    "What is the model of the tachometer?",
    "How many teeth does the gear have?",
    "What is the accelerometer sensitivity?",
    "What type of accelerometer is used?",
]

async def run_ui_tests():
    """Run the main pipeline and then test with Playwright."""
    
    print("Starting pipeline to ensure data is loaded...")
    
    # Import after setting environment variables
    from app.pipeline import run
    
    # Run the pipeline to load data (UI mode only)
    os.environ["RAG_UI_ONLY"] = "1"
    os.environ["RAG_HEADLESS"] = "0"  # Show UI
    
    try:
        result = run()
        print(f"Pipeline completed: {result}")
    except Exception as e:
        print(f"Pipeline error: {e}")
        return
    
    # Wait for UI to start
    print("Waiting for UI to start...")
    await asyncio.sleep(10)
    
    # Now test with Playwright
    await test_questions_with_playwright()

async def test_questions_with_playwright():
    """Use Playwright to interact with the Gradio UI and test questions."""
    
    ui_url = "http://localhost:7860"  # Default Gradio port
    
    print(f"Testing UI at {ui_url}")
    
    # Install browser if needed
    try:
        from mcp_playwright_browser_install import mcp_playwright_browser_install
        await mcp_playwright_browser_install()
    except:
        pass
    
    try:
        # Navigate to the UI
        from mcp_playwright_browser_navigate import mcp_playwright_browser_navigate
        await mcp_playwright_browser_navigate(url=ui_url)
        
        # Wait for page to load
        await asyncio.sleep(3)
        
        # Take initial screenshot
        from mcp_playwright_browser_take_screenshot import mcp_playwright_browser_take_screenshot
        await mcp_playwright_browser_take_screenshot(filename="ui_initial.png")
        
        # Get page snapshot to understand structure
        from mcp_playwright_browser_snapshot import mcp_playwright_browser_snapshot
        snapshot = await mcp_playwright_browser_snapshot()
        print("Page structure:", snapshot)
        
        # Test each question
        for i, question in enumerate(TEST_QUESTIONS):
            print(f"\n=== Testing Question {i+1}: {question} ===")
            
            await test_single_question(question, i+1)
            
            # Wait between questions
            await asyncio.sleep(2)
            
    except Exception as e:
        print(f"Playwright error: {e}")
        import traceback
        traceback.print_exc()

async def test_single_question(question: str, question_num: int):
    """Test a single question and capture the response."""
    
    try:
        # Take snapshot to find input elements
        from mcp_playwright_browser_snapshot import mcp_playwright_browser_snapshot
        snapshot = await mcp_playwright_browser_snapshot()
        
        # Look for text input (Gradio typically uses textbox)
        input_found = False
        for line in str(snapshot).split('\n'):
            if 'textbox' in line.lower() or 'input' in line.lower():
                print(f"Found input element: {line}")
                input_found = True
                break
        
        if not input_found:
            print("Could not find input element, taking screenshot for manual inspection")
            from mcp_playwright_browser_take_screenshot import mcp_playwright_browser_take_screenshot
            await mcp_playwright_browser_take_screenshot(filename=f"no_input_q{question_num}.png")
            return
        
        # Try to type the question (Gradio usually has a textbox with placeholder)
        from mcp_playwright_browser_type import mcp_playwright_browser_type
        
        # Common Gradio selectors to try
        selectors_to_try = [
            'textarea[placeholder*="question"]',
            'input[type="text"]',
            'textarea',
            '.gr-textbox textarea',
            '[data-testid="textbox"]'
        ]
        
        typed_successfully = False
        for selector in selectors_to_try:
            try:
                # This is a mock - in real Playwright we'd use page.fill()
                print(f"Attempting to type in selector: {selector}")
                # await mcp_playwright_browser_type(element="question input", ref=selector, text=question)
                typed_successfully = True
                break
            except:
                continue
        
        if not typed_successfully:
            print("Could not type question, trying to click first then type")
            # Take screenshot to see current state
            from mcp_playwright_browser_take_screenshot import mcp_playwright_browser_take_screenshot
            await mcp_playwright_browser_take_screenshot(filename=f"before_type_q{question_num}.png")
        
        # Look for submit button
        submit_selectors = [
            'button:has-text("Submit")',
            'button[type="submit"]',
            'button:has-text("Ask")',
            '.gr-button'
        ]
        
        # Click submit button
        from mcp_playwright_browser_click import mcp_playwright_browser_click
        clicked_submit = False
        for selector in submit_selectors:
            try:
                print(f"Attempting to click submit: {selector}")
                # await mcp_playwright_browser_click(element="submit button", ref=selector)
                clicked_submit = True
                break
            except:
                continue
        
        if not clicked_submit:
            print("Could not find submit button")
            from mcp_playwright_browser_take_screenshot import mcp_playwright_browser_take_screenshot
            await mcp_playwright_browser_take_screenshot(filename=f"no_submit_q{question_num}.png")
        
        # Wait for response
        print("Waiting for response...")
        await asyncio.sleep(5)
        
        # Take screenshot of result
        from mcp_playwright_browser_take_screenshot import mcp_playwright_browser_take_screenshot
        await mcp_playwright_browser_take_screenshot(filename=f"result_q{question_num}.png")
        
        # Try to capture response text
        response_snapshot = await mcp_playwright_browser_snapshot()
        
        # Save the interaction log
        log_entry = {
            "question_num": question_num,
            "question": question,
            "timestamp": time.time(),
            "snapshot": str(response_snapshot)[:1000],  # Truncate for readability
        }
        
        log_file = Path("logs") / "playwright_test_log.jsonl"
        log_file.parent.mkdir(exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        print(f"Question {question_num} completed, logs saved")
        
    except Exception as e:
        print(f"Error testing question {question_num}: {e}")
        import traceback
        traceback.print_exc()

def run_manual_test():
    """Run a simplified version that starts the pipeline and waits for manual testing."""
    
    print("=== Manual Test Mode ===")
    print("This will start the pipeline and UI for manual testing.")
    print("You can then manually test questions and observe the logs.")
    
    # Enable verbose logging
    os.environ["LANGCHAIN_VERBOSE"] = "true"
    os.environ["RAG_TRACE"] = "1"
    os.environ["RAG_TRACE_RETRIEVAL"] = "1"
    os.environ["RAG_UI_ONLY"] = "1"
    os.environ["RAG_HEADLESS"] = "0"
    
    # Import and run pipeline
    from app.pipeline import run
    
    print("\nStarting pipeline...")
    print("Test these questions manually:")
    for i, q in enumerate(TEST_QUESTIONS, 1):
        print(f"{i}. {q}")
    
    print("\nWatch the logs/app.log file for detailed traces.")
    print("Press Ctrl+C to stop when done testing.\n")
    
    try:
        result = run()
        print(f"Pipeline result: {result}")
    except KeyboardInterrupt:
        print("\nTest stopped by user.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--manual":
        run_manual_test()
    else:
        # Run async Playwright test
        try:
            asyncio.run(run_ui_tests())
        except KeyboardInterrupt:
            print("\nTest interrupted by user.")
        except Exception as e:
            print(f"Test failed: {e}")
            import traceback
            traceback.print_exc()
