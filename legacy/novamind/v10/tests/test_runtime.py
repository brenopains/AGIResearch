import pytest
from novamind.v10.models.runtime import LocalRuntime

def test_local_runtime_loop():
    runtime = LocalRuntime(max_iterations=3)
    
    # Run the loop and collect output responses
    responses = []
    
    # Mocking standard input is complex in pytest, so the runtime accepts a generator
    inputs = ["Wake up", "What is your name?", "Exit"]
    def mock_input_generator():
        for i in inputs:
            yield i
            
    # Should run 3 times and return the outputs generated
    outputs = runtime.start(input_source=mock_input_generator())
    
    assert len(outputs) == 3
    assert "Processed: Wake up" in outputs[0]
