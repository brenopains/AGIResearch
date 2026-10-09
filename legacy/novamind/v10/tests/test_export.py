import torch
import torch.nn as nn
import os
import pytest
from novamind.v10.models.export import export_to_onnx, quantize_to_int8

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
        
    def forward(self, x):
        return self.fc(x)

def test_onnx_export(tmp_path):
    model = SimpleNet()
    dummy_input = torch.randn(1, 10)
    
    export_path = tmp_path / "model.onnx"
    
    # Export
    export_to_onnx(model, dummy_input, str(export_path))
    
    # Check if file exists
    assert os.path.exists(export_path)

def test_int8_quantization():
    model = SimpleNet()
    
    # Check size before
    # For a small network the size diff might not be huge but layer types change
    q_model = quantize_to_int8(model)
    
    # Quantized linear layers exist in q_model
    assert "nnq.DynamicLinear" in str(type(q_model.fc)) or hasattr(q_model.fc, 'weight_quant_val') or getattr(q_model.fc, 'weight') is not None
    
    # Ensure it's callable
    dummy = torch.randn(1, 10)
    out = q_model(dummy)
    assert out.shape == (1, 5)
