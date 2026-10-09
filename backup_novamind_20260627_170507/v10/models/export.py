import torch
import torch.nn as nn

def export_to_onnx(model: nn.Module, dummy_input: torch.Tensor, save_path: str):
    """
    Exports a PyTorch model to ONNX format.
    """
    model.eval()
    
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            save_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        
def quantize_to_int8(model: nn.Module) -> nn.Module:
    """
    Applies dynamic INT8 quantization to the model to reduce size and increase CPU throughput.
    """
    import torch.ao.quantization
    
    model.eval()
    
    # PyTorch Dynamic Quantization targets specific modules (e.g. Linear)
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model, 
        {nn.Linear, nn.GRUCell}, # Target layers typical for our architectures
        dtype=torch.qint8
    )
    
    return quantized_model
