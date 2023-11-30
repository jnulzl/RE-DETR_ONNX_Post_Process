import argparse
import os
import sys
import onnx
import torch.onnx
import torch.nn as nn
import onnxsim

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', required=True, type=int, default=1, help='Batch size of onnx model.')
    parser.add_argument('--total_obj_num', required=False, type=int, default=300, help='total obj num.')
    parser.add_argument('--output_obj_num', required=True, type=int, default=50, help='Output obj num.')       
    parser.add_argument('--opset_version', required=False, type=int, default=16, help='Opser version of onnx model.')       
    parser.add_argument('--save_file', required=True, help='Path to save the new onnx model.')
    return parser.parse_args()

    
class PoseIncludePreProcess(nn.Module):
    def __init__(self):
        super(PoseIncludePreProcess, self).__init__()
        pass

    def forward(self, boxes):
        # ......        
        return boxes[:,:args.output_obj_num,:]
                
if __name__ == '__main__':
    args = parse_arguments()
    
    torch_model = PoseIncludePreProcess()         
    x = torch.randn(args.batch_size, args.total_obj_num, 6)    
    y = torch_model(x)
    print(y.shape)

    onnx_path = args.save_file
    # Export the model
    torch.onnx.export(torch_model,               # model being run
                      (x),                         # model input (or a tuple for multiple inputs)
                      onnx_path,   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=args.opset_version,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['post_input'],   # the model's input names
                      output_names = ['post_output'] # the model's output names
                      # output_names = ['locs', 'max_val_x', 'max_val_y'] # the model's output names
                                    )
                             
    #simplify ONNX...                          
    onnx_model = onnx.load(onnx_path)  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx model
    onnx_model, check = onnxsim.simplify(onnx_model)
    onnx.save(onnx_model, onnx_path)