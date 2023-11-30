import sys
import argparse
import onnx
import copy
import onnxsim

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_path_1', required=True, help='Path of input1 model.')
    parser.add_argument('--onnx_path_1_output', required=True, help='Output of input1 model.')
    parser.add_argument('--onnx_path_2', required=True, help='Path of input2 model.')
    parser.add_argument('--onnx_path_2_input', required=True, help='Input of input2 model.')    
    parser.add_argument('--save_file', required=False, help='Path to save the new onnx model.')
    return parser.parse_args()

        
if __name__ == '__main__':
    args = parse_arguments()

    onnx_path_1 = args.onnx_path_1
    onnx_path_2 = args.onnx_path_2

    onnx_model_1 = onnx.load(onnx_path_1)
    onnx_model_2 = onnx.load(onnx_path_2)
    io_map = [(args.onnx_path_1_output, args.onnx_path_2_input)]

    onnx_merge = onnx.compose.merge_models(onnx_model_1, onnx_model_2, io_map, 
                                        doc_string="jnulzl",
                                        producer_name="jnulzl",
                                        # prefix1="rtm_", 
                                        # prefix2="head_", 
                                        )

    onnx.checker.check_model(onnx_merge)  # check onnx model
    onnx_merge, check = onnxsim.simplify(onnx_merge)
    if args.save_file is None:    
        onnx.save(onnx_merge, onnx_path_1.replace('.onnx', '_merge.onnx'))
    else:
        if args.save_file.endswith(".onnx"):
            onnx.save(onnx_merge, args.save_file)
        else:
            onnx.save(onnx_merge, args.save_file + ".onnx")