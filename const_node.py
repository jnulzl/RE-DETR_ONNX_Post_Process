import argparse
import onnx
import onnx.helper as helper
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', required=True, type=int, help='Model batch size')
    parser.add_argument('--ir_version', required=False, type=int, default=8, help='The ir version of onnx')
    parser.add_argument('--opset_version', required=False, type=int, default=16, help='The opset version of onnx')

    return parser.parse_args()

args = parse_arguments()    
# Create constant tensor
vals = []
for _ in range(args.batch_size):
    # vals.append([1080, 1920]) 
    vals.append(1.0) # 需要在后处理中将y1y2乘以原图像的高(height)
    vals.append(1.0) # 需要在后处理中将x1x2乘以原图像的宽(width)  
const_tensor = helper.make_tensor(name='img_shape', data_type=onnx.TensorProto.FLOAT,                                  
                                  dims=(args.batch_size, 2), vals=vals) 

# Create constant node
const_node = helper.make_node('Constant', [], ['output'], value=const_tensor)

# Create graph and add node
graph = helper.make_graph([const_node], 'constant_img_shape', [], [helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, shape=(args.batch_size, 2))])
model = helper.make_model(graph, producer_name='jnulzl', ir_version=args.ir_version, opset_imports = [helper.make_opsetid("", args.opset_version)])

# Serialize model to save
onnx.save(model, 'constant_img_shape_bs%s.onnx'%(args.batch_size))
