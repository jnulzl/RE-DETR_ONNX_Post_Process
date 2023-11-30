import argparse
import os
import sys
import cv2
import onnxruntime  
import numpy as np  

def prepare_image_input(im_data, size=640): 
    im_data = cv2.resize(im_data, (size, size))           
    im_data = cv2.cvtColor(im_data, cv2.COLOR_BGR2RGB)
    im_data = im_data.astype(np.float32) / 255.0    
    input_array = im_data.transpose(2, 0, 1).astype(np.float32)

    return np.array([input_array])

    
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_path', required=True, type=str, help='Path of onnx model.')
    parser.add_argument('--inp_size', required=True, type=int, default=640, help='Path of onnx model.')
    parser.add_argument('--img_path', required=True, type=str, help='Path of test image.')
    parser.add_argument('--det_thresh', required=False, type=float, default=0.7, help='Det threshold.')
       
    return parser.parse_args()

        
if __name__ == '__main__':
    args = parse_arguments()

    # 加载 ONNX 模型      
    runtime = onnxruntime.InferenceSession(args.onnx_path)  

    # 创建输入数据      
    im_data = cv2.imread(args.img_path)
    img_height, img_width = im_data.shape[:2]
    input_data = prepare_image_input(im_data, args.inp_size)
    im_shape = np.array([img_height, img_width]).reshape(1, 2).astype(np.float32)    
    outputs = runtime.run(None, {'image': input_data})
    boxes = outputs[0][0] # 0th output, 0th batch
    det_thresh = args.det_thresh
    for index, box in enumerate(boxes):
        class_id, score, x1, y1, x2, y2 = box    
        if score < det_thresh:
            continue
        print(index, class_id, score)
        x1 = int(x1 * img_width)
        y1 = int(y1 * img_height)
        x2 = int(x2 * img_width)
        y2 = int(y2 * img_height)
        cv2.rectangle(im_data, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite("output.jpg", im_data)
