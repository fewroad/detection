# from fastapi import FastAPI, Query
# from fastapi.responses import JSONResponse
# import os
# import subprocess

# app = FastAPI()

# @app.post("/detect/")
# def detect_video(video_path: str = Query(..., description="待处理视频的完整路径")):
#     if not os.path.exists(video_path):
#         return JSONResponse(status_code=400, content={"error": "视频路径不存在"})

#     success, message = run_deepfake_detection(video_path)
#     if not success:
#         return JSONResponse(status_code=500, content={"error": message})                

    
#     return JSONResponse(status_code=200, content={"message": "模型处理完成",
#                                                   "deepfake_detection_result": message,
#                                                   "confidence": pred})

# def run_deepfake_detection(input_path):
#     command = [
#         "python3", "predict.py",
#         "--video_path", input_path,                 
#         "--model_weights", "./MINTIME_XC_Model_checkpoint30",
#         "--extractor_weights", "./MINTIME_XC_Extractor_checkpoint30",
#         "--config", "config/size_invariant_timesformer.yaml"
#     ]
 
#     try:
#         subprocess.run(command, check=True)                         
#         return True, "success"
#     except subprocess.CalledProcessError as e:
#         return False, f"模型处理失败: {e}"
    
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import os
import subprocess
import json

app = FastAPI()

@app.post("/detect/")
async def detect_video(video_path: str = Query(..., description="待处理视频的完整路径")):
    if not os.path.exists(video_path):
        return JSONResponse(status_code=400, content={"error": "视频路径不存在"})

    # 运行检测并捕获输出
    success, result = run_deepfake_detection(video_path)
    
    if not success:
        return JSONResponse(status_code=500, content={"error": result})                

    # 解析预测结果
    
    try:
        face=result["no faces detected"]
        if face==True:
            return JSONResponse(status_code=200, content={"message": "未检测到人脸",
                                                        "confidence": 0,
                                                        "faces_detected": 0,
                                                        "details": {
                                                            "raw_prediction": 0,
                                                            "frames_distribution": {}
                                                        }})
    except:
        pass
    try:
        pred = result["prediction"]
        confidence = pred if pred > 0.5 else 1 - pred
        status = "fake" if pred > 0.5 else "real"
        
        return JSONResponse(
            status_code=200,
            content={
                "status": status,
                "confidence": float(confidence),
                "faces_detected": result["identities"],
                "details": {
                    "raw_prediction": float(pred),
                    "frames_distribution": result["frames_per_identity"]
                }
            }
            
            # content={
            #     "status": "fake",
            #     "confidence": 98.89,
            #     "faces_detected": result["identities"],
            #     "details": {
            #         "raw_prediction": float(pred),
            #         "frames_distribution": result["frames_per_identity"]
            #     }
            # }
        )
    except KeyError as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"结果解析失败: {str(e)}"}
        )

def run_deepfake_detection(input_path):
    command = [
        "python3", "predict.py",
        "--video_path", input_path,                 
        "--model_weights", "./MINTIME_XC_Model_checkpoint30",
        "--extractor_weights", "./MINTIME_XC_Extractor_checkpoint30",
        "--config", "config/size_invariant_timesformer.yaml"
    ]
 
    try:
        # 捕获子进程输出
        process = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # 解析JSON输出
        output_lines = process.stdout.split('\n')
        json_output = None
        
        # 查找最后一行有效的JSON输出
        for line in reversed(output_lines):
            try:
                json_output = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
                
        if json_output is None:
            return False, "未能解析模型输出"
            
        return True, json_output
        
    except subprocess.CalledProcessError as e:
        error_msg = f"模型处理失败: {e.stderr}" if e.stderr else str(e)
        return False, error_msg
    except Exception as e:
        return False, f"未知错误: {str(e)}"
    #uvicorn wen:app --host 0.0.0.0 --port 8000 --reload
