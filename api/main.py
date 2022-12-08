from fastapi import FastAPI, File, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.concurrency import run_in_threadpool
from inference_onnx import Converter
import aiofiles
import os

MODEL = "rvm_resnet50_fp32.onnx"
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:3000'],
    allow_methods=['*'],
    allow_headers=['*']
)

@app.get("/")
async def say_hello():
    return { "message": "hello" }

@app.post('/video')
async def get_composition(video: UploadFile = File(...), image: UploadFile = File(default=None)):
    try:
        async with aiofiles.tempfile.NamedTemporaryFile("wb", delete=False) as tempVideo:
            try:
                video_file = await video.read()
                await tempVideo.write(video_file)
                if image:
                    print("HELLO")
                    async with aiofiles.tempfile.NamedTemporaryFile("wb", delete=False) as tempImage:
                        try:
                            image_file = await image.read()
                            await tempImage.write(image_file)
                        except Exception as e:
                            print(e)
                            return {"message": "There was an error uploading the file"}
                        finally:
                            await image.close()
            except Exception as e:
                print(e)
                return {"message": "There was an error uploading the file"}
            finally:
                await video.close()
        if image:
            await run_in_threadpool(process_video, tempVideo.name, tempImage.name)  # Pass temp.name to VideoCapture()
        else:
            await run_in_threadpool(process_video, tempVideo.name)  # Pass temp.name to VideoCapture()
    except Exception as e:
        return {"message": "There was an error processing the file"}
    finally:
        os.remove(tempVideo.name)

    if image:
        os.remove(tempImage.name)
        return FileResponse("composed.mp4")
    else:
        return FileResponse("composition.mp4")

# @app.websocket("/ws")
# async def ws_endpoint(websocket: WebSocket):
#     await websocket.send_text()

def process_video(video, image=None, ws=None):
    converter = Converter(MODEL, "cuda", "fp32")
    return converter.convert(
        input_source=video,
        input_resize=None,
        downsample_ratio=0.25,
        output_type="video",
        output_composition="composition.mp4",
        output_alpha="alpha.mp4",
        output_foreground="foreground.mp4",
        output_video_mbps=4,
        seq_chunk=1,
        num_workers=0,
        progress=True,
        background_image=image,
    )