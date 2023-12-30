from scripts.layer_divider_modules.installation import *
install_sam()
from scripts.layer_divider_modules import sam
from scripts.layer_divider_modules.ui_utils import *
from scripts.layer_divider_modules.html_constants import *
from scripts.layer_divider_modules.model_downloader import DEFAULT_MODEL_TYPE
from fastapi import FastAPI, Body, Request, Response
from fastapi.responses import FileResponse
from PIL import Image
import cv2

import io
import base64
import numpy as np

import gradio as gr
import os

from modules import scripts, script_callbacks

sam_inf = sam.SamInference()


def add_tab():
    with gr.Blocks(css=CSS) as tab:
        with gr.Row().style(equal_height=True):  # bug https://github.com/gradio-app/gradio/issues/3202
            with gr.Column(scale=5):
                img_input = gr.Image(label="Input image here")
            with gr.Column(scale=5):
                # Tunable Params
                dd_models = gr.Dropdown(label="Model", value=DEFAULT_MODEL_TYPE, choices=sam_inf.available_models)
                nb_points_per_side = gr.Number(label="points_per_side", value=32)
                sld_pred_iou_thresh = gr.Slider(label="pred_iou_thresh", value=0.88, minimum=0, maximum=1)
                sld_stability_score_thresh = gr.Slider(label="stability_score_thresh", value=0.95, minimum=0,
                                                       maximum=1)
                nb_crop_n_layers = gr.Number(label="crop_n_layers", value=0)
                nb_crop_n_points_downscale_factor = gr.Number(label="crop_n_points_downscale_factor", value=1)
                nb_min_mask_region_area = gr.Number(label="min_mask_region_area", value=0)
                html_param_explain = gr.HTML(PARAMS_EXPLANATION, elem_id="html_param_explain")

        with gr.Row():
            btn_generate = gr.Button("GENERATE111", variant="primary")
        with gr.Row():
            gallery_output = gr.Gallery(label="Output images will be shown here").style(grid=5, height="auto")
            with gr.Column():
                output_file = gr.outputs.File(label="Generated psd file")
                btn_open_folder = gr.Button("📁\nOpen PSD folder")
        params = [nb_points_per_side, sld_pred_iou_thresh, sld_stability_score_thresh, nb_crop_n_layers,
                  nb_crop_n_points_downscale_factor, nb_min_mask_region_area]
        btn_generate.click(fn=sam_inf.generate_mask_app,
                           inputs=[img_input, dd_models] + params, outputs=[gallery_output, output_file])
        btn_open_folder.click(fn=lambda: open_folder(os.path.join(base_dir, "layer_divider_outputs", "psd")),
                              inputs=None, outputs=None)

        return [(tab, "Layer Divider", "layer_divider")]


def on_unload():
    global sam_inf
    sam_inf = None

def on_app_started(_: gr.Blocks, app: FastAPI):
    @app.post("/layer_divider/divide")
    async def _divide(request: Request):
        data = await request.json()

        base64_decoded = base64.b64decode(data['input_image'])
        image = Image.open(io.BytesIO(base64_decoded))
        image_np = np.array(image)
       
        result = sam_inf.generate_mask_app(image_np, "vit_h", 32, 0.88, 0.95, 0, 1, 0)

        imageList = []
        index = 0
        for item in result[0]:
            im = Image.fromarray(item[0].astype("uint8"))
            rawBytes = io.BytesIO()
            im.save(rawBytes, "PNG")
            rawBytes.seek(0)
            imString = base64.b64encode(rawBytes.read())
            imageList.append(imString)
            print(f'{index} of {len(result[0])}')
            index = index + 1
            
        return {
            'result': {
                'psd': FileResponse(path=result[1], filename='layer.psd', media_type='image/psd'),
                'imageList': imageList
            },
        }

script_callbacks.on_ui_tabs(add_tab)
script_callbacks.on_app_started(on_app_started)
script_callbacks.on_script_unloaded(on_unload)
