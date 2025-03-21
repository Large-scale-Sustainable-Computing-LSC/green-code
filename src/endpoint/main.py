import json

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import login

from generators import GeneratorBase, Llama32WithAgent, GeneratorResult
from util import logger, get_parser

"""
Adapted version of: https://github.com/LucienShui/huggingface-vscode-endpoint-server

"""

app = FastAPI()
app.add_middleware(
    CORSMiddleware
)
generator: GeneratorBase = ...


@app.post("/api/generate/")
async def api(request: Request):
    json_request: dict = await request.json()
    inputs: str = json_request['inputs']
    parameters: dict = json_request['options']
    logger.info(f'{request.client.host}:{request.client.port} inputs = {json.dumps(inputs)}')

    if json_request['options']['RL_agent_thresh'] is not None:
        if generator.update_threshold(json_request['options']['RL_agent_thresh']):
            logger.info(
                f'{request.client.host}:{request.client.port} RL_agent_thresh = {json_request["options"]["RL_agent_thresh"]}')

    results: GeneratorResult = generator.generate(inputs, parameters)

    generated_text: str = results.generated_text
    logger.info(f'{request.client.host}:{request.client.port} generated_text = {json.dumps(generated_text)}')
    logger.info(f'energy_consumed = {results.energy_consumed}')
    logger.info(f'time_taken = {results.time_taken}')
    logger.info(f'exited_layers = {results.exited_layers}')
    return {
        "generated_text": generated_text.replace(inputs, ""),
        "status": 200
    }


def main():
    global generator
    args = get_parser().parse_args()
    logger.info(f"Using model {args.pretrained}")
    generator = Llama32WithAgent(args.pretrained, device='cuda')
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
