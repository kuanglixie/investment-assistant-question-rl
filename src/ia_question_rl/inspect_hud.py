import inspect
import modal

image = modal.Image.debian_slim(python_version="3.12").pip_install("hud-python>=0.6.6")
app = modal.App("ia-inspect-hud")

@app.function(image=image)
def inspect_hud() -> None:
    from hud.settings import settings
    print("=== HUD SETTINGS DICT ===", flush=True)
    print(settings.__dict__, flush=True)
    print("=== HUD SETTINGS DIR ===", flush=True)
    print(dir(settings), flush=True)
    
    import hud.graders.llm_judge
    print("=== LLM JUDGE SOURCE ===", flush=True)
    print(inspect.getsource(hud.graders.llm_judge), flush=True)

@app.local_entrypoint()
def main() -> None:
    inspect_hud.remote()
