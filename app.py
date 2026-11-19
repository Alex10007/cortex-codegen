"""Cortex Codegen UI. Talks to the local Cortex engines only — no external
model APIs. If the demo model has not been trained yet on this host, it says
so honestly instead of proxying a third-party model."""
from __future__ import annotations

from pathlib import Path

import gradio as gr

from cortex.core import CortexAgent

_ROOT = Path(__file__).parent
agent = CortexAgent(
    state_dir=str(_ROOT / "artifacts" / "cortex-state"),
    tokenizer_path=str(_ROOT / "artifacts" / "tokenizer" / "tokenizer.json"),
    checkpoint_path=str(_ROOT / "artifacts" / "runs" / "cortex-demo" / "final.pt"),
)


def respond(message, history):
    statuses = {r["engine"]: r["status"] for r in agent.service_reports()}
    model_status = statuses.get("ModelCore", "")
    if "operational" not in model_status:
        return ("Cortex Codegen core model is not trained on this host yet; "
                "run `python scripts/build_and_verify.py`. Engines: "
                + ", ".join(f"{k}={v}" for k, v in sorted(statuses.items())))
    try:
        return agent.generate(message, max_new_tokens=96)
    except Exception as exc:
        return f"generation unavailable: {exc}"


def status_text() -> str:
    return "\n".join(
        f"{r['engine']}: {r['status']}\n  evidence: {r['evidence']}"
        for r in agent.service_reports()
    )


with gr.Blocks(title="Cortex Codegen") as demo:
    gr.Markdown("# Cortex Codegen — Fibonera AI software-engineering system")
    with gr.Tab("Chat"):
        gr.ChatInterface(respond, type="messages")
    with gr.Tab("System status"):
        out = gr.Textbox(status_text(), label="Engine verification state", lines=24)
        gr.Button("Refresh").click(status_text, outputs=out)

if __name__ == "__main__":
    demo.launch()
