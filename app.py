import gradio as gr

from src.system import CarRecognizer

recognizer = CarRecognizer()
cfg = recognizer.cfg


def recognize(image):
    if image is None:
        return "Upload a photo first.", {}
    out = recognizer.predict(image)
    top = dict(out["top"])
    if out["decision"] == "not a car":
        return "This doesn't look like a car.", {}
    best = out["top"][0][0]
    if out["decision"] == "auto-fill":
        text = f"**{best}** (filled in automatically)"
    else:
        text = "Not sure. Please pick your car:\n\n" + "\n".join(f"- {n}" for n, _ in out["top"][:3])
    return text, top


demo = gr.Interface(
    fn=recognize,
    inputs=gr.Image(type="pil", label="Photo of the car"),
    outputs=[gr.Markdown(label="Listing"), gr.Label(num_top_classes=5, label="Top 5 models")],
    title="Which car is this? ResNet-18 + CLIP",
    description=(f"Recognises 196 car models (make, model, year; 1991-2012). Fills the listing in automatically "
                 f"when confidence is at least {cfg['conf_threshold']}, otherwise asks the seller to pick from the top 3."),
)

if __name__ == "__main__":
    demo.launch()
