import pickle
import pandas as pd
import gradio as gr

# GPU placeholder: needed on Hugging Face ZeroGPU, ignored everywhere else
try:
    import spaces
    gpu = spaces.GPU
except ImportError:
    gpu = lambda f: f

with open("insurance_model.pkl", "rb") as f:
    model = pickle.load(f)

@gpu
def _gpu_placeholder():
    return None

RMSE = 4732

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.teal,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Poppins"), "sans-serif"],
).set(
    body_background_fill="#eef2f6",
    block_background_fill="transparent",
    block_border_width="0px",
    block_shadow="none",
    block_label_background_fill="transparent",
    input_background_fill="#f1f5f9",
    input_border_width="1px",
    input_border_color="#e2e8f0",
    input_radius="12px",
    button_large_radius="14px",
    button_primary_background_fill="#0f766e",
    button_primary_background_fill_hover="#115e59",
    button_primary_text_color="#ffffff",
    checkbox_label_background_fill="#f1f5f9",
    checkbox_label_background_fill_selected="#0f766e",
    checkbox_label_text_color_selected="#ffffff",
    checkbox_label_border_color_selected="#0f766e",
    slider_color="#0f766e",
)

# Runs when the page loads: always start in light mode
START_LIGHT_JS = """() => {
    document.body.classList.remove('dark');
    document.body.classList.remove('night');
}"""

# Runs when the theme button is clicked
TOGGLE_JS = """() => {
    const on = document.body.classList.toggle('night');
    document.body.classList.toggle('dark', on);
    const b = document.querySelector('#theme-btn');
    if (b) { b.textContent = on ? 'Light mode' : 'Dark mode'; }
}"""

CSS = """
<style>
/* ================= LIGHT MODE (default) ================= */
body, .gradio-container {
    background-color: #eef2f6 !important;
    background-image: radial-gradient(#cbd5e1 1px, transparent 1px) !important;
    background-size: 22px 22px !important;
}
.gradio-container {
    max-width: 70% !important;
    margin: 0 auto !important;
    padding: 24px 0 40px 0 !important;
}
@media (max-width: 900px) {
    .gradio-container { max-width: 96% !important; }
}
footer { display: none !important; }

.gradio-container .hero {
    background: #0b2545;
    border-radius: 20px;
    padding: 30px 34px;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    gap: 22px;
    flex-wrap: wrap;
    box-shadow: 0 10px 30px rgba(11, 37, 69, 0.25);
    width: 100%;
    box-sizing: border-box;
}
.hero-icon {
    width: 62px; height: 62px;
    border-radius: 16px;
    background: #14b8a6;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.gradio-container .hero h1 { margin: 0; font-size: 1.9rem; font-weight: 700; color: #ffffff !important; letter-spacing: -0.5px; }
.gradio-container .hero p  { margin: 4px 0 12px 0; color: #cbd5e1 !important; font-size: 0.98rem; }
.gradio-container .hero .chip {
    display: inline-block;
    margin: 2px 6px 2px 0;
    padding: 4px 12px;
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.14) !important;
    color: #7ee8da !important;
    font-size: 0.8rem;
    font-weight: 600;
}

.gradio-container .card {
    background: #ffffff !important;
    border-radius: 20px !important;
    border: none !important;
    box-shadow: 0 8px 30px rgba(15, 23, 42, 0.10) !important;
    padding: 26px !important;
    width: 100% !important;
    box-sizing: border-box !important;
}
.gradio-container .card > *,
.gradio-container .card .styler,
.gradio-container .card .row,
.gradio-container .card .column,
.gradio-container .card .form,
.gradio-container .card .block {
    background: #ffffff !important;
    border: none !important;
    box-shadow: none !important;
}
.gradio-container .card input,
.gradio-container .card textarea { background: #f1f5f9 !important; color: #0f172a !important; }
.gradio-container .card .block-title,
.gradio-container .card [data-testid="block-info"] { color: #0f172a !important; font-weight: 600 !important; }
.gradio-container .card .wrap label.selected,
.gradio-container .card .wrap label.selected span { background: #0f766e !important; color: #ffffff !important; }

#predict-btn {
    background: #0f766e !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 14px !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    padding: 14px 0 !important;
    box-shadow: 0 6px 16px rgba(15, 118, 110, 0.35) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}
#predict-btn:hover { transform: translateY(-2px); box-shadow: 0 10px 22px rgba(15, 118, 110, 0.45) !important; }
#reset-btn {
    background: #f1f5f9 !important;
    color: #334155 !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 14px !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    padding: 14px 0 !important;
}
#reset-btn:hover { background: #e2e8f0 !important; }

/* Theme toggle: fixed in the top-right corner, always visible */
#theme-btn {
    position: fixed !important;
    top: 16px !important;
    right: 20px !important;
    z-index: 9999 !important;
    width: auto !important;
    min-width: 0 !important;
    padding: 10px 22px !important;
    border-radius: 999px !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    background: #ffffff !important;
    color: #0b2545 !important;
    border: 2px solid #0f766e !important;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.2) !important;
    cursor: pointer;
}
#theme-btn:hover { background: #0f766e !important; color: #ffffff !important; }

.gradio-container .result {
    background: #0f766e;
    border-radius: 18px;
    padding: 34px 26px;
    text-align: center;
    min-height: 330px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.gradio-container .result, .gradio-container .result * { color: #ffffff !important; }
.result .label { font-size: 0.85rem; letter-spacing: 1.5px; text-transform: uppercase; opacity: 0.85; font-weight: 600; }
.result .value { font-size: 3.1rem; font-weight: 800; margin: 12px 0 6px 0; line-height: 1.1; }
.result .sub { font-size: 1.05rem; opacity: 0.95; }
.result .divider { height: 1px; background: rgba(255, 255, 255, 0.3); margin: 20px 30px; }
.result .error { font-size: 0.9rem; opacity: 0.9; }
.gradio-container .result.idle { background: #e2e8f0; }
.gradio-container .result.idle, .gradio-container .result.idle * { color: #475569 !important; }
.gradio-container .result.idle .value { color: #94a3b8 !important; }

.gradio-container .note { text-align: center; color: #64748b !important; font-size: 0.82rem; margin-top: 16px; }

/* ================= DARK MODE ================= */
body.night, body.night .gradio-container {
    background-color: #0b1220 !important;
    background-image: radial-gradient(#1e293b 1px, transparent 1px) !important;
}
body.night .gradio-container .hero { background: #12203a; border: 1px solid #1e3a5f; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5); }

body.night .gradio-container .card { background: #111c2e !important; box-shadow: 0 8px 30px rgba(0, 0, 0, 0.55) !important; }
body.night .gradio-container .card > *,
body.night .gradio-container .card .styler,
body.night .gradio-container .card .row,
body.night .gradio-container .card .column,
body.night .gradio-container .card .form,
body.night .gradio-container .card .block { background: #111c2e !important; }
body.night .gradio-container .card input,
body.night .gradio-container .card textarea { background: #1e293b !important; color: #f1f5f9 !important; border-color: #334155 !important; }
body.night .gradio-container .card .block-title,
body.night .gradio-container .card [data-testid="block-info"] { color: #f1f5f9 !important; }
body.night .gradio-container .card .wrap label { background: #1e293b !important; color: #e2e8f0 !important; border-color: #334155 !important; }
body.night .gradio-container .card .wrap label span { color: #e2e8f0 !important; }
body.night .gradio-container .card .wrap label.selected,
body.night .gradio-container .card .wrap label.selected span { background: #0f766e !important; color: #ffffff !important; }

body.night #reset-btn { background: #1e293b !important; color: #e2e8f0 !important; border: 1px solid #334155 !important; }
body.night #reset-btn:hover { background: #334155 !important; }
body.night #theme-btn { background: #1e293b !important; color: #fde68a !important; border: 2px solid #fde68a !important; }
body.night #theme-btn:hover { background: #fde68a !important; color: #0b1220 !important; }

body.night .gradio-container .result.idle { background: #1e293b; }
body.night .gradio-container .result.idle, body.night .gradio-container .result.idle * { color: #94a3b8 !important; }
body.night .gradio-container .result.idle .value { color: #64748b !important; }
body.night .gradio-container .note { color: #94a3b8 !important; }
</style>
"""

HEADER = """
<div class="hero">
  <div class="hero-icon">
    <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M3 12h4l2-5 4 10 2-5h6"/>
    </svg>
  </div>
  <div class="hero-text">
    <h1>Medical Insurance Cost Predictor</h1>
    <p>Estimate a person's yearly medical insurance charge from six simple details.</p>
    <span class="chip">CatBoost</span>
    <span class="chip">10-Fold Cross Validation</span>
    <span class="chip">Mean RMSE about $4,732</span>
  </div>
</div>
"""

IDLE = """
<div class="result idle">
  <div class="label">Estimated yearly charge</div>
  <div class="value">$ - - - -</div>
  <div class="sub">Fill in the details and press Predict</div>
</div>
"""

def predict_charges(age, sex, bmi, children, smoker, region):
    row = pd.DataFrame([{
        "age": int(age), "bmi": float(bmi), "children": int(children),
        "sex": sex, "smoker": smoker, "region": region,
    }])
    prediction = float(model.predict(row)[0])
    return (
        '<div class="result">'
        '<div class="label">Estimated yearly charge</div>'
        f'<div class="value">${prediction:,.2f}</div>'
        f'<div class="sub">about ${prediction / 12:,.2f} per month</div>'
        '<div class="divider"></div>'
        f'<div class="error">Typical model error: about &plusmn; ${RMSE:,}</div>'
        '</div>'
    )

def reset():
    return 35, "male", 28.0, 0, "no", "northeast", IDLE

try:
    blocks = gr.Blocks(title="Medical Insurance Cost Predictor", theme=THEME, js=START_LIGHT_JS)
except TypeError:
    blocks = gr.Blocks(title="Medical Insurance Cost Predictor")

with blocks as demo:
    gr.HTML(CSS)
    theme_btn = gr.Button("Dark mode", elem_id="theme-btn")
    gr.HTML(HEADER)

    with gr.Group(elem_classes="card"):
        with gr.Row(equal_height=True):
            with gr.Column(scale=5):
                age = gr.Slider(18, 64, value=35, step=1, label="Age")
                bmi = gr.Slider(15, 55, value=28.0, step=0.1, label="BMI")
                children = gr.Slider(0, 5, value=0, step=1, label="Number of children")
                sex = gr.Radio(["male", "female"], value="male", label="Sex")
                smoker = gr.Radio(["no", "yes"], value="no", label="Smoker")
                region = gr.Dropdown(
                    ["northeast", "northwest", "southeast", "southwest"],
                    value="northeast", label="Region",
                )
                with gr.Row():
                    predict_btn = gr.Button("Predict Cost", elem_id="predict-btn", variant="primary")
                    reset_btn = gr.Button("Reset", elem_id="reset-btn")
            with gr.Column(scale=4):
                result = gr.HTML(IDLE)

    gr.HTML('<div class="note">Estimates are for educational purposes only and are not real insurance quotes.</div>')

    inputs = [age, sex, bmi, children, smoker, region]
    predict_btn.click(predict_charges, inputs=inputs, outputs=result)
    reset_btn.click(reset, inputs=None, outputs=inputs + [result])
    theme_btn.click(None, None, None, js=TOGGLE_JS)

if __name__ == "__main__":
    demo.launch()
