import pickle
import pandas as pd
import gradio as gr
import spaces

with open("insurance_model.pkl", "rb") as f:
    model = pickle.load(f)

@spaces.GPU
def _gpu_placeholder():
    return None

def predict_charges(age, sex, bmi, children, smoker, region):
    row = pd.DataFrame([{
        "age": int(age), "bmi": float(bmi), "children": int(children),
        "sex": sex, "smoker": smoker, "region": region,
    }])
    prediction = model.predict(row)[0]
    return f"${prediction:,.2f} per year"

demo = gr.Interface(
    fn=predict_charges,
    inputs=[
        gr.Slider(18, 64, value=35, step=1, label="Age"),
        gr.Radio(["male", "female"], value="male", label="Sex"),
        gr.Slider(15, 55, value=28, step=0.1, label="BMI"),
        gr.Slider(0, 5, value=0, step=1, label="Number of children"),
        gr.Radio(["no", "yes"], value="no", label="Smoker"),
        gr.Dropdown(["northeast", "northwest", "southeast", "southwest"],
                    value="northeast", label="Region"),
    ],
    outputs=gr.Textbox(label="Predicted yearly insurance charge"),
    title="Medical Insurance Cost Predictor",
    description="Enter a person's details to predict their yearly medical insurance charge (CatBoost model, 10-fold CV mean RMSE of about $4,732).",
)

if __name__ == "__main__":
    demo.launch()
