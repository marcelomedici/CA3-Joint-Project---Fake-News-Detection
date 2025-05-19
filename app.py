from shiny import App, ui, reactive, render
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

endpoint_name = "huggingface-pytorch-inference-2025-05-19-00-56-14-013"

predictor = Predictor(
    endpoint_name=endpoint_name,
    serializer=JSONSerializer(),
    deserializer=JSONDeserializer()
)

app_ui = ui.page_fluid(
    ui.h2("Statement Check"),
    ui.input_text("input_text", "type or paste your text:", ""),
    ui.output_text_verbatim("prediction")
)

def server(input, output, session):
    @output()
    @render.text
    def prediction():
        textt = input.input_text()
        if textt:
            try:
                response = predictor.predict({"inputs": textt})
                return f"output: {response}"
            except Exception as e:
                return f"Error: {e}"
        else:
            return "type or paste a text for verification"

app = App(app_ui, server)
