from flask import Flask, request, jsonify
from txtpreprocessing.text_processing import TextProcessing
from inference.predict import PredictSentiment

TP = TextProcessing()
predict_model = PredictSentiment()
app = Flask(__name__)

def mapping_result(result_prediction):
    if result_prediction == 0:
        return "negative"
    elif  result_prediction == 1:
        return "positive"
    else:
        return "neutral"

#Homepage
@app.route('/', methods=['GET'])
def get():
    return "Welcome"

@app.route("/predict_sentiment/v1", methods=["POST"])
def predict_sentiment_ann():
    text = request.get_json()['text']
    clean_text,bow = TP.get_bow(text)
    result_prediction = predict_model.predict_ann(bow)
    result_prediction = mapping_result(result_prediction)
    return jsonify({"text":clean_text, "result_sentiment":result_prediction})

@app.route("/predict_sentiment_lstm/v1", methods=["POST"])
def predict_sentiment_lstm():
    text = request.get_json()['text']
    clean_text,input_ids = TP.get_tokenizer(text)
    result_prediction = predict_model.predict_lstm(input_ids)
    result_prediction = mapping_result(result_prediction)
    return jsonify({"text":clean_text, "result_sentiment":result_prediction})
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5555, debug=True)
