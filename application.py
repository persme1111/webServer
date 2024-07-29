from flask import Flask, request, jsonify, render_template
import folium
import pandas as pd
import plotly.express as px
from jinja2 import Template
import generate_bar_plot_and_map as gbmp

app = Flask(__name__)


@app.route('/')
def home():
    sorted_restaurants_df = pd.read_pickle('datasets/sorted_restaurants_df.pkl')
    map_html = gbmp.plot_restaurants_on_map(sorted_restaurants_df)
    plot_html = gbmp.plot_popular_dishes(sorted_restaurants_df)

    return render_template('index.html', map_html=map_html, plot_html=plot_html)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        content = file.read().decode('utf-8')
        # jsonify({'content': content})
        print('Selected: Successfully Uploaded!')
        return jsonify({'content': 'File successfully uploaded!'})


@app.route('/get_data', methods=['POST'])
def get_data():
    selected_value = request.json.get('selected_value')
    print(selected_value)
    df = gbmp.read_review_data(reviewfile=None, businessfile=None, dish_name=[selected_value])
    map_html = gbmp.plot_restaurants_on_map(df)
    plot_html = gbmp.plot_popular_dishes(df)

    return jsonify({'map_html': map_html, 'plot_html': plot_html})


if __name__ == '__main__':
    app.run(debug=True)
