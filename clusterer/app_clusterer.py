import os
import numpy as np
from random import shuffle
from itertools import groupby
from operator import itemgetter
from flask import Flask, jsonify, request
from flask_restplus import Api, Resource, fields
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModel
from waitress import serve

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
api = Api(app)

name_space = api.namespace('api', 'На вход поступает JSON, возвращает JSON')
input_data = name_space.model("Insert JSON",
                              {"texts": fields.List(fields.String(description="Insert texts", required=True)),
                               "score": fields.Float(description="Distance", required=True)}, )


def cluster_name_number(vectors: np.array) -> np.array:
    """Function get vectors, finds vector most close to average of vectors and returns it's number."""
    # weight_average_vector = np.average(vectors, axis=0, weights=vectors)
    weight_average_vector = np.average(vectors, axis=0)
    weight_average_vector_ = weight_average_vector.reshape(1, weight_average_vector.shape[0])
    distances_from_average = cosine_similarity(vectors, weight_average_vector_)
    return np.argmax(distances_from_average)


def grouped_func(data: list) -> [{}]:
    """Function groups input list of data with format: [(label, vector, text)]
    into list of dictionaries, each dictionary of type:
    {
    label: label,
    texts: list of texts correspond to label
    vectors_matrix: numpy matrix of vectors correspond to label
    }
    """
    data = sorted(data, key=lambda x: x[0])
    grouped_data = []
    for key, group_items in groupby(data, key=itemgetter(0)):
        d = {"label": key, "texts": []}
        temp_vectors = []
        for item in group_items:
            temp_vectors.append(item[1])
            d["texts"].append(item[2])
        d["vectors_matrix"] = np.vstack(temp_vectors)
        grouped_data.append(d)
    return grouped_data


def clustering_func(tokenizer, model, clusterer: AgglomerativeClustering, texts: []) -> {}:
    """Function for text collection clustering"""
    encoded_input = tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    embeddings = model_output.pooler_output
    vectors = torch.nn.functional.normalize(embeddings)
    clusters = clusterer.fit(vectors)
    data = [(lb, v, tx) for lb, v, tx in zip(clusters.labels_, vectors, texts)]
    grouped_data = grouped_func(data)
    result_list = []
    for d in grouped_data:
        label = str(d["label"])
        title_number = cluster_name_number(d["vectors_matrix"])
        title = d["texts"][title_number]
        cluster_size = len(d["texts"])
        result_list += [(label, title, tx, cluster_size) for tx in d["texts"]]
    return {"texts_with_labels": result_list}


@name_space.route('/clusterer')
class Clustering(Resource):
    @name_space.expect(input_data)
    def post(self):
        """POST method on input csv file with texts and score, output clustering texts as JSON file."""
        json_data = request.json
        texts_list = [str(tx) for tx in json_data["texts"]]

        """restricting number of texts fragments (resource limit)"""
        shuffle(texts_list)
        clustering_texts = texts_list[:30000]
        clusterer = AgglomerativeClustering(n_clusters=None, distance_threshold=json_data['score'],
                                            memory=os.path.join("cache"))

        tokenizer = AutoTokenizer.from_pretrained("cointegrated/LaBSE-en-ru")
        model = AutoModel.from_pretrained("cointegrated/LaBSE-en-ru")

        return jsonify(clustering_func(tokenizer, model, clusterer, clustering_texts))


if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=6007)
    # app.run(host='0.0.0.0', port=8080)
