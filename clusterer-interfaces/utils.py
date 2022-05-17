import io
import requests
import pandas as pd
from flask import Response
from werkzeug.datastructures import Headers
from werkzeug.datastructures import FileStorage


def api_configurator(name_space):
    """"""
    upload_parser = name_space.parser()
    upload_parser.add_argument("file", type=FileStorage, location='files', required=True)
    upload_parser.add_argument("score", type=float, required=True)
    return upload_parser


def remote_clustering(args, clustering_url, upload_type="excel"):
    """"""
    if upload_type == "json":
        json_data = args
    elif upload_type == "excel":
        df = pd.read_excel(args['file'], header=None)
        json_data = {"texts": list(set(df[0])), "score": args['score']}
    else:
        """The function expects csv type of upload data"""
        df = pd.read_csv(args['file'], header=None)
        json_data = {"texts": list(set(df[0])), "score": args['score']}
    clustering_texts_response = requests.post(clustering_url, json=json_data)
    clustering_texts = clustering_texts_response.json()
    return pd.DataFrame(clustering_texts["texts_with_labels"], columns=["label", "cluster_name", "texts", "cluster_size"])


def response_func(clustering_texts_df, response_type="excel"):
    """"""
    headers = Headers()
    if response_type == "excel":
        headers.add('Content-Disposition', 'attachment', filename="clustering_results.xlsx")
        buffer = io.BytesIO()
        clustering_texts_df.to_excel(buffer, index=False, encoding='cp1251'),
        return Response(buffer.getvalue(),
                        mimetype='application/vnd.ms-excel',
                        headers=headers)

    else:
        headers.add('Content-Disposition', 'attachment', filename="clustering_results.csv")
        return Response(clustering_texts_df.to_csv(index=False),
                        mimetype="text/csv",
                        headers=headers)
