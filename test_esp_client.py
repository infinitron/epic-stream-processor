import grpc
from epic_stream_processor import epic_image_pb2_grpc
from epic_stream_processor import epic_image_pb2
import json
import pandas as pd
with grpc.insecure_channel('localhost:2020') as channel:
    stub = epic_image_pb2_grpc.epic_post_processStub(channel)
    response = stub.fetch_watchlist(epic_image_pb2.empty())
    df_json = json.loads(response.pd_json)
    print(pd.read_json(df_json))