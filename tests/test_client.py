from epic_stream_processor import client
from epic_stream_processor.epic_services.server import epic_postprocessor
from epic_stream_processor.epic_services.service_hub import ServiceHub
from sqlalchemy import create_engine
from pytest_pgsql import PostgreSQLTestDB 
from epic_stream_processor.epic_grpc.epic_image_pb2_grpc import (
    epic_post_processStub,
)
from typing import Type


def test_send_dummy_data(
    grpc_stub: Type[epic_post_processStub],
    postgresql_db: Type[PostgreSQLTestDB],    
    grpc_servicer: epic_postprocessor,
) -> None:
    service_hub = ServiceHub(postgresql_db.engine)
    grpc_servicer.set_storage_servicer(service_hub)
    # service_hub._connect_pgdb()
    service_hub._pgdb.create_all_tables()
    rpc_client = client.EpicRPCClient()
    _ = grpc_stub.filter_and_save_chunk(next(rpc_client.get_dummy_data(1)))


# coverage run --parallel -m pytest --grpc-fake-server --pg-extensions=postgis
