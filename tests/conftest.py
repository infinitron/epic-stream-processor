import pytest

from epic_stream_processor import epic_image_pb2
from epic_stream_processor import epic_image_pb2_grpc
from sqlalchemy import create_engine
from epic_stream_processor.epic_orm.pg_pixel_storage import _default_pg_conn_str
from epic_stream_processor.epic_services import epic_postprocessor
from epic_stream_processor.epic_grpc.epic_image_pb2_grpc import (
        epic_post_processStub,
    )
from typing import Callable, Type
from types import ModuleType



@pytest.fixture(scope="module")
def grpc_add_to_server() -> Callable[...,None]:
    return epic_image_pb2_grpc.add_epic_post_processServicer_to_server

@pytest.fixture(scope="module")
def grpc_servicer() -> epic_postprocessor:
    
    return epic_postprocessor()

@pytest.fixture(scope="module")
def grpc_stub_cls() -> Type[epic_post_processStub]:

    return epic_post_processStub


