# from typing import Callable
# from typing import Type

# import pytest

# from epic_stream_processor import epic_image_pb2_grpc
# from epic_stream_processor.epic_grpc.epic_image_pb2_grpc import epic_post_processStub


# @pytest.fixture(scope="module")
# def grpc_add_to_server() -> Callable[..., None]:
#     return epic_image_pb2_grpc.add_epic_post_processServicer_to_server


# @pytest.fixture(scope="module")
# def grpc_servicer() -> epic_postprocessor:

#     return epic_postprocessor()


# @pytest.fixture(scope="module")
# def grpc_stub_cls() -> Type[epic_post_processStub]:

#     return epic_post_processStub
