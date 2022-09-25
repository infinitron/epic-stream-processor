# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from . import epic_image_pb2 as epic__image__pb2


# flake8: noqa


class epic_post_processStub:
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.filter_and_save = channel.unary_unary(
            "/epic_post_process/filter_and_save",
            request_serializer=epic__image__pb2.epic_image.SerializeToString,
            response_deserializer=epic__image__pb2.empty.FromString,
        )
        self.filter_and_save_chunk = channel.stream_unary(
            "/epic_post_process/filter_and_save_chunk",
            request_serializer=epic__image__pb2.epic_image.SerializeToString,
            response_deserializer=epic__image__pb2.empty.FromString,
        )


class epic_post_processServicer:
    """Missing associated documentation comment in .proto file."""

    def filter_and_save(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def filter_and_save_chunk(self, request_iterator, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")


def add_epic_post_processServicer_to_server(servicer, server):
    rpc_method_handlers = {
        "filter_and_save": grpc.unary_unary_rpc_method_handler(
            servicer.filter_and_save,
            request_deserializer=epic__image__pb2.epic_image.FromString,
            response_serializer=epic__image__pb2.empty.SerializeToString,
        ),
        "filter_and_save_chunk": grpc.stream_unary_rpc_method_handler(
            servicer.filter_and_save_chunk,
            request_deserializer=epic__image__pb2.epic_image.FromString,
            response_serializer=epic__image__pb2.empty.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        "epic_post_process", rpc_method_handlers
    )
    server.add_generic_rpc_handlers((generic_handler,))


# This class is part of an EXPERIMENTAL API.
class epic_post_process:
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def filter_and_save(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/epic_post_process/filter_and_save",
            epic__image__pb2.epic_image.SerializeToString,
            epic__image__pb2.empty.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def filter_and_save_chunk(
        request_iterator,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.stream_unary(
            request_iterator,
            target,
            "/epic_post_process/filter_and_save_chunk",
            epic__image__pb2.epic_image.SerializeToString,
            epic__image__pb2.empty.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )
