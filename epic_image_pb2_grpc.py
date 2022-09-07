# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import epic_image_pb2 as epic__image__pb2


class epic_post_processStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.filter_and_save = channel.unary_unary(
        '/epic_post_process/filter_and_save',
        request_serializer=epic__image__pb2.epic_image.SerializeToString,
        response_deserializer=epic__image__pb2.empty.FromString,
        )
    self.filter_and_save_chunk = channel.stream_unary(
        '/epic_post_process/filter_and_save_chunk',
        request_serializer=epic__image__pb2.epic_image.SerializeToString,
        response_deserializer=epic__image__pb2.empty.FromString,
        )


class epic_post_processServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def filter_and_save(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def filter_and_save_chunk(self, request_iterator, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_epic_post_processServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'filter_and_save': grpc.unary_unary_rpc_method_handler(
          servicer.filter_and_save,
          request_deserializer=epic__image__pb2.epic_image.FromString,
          response_serializer=epic__image__pb2.empty.SerializeToString,
      ),
      'filter_and_save_chunk': grpc.stream_unary_rpc_method_handler(
          servicer.filter_and_save_chunk,
          request_deserializer=epic__image__pb2.epic_image.FromString,
          response_serializer=epic__image__pb2.empty.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'epic_post_process', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
