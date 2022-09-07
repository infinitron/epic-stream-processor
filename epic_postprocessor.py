from concurrent import futures

import grpc
import epic_image_pb2_grpc
import epic_image_pb2
import json
import numpy as np
import socket

class epic_postprocessor(epic_image_pb2_grpc.epic_post_processServicer):
    def filter_and_save(self, request, context):
        #decode the header
        print(json.loads(request.header))

        #decode the numpy array
        img_cube =np.frombuffer(request.image_cube)
        print(img_cube.shape)
        return epic_image_pb2.empty()

    def filter_and_save_chunk(self, request_iterator, context):
        header = {}
        header_recvd_flag = False
        img_cube=bytes()
        for image in request_iterator:
            if header_recvd_flag is False and image.header is not None:
                print(json.loads(image.header))
                header_recvd_flag = True
            img_cube += image.image_cube
        #decode the numpy array
        img_cube =np.frombuffer(img_cube)
        print(img_cube.shape)
        return epic_image_pb2.empty() 
            



def get_uds_id():
    """Returns an ID to register the processor on the registry service"""
    # registration logic 
    return f'{socket.gethostname()}_epic_processor'


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1)
    ,options=[('grpc.max_receive_message_length',-1)])
    print('Setting up')
    epic_image_pb2_grpc.add_epic_post_processServicer_to_server(epic_postprocessor(),server)
    server.add_insecure_port(f'unix-abstract:{get_uds_id()}')
    print('Starting')
    server.start()
    print(f"Running on {f'unix-abstract:{get_uds_id()}'}...")
    server.wait_for_termination()

if __name__=="__main__":
    serve()