import chunk
import grpc
import numpy as np
import socket
import epic_image_pb2_grpc
import epic_image_pb2
import json
import time

_CHUNK_SIZE_=int(819200)

def get_epic_ppro_uds_id():
    """
    Query the central registry to get the UDS ID
    """
    #querying logic
    return f'unix-abstract:{socket.gethostname()}_epic_processor'

def chunk_data(header,data):
    size = data.size
    buffer = data
    for i in range(0,size,_CHUNK_SIZE_):
        if i+_CHUNK_SIZE_>size:
            if i==0:
                yield epic_image_pb2.epic_image(header=header,image_cube=buffer[i:size].tobytes())
            else:
                yield epic_image_pb2.epic_image(header=None,image_cube=buffer[i:size].tobytes())
        else:
            if i==0:
                yield epic_image_pb2.epic_image(header=header,image_cube=buffer[i:i+_CHUNK_SIZE_].tobytes())
            else:
                yield epic_image_pb2.epic_image(header=None,image_cube=buffer[i:i+_CHUNK_SIZE_].tobytes())


def get_dummy_data():
    for i in range(10):
        header = json.dumps(dict(a=1,b=2))
        data = np.random.random(64*64*100*4*2) #
        time.sleep(1)
        yield chunk_data(header,data)

def run():
    with grpc.insecure_channel(get_epic_ppro_uds_id(),
    options=[('grpc.max_send_message_length',-1)]) as channel:
        stub = epic_image_pb2_grpc.epic_post_processStub(channel)
        print('Starting to send data')
        for i in get_dummy_data():
            t=time.time()
            print('Sending')
            _ = stub.filter_and_save(epic_image_pb2.epic_image(header=i[0],image_cube=i[1]))
            print(time.time()-t)

if __name__=="__main__":
    # for i in get_dummy_data():
    #     print(type(i))
    #     for j in i:
    #         print(j)
    #run()
    channel = grpc.insecure_channel(get_epic_ppro_uds_id(),
    options=[('grpc.max_send_message_length',-1)])
    stub = epic_image_pb2_grpc.epic_post_processStub(channel)
    print('Starting to send data')
    for i in get_dummy_data():
        t=time.time()
        print('Sending',type(i))
        _ = stub.filter_and_save_chunk(i)
        print(time.time()-t)
    channel.close()

