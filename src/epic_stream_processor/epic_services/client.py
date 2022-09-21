import json
import socket
import time
from typing import Iterator
from typing import List
from typing import Optional

import grpc
from astropy.io import fits

from ..epic_grpc import epic_image_pb2
from ..epic_grpc import epic_image_pb2_grpc
from ..epic_grpc.epic_image_pb2 import epic_image
from ..epic_types import NDArrayNum_t


_CHUNK_SIZE_ = int(819200)


def get_epic_ppro_uds_id() -> str:
    """
    Query the central registry to get the UDS ID
    """
    # querying logic
    return f"unix-abstract:{socket.gethostname()}_epic_processor"


def chunk_data(  # type: ignore[no-any-unimported]
    headers: List[str],
    data: NDArrayNum_t,
    chunk_size: int = 819200,
) -> Iterator[epic_image]:
    size = data.size
    buffer = data
    image_cube = bytes()
    hdr: Optional[str] = ""
    for i in range(0, size, chunk_size):
        hdr = json.dumps(headers) if i == 0 else None
        if i + chunk_size > size:
            image_cube = buffer[i:size].tobytes()
        else:
            image_cube = buffer[i : i + chunk_size].tobytes()

        yield epic_image_pb2.epic_image(header=hdr, image_cube=image_cube)


def get_dummy_data() -> Iterator[epic_image]:  # type: ignore[no-any-unimported]
    with fits.open("EPIC_1661990950.000000_73.487MHz.fits") as hdu:
        for i in range(1):
            # header = json.dumps(dict(a=1,b=2))
            # data = np.random.random(64*64*32*4*2) #
            header = [hdu[0].header.tostring(), hdu[1].header.tostring()]
            data = hdu[1].data
            header.append(
                json.dumps(
                    dict(dtype=str(data.dtype), shape=data.shape, strides=data.strides)
                )
            )
            time.sleep(1)
            yield chunk_data(header, data)


def run() -> None:
    with grpc.insecure_channel(
        get_epic_ppro_uds_id(), options=[("grpc.max_send_message_length", -1)]
    ) as channel:
        stub = epic_image_pb2_grpc.epic_post_processStub(channel)  # type: ignore
        print("Starting to send data")
        for i in get_dummy_data():
            t = time.time()
            print("Sending")
            _ = stub.filter_and_save(
                epic_image_pb2.epic_image(header=i[0], image_cube=i[1])
            )
            print(time.time() - t)


if __name__ == "__main__":
    # for i in get_dummy_data():
    #     print(type(i))
    #     for j in i:
    #         print(j)
    # run()
    channel = grpc.insecure_channel(
        get_epic_ppro_uds_id(), options=[("grpc.max_send_message_length", -1)]
    )
    stub = epic_image_pb2_grpc.epic_post_processStub(channel)  # type: ignore
    print("Starting to send data")
    for i in get_dummy_data():
        t = time.time()
        print("Sending", type(i))
        _ = stub.filter_and_save_chunk(i)
        print(time.time() - t)
    channel.close()
