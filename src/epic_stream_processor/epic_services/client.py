import json
import time
from importlib.resources import path as res_path
from pathlib import Path
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple

import grpc
import numpy as np
from astropy.io import fits

from .. import example_data
from .._utils import get_epic_stpro_uds_id
from ..epic_grpc import epic_image_pb2
from ..epic_grpc.epic_image_pb2 import empty
from ..epic_grpc.epic_image_pb2 import epic_image
from ..epic_grpc.epic_image_pb2_grpc import epic_post_processStub
from ..epic_types import NDArrayNum_t


_CHUNK_SIZE_ = int(819200)


class EpicRPCClient:
    def __init__(self, connect: bool = False) -> None:
        self._connected = False
        if connect:
            self.connect()
            self._connected = True

    def connect(self) -> None:
        self._channel = grpc.insecure_channel(
            get_epic_stpro_uds_id(),
            options=[
                ("grpc.max_send_message_length", -1),
                ("grpc.max_concurrent_streams", 100),
                ("grpc.http2.lookahead_bytes", 1 << 13)
                # ("SingleThreadedUnaryStream", 1),
            ],
        )
        self._stub = epic_post_processStub(self._channel)

        # self._aio_channel = grpc.aio.insecure_channel(
        #     get_epic_stpro_uds_id(),
        #     options=[
        #         ("grpc.max_send_message_length", -1),
        #         ("SingleThreadedUnaryStream", 1),
        #     ],
        # )

        # self._aio_stub = epic_post_processStub(self._aio_channel)

    def chunk_data(
        self,
        headers: List[str],
        data: NDArrayNum_t,
        chunk_size: int = 1 << 16,
    ) -> Iterator[epic_image]:
        size = data.size
        buffer = data
        image_cube = bytes()
        hdr: str = ""
        print("CHUNK:", chunk_size)
        for i in range(0, size, chunk_size):
            hdr = json.dumps(headers) if i == 0 else ""
            if i + chunk_size > size:
                image_cube = buffer[i:size].tobytes()
            else:
                image_cube = buffer[i : i + chunk_size].tobytes()

            yield epic_image_pb2.epic_image(header=hdr, image_cube=image_cube)

    def get_dummy_data(
        self,
        dataset: str = "EPIC_1661990950.000000_73.487MHz.fits",
    ) -> Tuple[List[str], NDArrayNum_t]:
        test_file = Path(".")
        with res_path(example_data, dataset) as f:
            test_file = f
        with fits.open(test_file) as hdu:
            header = [hdu[0].header.tostring(), hdu[1].header.tostring()]
            data = hdu[1].data
            data = np.random.random(data.shape).astype(np.float32)
            header.append(
                json.dumps(
                    dict(
                        dtype=str(data.dtype),
                        shape=data.shape,
                        strides=data.strides,
                    )
                )
            )
        return header, data

    def get_dummy_stream(
        self,
        n_iter: int = 10,
        cadence: float = 1,
        dataset: str = "EPIC_1661990950.000000_73.487MHz.fits",
        chunk_size: int = 1 << 16,
    ) -> Iterator[Iterator[epic_image]]:
        # test_file = Path(".")
        # with res_path(example_data, dataset) as f:
        #     test_file = f
        # th fits.open(test_file) as hdu:
        header, data = self.get_dummy_data(dataset)
        for _ in range(n_iter):
            # header = json.dumps(dict(a=1,b=2))
            # data = np.random.random(64*64*32*4*2) #
            # header = [hdu[0].header.tostring(), hdu[1].header.tostring()]
            # data = hdu[1].data
            # data = np.random.random(data.shape).astype(data.dtype)
            # header.append(
            #     json.dumps(
            #         dict(
            #             dtype=str(data.dtype),
            #             shape=data.shape,
            #             strides=data.strides,
            #         )
            #     )
            # )

            time.sleep(cadence)
            yield self.chunk_data(header, data, chunk_size=chunk_size)

    def send_dummy_data(
        self, n_items: int = 10, cadence: float = 1, chunk_size: int = 1 << 16
    ) -> None:
        # with grpc.insecure_channel(
        #     self.get_epic_ppro_uds_id(),
        #     options=[
        #         ("grpc.max_send_message_length", -1),
        #         # ("SingleThreadedUnaryStream", 1),
        #     ],
        # ) as channel:
        print("Starting to send data")
        try:
            for i in self.get_dummy_stream(n_items, cadence, chunk_size=chunk_size):
                t = time.time()
                print("Sending")
                _ = self._stub.filter_and_save_chunk(i)
                print(time.time() - t)
        except Exception as e:
            print("Unable to send data")
            print(e)

    def send_data(self, header_arr: List[str], data: NDArrayNum_t) -> Optional[empty]:
        try:
            response: empty = self._stub.filter_and_save_chunk(
                self.chunk_data(header_arr, data)
            )

            return response
        except Exception as e:
            print(e)
            return None

    # async def send_data_aio(
    #     self, header_arr: List[str], data: NDArrayNum_t
    # ) -> Optional[empty]:
    #     raise NotImplementedError()
    #     try:
    #         response: empty = await self._aio_stub.filter_and_save_chunk(
    #             self.chunk_data(header_arr, data)
    #         )

    #         return response
    #     except Exception as e:
    #         print(e)
    #         return None

    def __del__(self) -> None:
        if self._connected:
            self._channel.close()


# if __name__ == "__main__":
#     # for i in get_dummy_data():
#     #     print(type(i))
#     #     for j in i:
#     #         print(j)
#     # run()
#     channel = grpc.insecure_channel(
#         get_epic_ppro_uds_id(), options=[("grpc.max_send_message_length", -1)]
#     )
#     stub = epic_image_pb2_grpc.epic_post_processStub(channel)  # type: ignore
#     print("Starting to send data")
#     for i in get_dummy_data():
#         t = time.time()
#         print("Sending", type(i))
#         _ = stub.filter_and_save_chunk(i)
#         print(time.time() - t)
#     channel.close()
