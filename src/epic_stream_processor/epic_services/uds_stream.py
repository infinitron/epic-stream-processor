import json
import socket
import threading
import traceback
from importlib.resources import path as res_path
from pathlib import Path
# from socketserver import BaseServer
# from socketserver import StreamRequestHandler
# from socketserver import ThreadingUnixStreamServer
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union
from socket import socket as socket_c

import numpy as np
from astropy.io import fits
from astropy.io.fits import Header
from numpy.lib.stride_tricks import as_strided

from .. import example_data
from .._utils.Utils import get_thread_UDS_addr
from ..epic_grpc.epic_image_pb2 import epic_image
from ..epic_types import NDArrayNum_t
from .service_hub import ServiceHub
from .watch_dog import EpicPixels
from .watch_dog import WatchDog


try:
    from socketserver import _AddressType
    from socketserver import _RequestType
    from socket import _RetAddress
except Exception:
    _RequestType = Union[Type[socket_c], Type[Tuple[bytes, socket_c]]]  # type: ignore[misc]
    _AddressType = Union[Type[str], Type[Tuple[str, int]]]  # type: ignore[misc]
    _RetAddress = Type[str]    # type: ignore[misc]


# class EpicUDSHandler(StreamRequestHandler):
#     def __init__(
#         self, request: _RequestType, client_address: _AddressType, server: BaseServer
#     ) -> None:
#         super().__init__(request, client_address, server)
#         # self._service_hub = ServiceHub()
#         # self._watch_dog = WatchDog(self._service_hub)
#         # print("Initialized the handler")

#     def handle(self) -> None:
#         buffer = bytes()
#         try:
#             while True:
#                 data = self.rfile.readline()
#                 # data = data.strip()
#                 if data:
#                     buffer += data
#                 else:
#                     break

#             # print(buffer)
#             # return
#             img_data = epic_image.FromString(buffer)
#             if img_data.ByteSize() == 0:
#                 # wth?
#                 return
#             header = json.loads(img_data.header)
#             # 0: primaryHdr, 1: ImageHdr, 2: buffer metadata
#             buffer_metadata = json.loads(header[2])

#             # decode the numpy array
#             img_array = np.frombuffer(
#                 b"".join(img_data.image_cube), dtype=buffer_metadata["dtype"]
#             )

#             # check the integrity of the array
#             if img_array.size != np.prod(buffer_metadata["shape"]):
#                 raise Exception(
#                     f"Data lost in the transit. Expected shape {buffer_metadata['shape']} and deduced shape {img_array.shape} are unequal"
#                 )

#             img_array = as_strided(
#                 img_array, buffer_metadata["shape"], buffer_metadata["strides"]
#             )
#             print("Received:", img_array.shape)

#             pixels = EpicPixels(
#                 header[1],
#                 header[0],
#                 img_array,
#                 self._watch_dog._watch_df,
#                 epic_ver="0.0.2",
#             )
#             pixels.gen_pixdata_dfs()
#             pixels.store_pg(self._service_hub)

#         except Exception as e:
#             print("Exception")
#             print(e)


# def serve() -> None:
#     addr = get_thread_UDS_addr()
#     print("Setting up..")
#     with ThreadingUnixStreamServer(
#         addr, EpicUDSHandler, bind_and_activate=True
#     ) as server:
#         # server.allow_reuse_address = True
#         server.socket.listen(6)
#         print("Started")
#         server.serve_forever()


def stream_data_uds(primary_hdr: Header, img_hdr: Header, data: NDArrayNum_t, addr: Optional[str]) -> None:
    addr = addr or get_thread_UDS_addr()
    header = [primary_hdr.header.tostring(), img_hdr.header.tostring()]
    header.append(
        json.dumps(
            dict(
                dtype=str(data.dtype),
                shape=data.shape,
                strides=data.strides,
            )
        )
    )
    
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(addr)
        client.send(
            epic_image(header=json.dumps(header), image_cube=data.tobytes()).SerializeToString()
        )


def get_dummy_data(
    dataset: str = "EPIC_1661990950.000000_73.487MHz.fits", randomize: bool = True
) -> Tuple[List[str], NDArrayNum_t]:
    test_file = Path(".")
    with res_path(example_data, dataset) as f:
        test_file = f
    with fits.open(test_file) as hdu:
        header = [hdu[0].header.tostring(), hdu[1].header.tostring()]
        data = (
            np.random.random(hdu[1].data.shape).astype(np.float32)
            if randomize is True
            else hdu[1].data
        )
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


def stream_packed_uds(hdr: str, data: NDArrayNum_t, addr: str) -> None:
    addr = addr or get_thread_UDS_addr()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(addr)
        client.sendall(
            epic_image(header=hdr, image_cube=data.tobytes()).SerializeToString()
        )
    # sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    # sock.connect(addr)
    # sock.sendall(epic_image(header=hdr, image_cube=data.tobytes()).SerializeToString())
    # sock.close()


class ThreadedServer(object):
    def __init__(self, addr: Optional[str], max_conn: int = 5) -> None:
        self.addr = addr or get_thread_UDS_addr()
        self.max_conn = max_conn
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        #self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(self.addr)
        self._service_hub = ServiceHub()
        self._watch_dog = WatchDog(self._service_hub)

    def listen(self) -> None:
        self.sock.listen(self.max_conn)
        while True:
            client, address = self.sock.accept()
            client.settimeout(60)
            threading.Thread(
                target=self.listen_to_client, args=(client, address)
            ).start()

    def listen_to_client(self, client: socket_c, address: _RetAddress) -> None:
        size = 1 << 20
        buffer = bytes()
        while True:
            try:
                data = client.recv(size)
                if data:
                    # Set the response to echo back the recieved data
                    buffer += data
                    # client.send(response)
                else:
                    break
                    # raise Exception("Client disconnected")
            except Exception:
                client.close()

        try:
            img_data = epic_image.FromString(buffer)
            if img_data.ByteSize() == 0:
                # wth?
                return
            header = json.loads(img_data.header)
            # 0: primaryHdr, 1: ImageHdr, 2: buffer metadata
            buffer_metadata = json.loads(header[2])

            # decode the numpy array
            img_array = np.frombuffer(
                img_data.image_cube, dtype=buffer_metadata["dtype"]
            )

            # check the integrity of the array
            if img_array.size != np.prod(buffer_metadata["shape"]):
                raise Exception(
                    f"Data lost in the transit. Expected shape {buffer_metadata['shape']} and deduced shape {img_array.shape} are unequal"
                )

            img_array = as_strided(
                img_array, buffer_metadata["shape"], buffer_metadata["strides"]
            )
            print("Received:", img_array.shape)

            pixels = EpicPixels(
                header[1],
                header[0],
                img_array,
                self._watch_dog._watch_df,
                epic_ver="0.0.2",
            )
            pixels.gen_pixdata_dfs()
            pixels.store_pg(self._service_hub)

        except Exception:
            #print("Exception")
            print(print(traceback.format_exc()))
