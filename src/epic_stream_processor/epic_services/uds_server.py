from concurrent.futures import ThreadPoolExecutor
import json
import socket
import traceback
# from importlib.resources import path as res_path
# from pathlib import Path

# from socketserver import BaseServer
# from socketserver import StreamRequestHandler
# from socketserver import ThreadingUnixStreamServer
# from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union
from socket import socket as socket_c
# from datetime import datetime, timedelta

import numpy as np
# from astropy.io import fits
# from astropy.io.fits import Header
from numpy.lib.stride_tricks import as_strided
from sqlalchemy.engine import Engine

# from .. import example_data
from .._utils.Utils import get_thread_UDS_addr
from ..epic_grpc.epic_image_pb2 import epic_image
# from ..epic_types import NDArrayNum_t
from .service_hub import ServiceHub
from .watch_dog import EpicPixels
from .watch_dog import WatchDog
# from ..epic_types import WatchMode_t, Patch_t

# from epic_stream_processor.epic_grpc import epic_image_pb2
# from .._utils.Utils import DotDict as dotdict
from dataclasses import dataclass

try:
    from socketserver import _AddressType
    from socketserver import _RequestType
    from socket import _RetAddress
except Exception:
    _RequestType = Union[Type[socket_c], Type[Tuple[bytes, socket_c]]]  # type: ignore[misc]
    _AddressType = Union[Type[str], Type[Tuple[str, int]]]  # type: ignore[misc]
    _RetAddress = Type[str]  # type: ignore[misc]


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


# def stream_data_uds(
#     primary_hdr: Header, img_hdr: Header, data: NDArrayNum_t, addr: Optional[str]
# ) -> None:
#     addr = addr or get_thread_UDS_addr()
#     header = [primary_hdr.header.tostring(), img_hdr.header.tostring()]
#     header.append(
#         json.dumps(
#             dict(
#                 dtype=str(data.dtype),
#                 shape=data.shape,
#                 strides=data.strides,
#             )
#         )
#     )
#     stream_packed_uds(json.dumps(header), data)

    # with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
    #     client.connect(addr)
    #     client.sendall(b"epic_image")
    #     resp = client.recv(256)
    #     if resp is not None and resp.decode("utf-8") == "proceed":
    #         client.sendall(
    #             epic_image(
    #                 header=json.dumps(header), image_cube=data.tobytes()
    #             ).SerializeToString()
    #         )
    #     else:
    #         pass


# def get_dummy_data(
#     dataset: str = "EPIC_1661990950.000000_73.487MHz.fits", randomize: bool = True
# ) -> Tuple[List[str], NDArrayNum_t]:
#     test_file = Path(".")
#     with res_path(example_data, dataset) as f:
#         test_file = f
#     with fits.open(test_file) as hdu:
#         header = [hdu[0].header.tostring(), hdu[1].header.tostring()]
#         data = (
#             np.random.random(hdu[1].data.shape).astype(np.float32)
#             if randomize is True
#             else hdu[1].data
#         )
#         header.append(
#             json.dumps(
#                 dict(
#                     dtype=str(data.dtype),
#                     shape=data.shape,
#                     strides=data.strides,
#                 )
#             )
#         )
#     return header, data


# def stream_packed_uds(hdr: str, data: NDArrayNum_t, addr: str) -> None:
#     addr = addr or get_thread_UDS_addr()
#     with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
#         payload = epic_image(header=hdr, image_cube=data.tobytes()).SerializeToString()
#         client.connect(addr)
#         client.sendall(bytes(json.dumps(["epic_image", len(payload)]), "utf-8"))
#         resp = client.recv(256)
#         if resp is not None and resp.decode("utf-8") == "proceed":
#             print("proceeding")
#             client.sendall(payload)
#         else:
#             pass


# def send_man_watch_req(
#     source_name: str,
#     ra: float,
#     dec: float,
#     author: str = "batman",
#     event_time: Optional[datetime] = None,
#     t_start: Optional[datetime] = None,
#     t_end: Optional[datetime] = None,
#     reason: str = "Detection of FRBs",
#     watch_mode: WatchMode_t = "continuous",
#     patch_type: Patch_t = "3x3",
#     event_type: str = "Manual trigger",
# ) -> str:
#     def fmt_time(t: Optional[datetime], add_s: float = 0):
#         return (
#             t.isoformat()
#             if t is not None
#             else (datetime.utcnow() + timedelta(seconds=add_s)).isoformat()
#         )

#     try:
#         watch_payload = dict(
#             source_name=source_name,
#             ra=ra,
#             dec=dec,
#             author=author,
#             event_time=fmt_time(event_time),
#             t_start=fmt_time(t_start),
#             t_end=fmt_time(t_end, add_s=7 * 86400),
#             reason=reason,
#             watch_mode=watch_mode,
#             patch_type=patch_type,
#             event_type=event_type,
#         )

#         addr = get_thread_UDS_addr()
#         with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
#             client.connect(addr)
#             # client.sendall(b"watch_source")\
#             payload = bytes(json.dumps(watch_payload), "utf-8")
#             client.sendall(bytes(json.dumps(["watch_source", len(payload)]), "utf-8"))
#             resp = client.recv(1024).decode("utf-8")
#             if resp == "proceed":
#                 client.sendall(payload)
#                 resp = client.recv(1024).decode("utf-8")
#                 return resp
#             else:
#                 return resp
#         return ""
#     except Exception:
#         print(traceback.format_exc())


@dataclass
class ThreadedServerContext:
    """Dataclass for the UDS server context."""

    service_hub: ServiceHub = None
    watch_dog: WatchDog = None


class Processors:
    @staticmethod
    def epic_image_p(
        buffer: bytes, ctx: ThreadedServerContext, client: socket_c
    ) -> None:
        img_data = epic_image.FromString(buffer)
        if img_data.ByteSize() == 0:
            # wth?
            return
        header = json.loads(img_data.header)
        # 0: primaryHdr, 1: ImageHdr, 2: buffer metadata
        buffer_metadata = json.loads(header[2])

        # decode the numpy array
        img_array = np.frombuffer(img_data.image_cube, dtype=buffer_metadata["dtype"])

        # check the integrity of the array
        if img_array.size != np.prod(buffer_metadata["shape"]):
            raise Exception(
                f"Data lost in the transit. Expected shape {buffer_metadata['shape']} and deduced shape {img_array.shape} are unequal"
            )

        img_array = as_strided(
            img_array, buffer_metadata["shape"], buffer_metadata["strides"]
        )
        # print("Received:", img_array.shape)

        pixels = EpicPixels(
            header[1],
            header[0],
            img_array,
            ctx.watch_dog._watch_df,
            epic_ver="0.0.2",
        )
        pixels.gen_pixdata_dfs()
        pixels.store_pg(ctx.service_hub)

    @staticmethod
    def watch_source_p(watch_conf: str, ctx: ThreadedServerContext, client: socket_c):
        try:
            config: dict = json.loads(watch_conf)
            req_pars = ["source_name", "ra", "dec", "author"]
            for par in req_pars:
                if par not in req_pars:
                    raise Exception(f"{par} is required to watch")
            ctx.watch_dog.add_source_and_watch(**config)
            client.sendall(b"added")
        except Exception:
            client.sendall(bytes(
                traceback.format_exc(), "utf-8"
            ))
            client.close()
            # print(traceback.format_exc())


class ThreadedServer(object):
    def __init__(
        self, addr: Optional[str], max_conn: int = 5, engine: Optional[Engine] = None
    ) -> None:
        self.addr = addr or get_thread_UDS_addr()
        self.max_conn = max_conn
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        # self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(self.addr)
        self.ctx = ThreadedServerContext()
        self.ctx.service_hub = ServiceHub(engine=engine)
        self.ctx.watch_dog = WatchDog(self.ctx.service_hub)
        self.executor = ThreadPoolExecutor(max_workers=10)

    def listen(self) -> None:
        self.sock.listen(self.max_conn)
        while True:
            client, address = self.sock.accept()
            client.settimeout(60)
            self.executor.submit(self.listen_to_client, client, address).result()
            # threading.Thread(
            #     target=self.listen_to_client, args=(client, address)
            # ).start()

    def handshake(self, client: socket_c) -> Tuple[Optional[str], Optional[int]]:
        data = client.recv(256).decode("utf-8")  # receive the definition

        try:
            if data is not None:
                processor, payload_size = json.loads(data)
                processor_f = processor + "_p"
                if hasattr(Processors, processor_f):
                    client.sendall(b"proceed")
                    return processor_f, payload_size
                else:
                    client.sendall(b"Invalid data type")
                    client.close()
                    return None, None
            else:
                return None, None

            # processor_f = data.decode("utf-8") + "_p"
        except Exception:
            client.sendall(traceback.format_exc())
            client.close()
            print(traceback.format_exc())
            return None, None

    def listen_to_client(self, client: socket_c, address: _RetAddress) -> None:
        size = 1 << 20
        buffer = bytes()
        # data = client.recv(256)  # receive the definition
        # try:
        #     if data is not None and hasattr(Processors, data.decode("utf-8") + "_p"):
        #         client.sendall(b"proceed")
        #     else:
        #         client.sendall(b"Invalid data type")
        #         client.close()
        #         return

        #     processor_f = data.decode("utf-8") + "_p"
        # except Exception:
        #     client.sendall(traceback.format_exc())
        #     client.close()
        #     print(traceback.format_exc())

        processor_f, payload_size = self.handshake(client)
        # print(processor_f, payload_size)

        # if a valid processor is requested, grab all the data
        while len(buffer) < payload_size:
            try:
                data = client.recv(size)
                if data:
                    buffer += data
                else:
                    break
            except Exception:
                client.close()

        try:
            getattr(Processors, processor_f)(buffer, self.ctx, client)

        except Exception:
            client.close()
            # print("Exception")
            # print(print(traceback.format_exc()))
