import json
import socket
import traceback
from concurrent.futures import ThreadPoolExecutor

# from epic_stream_processor.epic_grpc import epic_image_pb2
# from .._utils.Utils import DotDict as dotdict
from dataclasses import dataclass
from socket import socket as socket_c

# from socketserver import BaseServer
# from socketserver import StreamRequestHandler
# from socketserver import ThreadingUnixStreamServer
# from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

import numpy as np
import os

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

from subprocess import Popen, DEVNULL, STDOUT


# from importlib.resources import path as res_path
# from pathlib import Path


# from datetime import datetime, timedelta


# from ..epic_types import WatchMode_t, Patch_t


try:
    from socket import _RetAddress
    from socketserver import _AddressType
    from socketserver import _RequestType
except Exception:
    _RequestType = Union[Type[socket_c], Type[Tuple[bytes, socket_c]]]  # type: ignore[misc]
    _AddressType = Union[Type[str], Type[Tuple[str, int]]]  # type: ignore[misc]
    _RetAddress = Type[str]  # type: ignore[misc]


@dataclass
class ThreadedServerContext:
    """Dataclass for the UDS server context."""

    service_hub: ServiceHub
    watch_dog: WatchDog


_epic_python_executable = "/home/epic/anaconda/envs/work/bin/python"
_epic_script = "/home/epic/batman/LWA_EPIC/LWA_EPIC/LWA_EPIC_jdd.py"


def _get_VMA_env():
    env = os.environ.copy()
    env["VMA_RX_POLL"] = "1000"
    env["VMA_INTERNAL_THREAD_AFFINITY"] = "0"
    env["VMA_RX_PREFETCH_BYTES"] = "128"
    env["VMA_THREAD_MODE"] = "0"
    env["VMA_MTU"] = "9000"
    env["VMA_TRACELEVEL"] = "1,2,3,4"
    env["LD_PRELOAD"] = "libvma.so"
    return env


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
            elevation_limit=30,
        )
        pixels.gen_pixdata_dfs()
        pixels.store_pg(ctx.service_hub)

    @staticmethod
    def watch_source_p(
        watch_conf: bytes, ctx: ThreadedServerContext, client: socket_c
    ) -> None:
        try:
            config = json.loads(watch_conf)
            req_pars = ["source_name", "ra", "dec", "author"]
            for par in req_pars:
                if par not in req_pars:
                    raise Exception(f"{par} is required to watch")
            ctx.watch_dog.add_source_and_watch(**config)
            client.sendall(b"added")
        except Exception:
            client.sendall(bytes(traceback.format_exc(), "utf-8"))
            client.close()
            # print(traceback.format_exc())

    @staticmethod
    def run_epic_p(
        epic_args: bytes, ctx: ThreadedServerContext, client: socket_c
    ) -> None:
        try:
            args = epic_args.decode("utf-8")
            cur_epic_instances = ctx.service_hub.get_num_epic_instances()
            if cur_epic_instances == 2:
                client.sendall(
                    bytes(
                        "Cannot run more than two instances\
                     of EPIC simultaneously"
                    )
                )
                client.close()

            run_env = _get_VMA_env()
            bind_num = 0 if cur_epic_instances < 2 else 1
            executable = f"numactl --cpubind={bind_num} --membind={bind_num}"
            executable = f"{executable} {_epic_python_executable} {_epic_script} {args}"

            # launch the instance
            process = Popen(
                executable.split(" "),
                env=run_env,
                stdout=DEVNULL,
                stderr=STDOUT,
            )
            ctx.service_hub.add_epic_instance(
                dict(
                    options=executable,
                    process=process,
                )
            )
            pid = process.pid
            print(f"Running EPIC instance. PID {pid}")
            # ctx.service_hub.epic_instances[-1].run()
            client.sendall(
                bytes(
                    f"Running {pid}",
                    "utf-8",
                )
            )
        except Exception:
            client.sendall(bytes(traceback.format_exc(), "utf-8"))
            client.close()

    def get_epic_instances_p(
        _: bytes, ctx: ThreadedServerContext, client: socket_c
    ) -> None:
        try:
            instances = ctx.service_hub.get_epic_instances()
            client.sendall(bytes(json.dumps(instances), "utf-8"))
        except Exception:
            client.sendall(bytes(traceback.format_exc(), "utf-8"))
            client.close()


class ThreadedServer:
    def __init__(
        self, addr: Optional[str], max_conn: int = 5, engine: Optional[Engine] = None
    ) -> None:
        self.addr = addr or get_thread_UDS_addr()
        self.max_conn = max_conn
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        # self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(self.addr)
        self.service_hub = ServiceHub(engine=engine)
        self.watch_dog = WatchDog(self.service_hub)
        self.ctx = ThreadedServerContext(self.service_hub, self.watch_dog)
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
            if data == "":
                return None, None

            processor, payload_size = json.loads(data)
            processor_f = processor + "_p"
            if hasattr(Processors, processor_f):
                client.sendall(b"proceed")
                return processor_f, payload_size
            else:
                return None, None
        except Exception:
            client.sendall(bytes(traceback.format_exc(), "utf-8"))
            client.close()
            # print(traceback.format_exc())
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
        if processor_f is None or payload_size is None:
            client.sendall(b"Invalid data type")
            client.close()
            return

        if payload_size == 0:
            client.sendall(b"Payload size cannot be zero.")
            client.close()
            return

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
            print(print(traceback.format_exc()))
