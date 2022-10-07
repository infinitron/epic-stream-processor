import json
import socket
import traceback
from datetime import datetime
from datetime import timedelta
from importlib.resources import path as res_path
from pathlib import Path
from socket import socket as socket_c

# from socketserver import BaseServer
# from socketserver import StreamRequestHandler
# from socketserver import ThreadingUnixStreamServer
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

import numpy as np
from astropy.io import fits
from astropy.io.fits import Header

from .. import example_data
from .._utils.Utils import get_thread_UDS_addr
from ..epic_grpc.epic_image_pb2 import epic_image
from ..epic_types import NDArrayNum_t
from ..epic_types import Patch_t
from ..epic_types import WatchMode_t


# from epic_stream_processor.epic_grpc import epic_image_pb2
# from .._utils.Utils import DotDict as dotdict


try:
    from socket import _RetAddress
    from socketserver import _AddressType
    from socketserver import _RequestType
except Exception:
    _RequestType = Union[Type[socket_c], Type[Tuple[bytes, socket_c]]]  # type: ignore[misc]
    _AddressType = Union[Type[str], Type[Tuple[str, int]]]  # type: ignore[misc]
    _RetAddress = Type[str]  # type: ignore[misc]


def stream_data_uds(
    primary_hdr: Header, img_hdr: Header, data: NDArrayNum_t, addr: Optional[str]
) -> None:
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
    stream_packed_uds(json.dumps(header), data, addr)


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
        payload = epic_image(header=hdr, image_cube=data.tobytes()).SerializeToString()
        client.connect(addr)
        client.sendall(bytes(json.dumps(["epic_image", len(payload)]), "utf-8"))
        resp = client.recv(256)
        if resp is not None and resp.decode("utf-8") == "proceed":
            print("proceeding")
            client.sendall(payload)
        else:
            pass


def send_man_watch_req(
    source_name: str,
    ra: float,
    dec: float,
    author: str = "batman",
    event_time: Optional[datetime] = None,
    t_start: Optional[datetime] = None,
    t_end: Optional[datetime] = None,
    reason: str = "Detection of FRBs",
    watch_mode: WatchMode_t = "continuous",
    patch_type: Patch_t = "3x3",
    event_type: str = "Manual trigger",
) -> str:
    def fmt_time(t: Optional[datetime], add_s: float = 0) -> str:
        return (
            t.isoformat()
            if t is not None
            else (datetime.utcnow() + timedelta(seconds=add_s)).isoformat()
        )

    try:
        watch_payload = dict(
            source_name=source_name,
            ra=ra,
            dec=dec,
            author=author,
            event_time=fmt_time(event_time),
            t_start=fmt_time(t_start),
            t_end=fmt_time(t_end, add_s=7 * 86400),
            reason=reason,
            watch_mode=watch_mode,
            patch_type=patch_type,
            event_type=event_type,
        )

        addr = get_thread_UDS_addr()
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.connect(addr)
            # client.sendall(b"watch_source")\
            payload = bytes(json.dumps(watch_payload), "utf-8")
            client.sendall(bytes(json.dumps(["watch_source", len(payload)]), "utf-8"))
            resp = client.recv(1024).decode("utf-8")
            if resp == "proceed":
                client.sendall(payload)
                resp = client.recv(1024).decode("utf-8")
                return resp
            else:
                return resp
    except Exception:
        print(traceback.format_exc())
        return ""


# def add_to
