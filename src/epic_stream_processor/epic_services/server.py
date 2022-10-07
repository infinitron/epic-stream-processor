import json
import warnings
from concurrent import futures
from timeit import default_timer as timer
from typing import Iterator
from typing import Optional

import grpc
import numpy as np
from numpy.lib.stride_tricks import as_strided

from .._utils import get_epic_stpro_uds_id
from ..epic_grpc import epic_image_pb2
from ..epic_grpc import epic_image_pb2_grpc
from ..epic_grpc.epic_image_pb2 import empty
from ..epic_grpc.epic_image_pb2 import epic_image
from ..epic_grpc.epic_image_pb2_grpc import (
    epic_post_processServicer as epic_post_servicer,
)
from .service_hub import ServiceHub
from .watch_dog import EpicPixels
from .watch_dog import WatchDog


class epic_postprocessor(epic_post_servicer):
    warnings.simplefilter("ignore", DeprecationWarning)
    warnings.warn(
        "Server based on gRPC is much slower than the ThreadedServer implementation. \
        This module will be removed in the future versions.",
        DeprecationWarning,
    )

    def __init__(
        self,
        storage_servicer: Optional[ServiceHub] = None,
        use_default_servicer: bool = False,
    ) -> None:
        super().__init__()
        self.storage_servicer = None
        if storage_servicer is not None:
            self.storage_servicer = storage_servicer

        if use_default_servicer is True:
            self.storage_servicer = ServiceHub()

        self.watcher = WatchDog(self.storage_servicer)
        # self._pipeline = Stream(ensure_io_loop=True)
        # self._pipeline.map(self._fs)#.sink(self._oblivion)

    def set_storage_servicer(self, servicer: ServiceHub) -> None:
        self.watcher.change_storage_servicer(servicer=servicer)

    def filter_and_save(
        self,
        request: epic_image,
        context: object,
    ) -> empty:
        # decode the header
        print(json.loads(request.header))

        # decode the numpy array
        img_cube = np.frombuffer(request.image_cube)
        print(img_cube.shape)
        return epic_image_pb2.empty()

    def filter_and_save_chunk(
        self,
        request_iterator: Iterator[epic_image],
        context: object,
    ) -> empty:
        # return empty()
        # self._ppl.emit(request_iterator)

        start = timer()
        header = []
        # img_buffer = bytearray()
        # for image in request_iterator:
        #     if image.header is not None:
        #         header.append(image.header)
        #     img_buffer += image.image_cube

        def temp(hdr: str, img: bytes) -> bytes:
            if hdr is not None:
                header.append(hdr)
            return img

        img_buffer = [temp(i.header, i.image_cube) for i in request_iterator]
        # img_buffer = b''.join(img_buffer)
        print(f"Elapsed0: {timer()-start}")
        header = json.loads(header[0])
        # 0: primaryHDU, 1: Image, 2: buffer details
        buffer_metadata = json.loads(header[2])

        # decode the numpy array
        img_array = np.frombuffer(b"".join(img_buffer), dtype=buffer_metadata["dtype"])
        img_array = as_strided(
            img_array, buffer_metadata["shape"], buffer_metadata["strides"]
        )
        print(img_array.nbytes, 1 << 16, len(img_buffer))
        print(f"Elapsed: {timer()-start}")
        # return empty()
        # pixel_idx_df = watcher.get_watch_indices(header[1])
        # pixel_meta_df = pd.DataFrame.from_dict(
        #     watcher.header_to_metadict(header[1], epic_version='0.0.2'))
        # pixel_idx_df['id'] = pixel_meta_df.iloc[0]['uuid']

        # pixel_idx_df = watcher.insert_pixels_df(
        #     pixel_idx_df,
        #     img_cube, pixel_idx_col='patch_pixels', val_col='pixel_values')

        # pixel_idx_df = watcher.format_skypos_pg(
        #     pixel_idx_df, 'patch_skypos', 'pixel_skypos')

        ###############
        start = timer()
        # pixel_idx_df, pixel_meta_df =
        # self.watcher.filter_and_store_imgdata(
        #     header[1], img_array, epic_version="0.0.2"
        # )
        pixels = EpicPixels(
            header[1], header[0], img_array, self.watcher._watch_df, epic_ver="0.0.2"
        )
        pixels.gen_pixdata_dfs()
        pixels.store_pg(self.watcher._service_Hub)
        # self._pipeline.emit((header[1], img_array, "0.0.2"))
        print(f"Elapsed2: {timer()-start}")

        # self._pipeline.emit((header[1], img_array, "0.0.2"))
        ####################

        # print()
        # print(img_array.shape, pixel_idx_df.columns, pixel_meta_df.columns)
        return empty()


# def get_uds_id() -> str:
#     """Returns an ID to register the processor on the registry service"""
#     # registration logic
#     return f"{socket.gethostname()}_epic_processor"


def serve(max_workers: int = 1) -> None:
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[("grpc.max_receive_message_length", -1)],
        maximum_concurrent_rpcs=2,
    )
    print("Setting up")
    epic_image_pb2_grpc.add_epic_post_processServicer_to_server(
        epic_postprocessor(use_default_servicer=True), server
    )
    server.add_insecure_port(get_epic_stpro_uds_id())
    print("Starting")
    server.start()
    print(f"Running on {get_epic_stpro_uds_id()}...")
    server.wait_for_termination()


# async def serve_aio(max_workers: int = 1) -> None:
#     server = grpc.aio.server(
#         futures.ThreadPoolExecutor(max_workers=max_workers),
#         options=[("grpc.max_receive_message_length", -1)],
#     )
#     print("Setting up (asyn mode)")
#     epic_image_pb2_grpc.add_epic_post_processServicer_to_server(
#         epic_postprocessor(use_default_servicer=True), server
#     )
#     server.add_insecure_port(get_epic_stpro_uds_id())
#     print("Starting")
#     await server.start()
#     print(f"Running on {f'unix-abstract:{get_uds_id()}'}...")
#     await server.wait_for_termination()
