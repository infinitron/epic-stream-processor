import json
import socket
from concurrent import futures

import grpc
import numpy as np
from epic_grpc import epic_image_pb2
from epic_grpc import epic_image_pb2_grpc
from epic_grpc.epic_image_pb2_grpc import (
    epic_post_processServicer as epic_post_servicer,
)
from numpy.lib.stride_tricks import as_strided
from ServiceHub import ServiceHub
from Watchdog import WatchDog


storage_servicer = ServiceHub()
watcher = WatchDog(storage_servicer)


class epic_postprocessor(epic_post_servicer):  # type: ignore[misc,no-any-unimported]
    def filter_and_save(  # type: ignore[no-any-unimported]
        self,
        request: epic_image_pb2.epic_image,
        context: object,
    ) -> epic_image_pb2.empty:
        # decode the header
        print(json.loads(request.header))

        # decode the numpy array
        img_cube = np.frombuffer(request.image_cube)
        print(img_cube.shape)
        return epic_image_pb2.empty()

    def filter_and_save_chunk(  # type: ignore[no-any-unimported]
        self,
        request_iterator: epic_image_pb2.epic_image,
        context: object,
    ) -> epic_image_pb2.empty:
        header = []
        img_buffer = bytes()
        for image in request_iterator:
            if image.header is not None:
                header.append(image.header)
            img_buffer += image.image_cube

        header = json.loads(header[0])
        # 0: primaryHDU, 1: Image, 2: buffer details
        buffer_metadata = json.loads(header[2])

        # decode the numpy array
        img_array = np.frombuffer(img_buffer, dtype=buffer_metadata["dtype"])
        img_array = as_strided(
            img_array, buffer_metadata["shape"], buffer_metadata["strides"]
        )

        # pixel_idx_df = watcher.get_watch_indices(header[1])
        # pixel_meta_df = pd.DataFrame.from_dict(
        #     watcher.header_to_metadict(header[1], epic_version='0.0.2'))
        # pixel_idx_df['id'] = pixel_meta_df.iloc[0]['uuid']

        # pixel_idx_df = watcher.insert_pixels_df(
        #     pixel_idx_df,
        #     img_cube, pixel_idx_col='patch_pixels', val_col='pixel_values')

        # pixel_idx_df = watcher.format_skypos_pg(
        #     pixel_idx_df, 'patch_skypos', 'pixel_skypos')

        pixel_idx_df, pixel_meta_df = watcher.filter_and_store_imgdata(
            header[1], img_array, epic_version="0.0.2"
        )

        # print()
        print(img_array.shape, pixel_idx_df.columns, pixel_meta_df.columns)
        return epic_image_pb2.empty()


def get_uds_id() -> str:
    """Returns an ID to register the processor on the registry service"""
    # registration logic
    return f"{socket.gethostname()}_epic_processor"


def serve(max_workers: int = 1) -> None:
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[("grpc.max_receive_message_length", -1)],
    )
    print("Setting up")
    epic_image_pb2_grpc.add_epic_post_processServicer_to_server(
        epic_postprocessor(), server
    )
    server.add_insecure_port(f"unix-abstract:{get_uds_id()}")
    print("Starting")
    server.start()
    print(f"Running on {f'unix-abstract:{get_uds_id()}'}...")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
