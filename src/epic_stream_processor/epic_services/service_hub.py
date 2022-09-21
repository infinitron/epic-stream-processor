from __future__ import annotations

from datetime import datetime
from datetime import timedelta
from typing import Any
from typing import Callable
from typing import TypeVar

import pandas as pd
import psycopg
from apscheduler.schedulers.background import BackgroundScheduler
from geoalchemy2 import Geometry
from pytz import utc  # type: ignore[import]
from sqlalchemy import create_engine
from streamz import Stream


T = TypeVar("T")


class ServiceHub:
    _pg_conn = None
    _scheduler = BackgroundScheduler(timezone=utc)
    _pipeline = Stream()
    _pg_engine = create_engine("postgresql:///postgres?host=/var/run/postgresql")

    def __init__(self) -> None:
        """connect to ectd and and also have a scheduler from apscheduler"""
        self._scheduler.start()
        self._connect_pgdb()

        # define the stream processor timed_window(5)
        self._pipeline.map(ServiceHub.detect_transient).timed_window(1).sink(
            ServiceHub.insert_pgdb
        )

    def __new__(cls) -> ServiceHub:
        if not hasattr(cls, "instance"):
            cls.instance = super().__new__(cls)
        return cls.instance

    def _connect_pgdb(self) -> None:
        # establish postgres connection
        # TODO: fetch the connection details from sys config service

        try:
            self._pg_conn = psycopg.connect(dbname="postgres", user="batman")
            self._pg_conn.autocommit = True
            print("Postgres connection established")
        except Exception as e:
            print(e)
            print("Retrying connection in 60 seconds")
            self.schedule_job_delay(self._connect_pgdb, None, 60)

    def insert_into_db(  # type: ignore[no-any-unimported]
        self,
        pixel_df: pd.DataFrame,
        pixel_meta_df: pd.DataFrame,
    ) -> None:
        self._pipeline.emit(dict(pixels=pixel_df, meta=pixel_meta_df))

    @staticmethod
    def detect_transient(upstream: object) -> object:
        return upstream

    @staticmethod
    def insert_pgdb(  # type: ignore[no-any-unimported]
        upstream: list[dict[pd.DataFrame, pd.DataFrame]]
    ) -> None:
        pixels_df = []
        metadata_df = []

        if len(upstream) < 1:
            return

        for i in upstream:
            pixels_df.append(i["pixels"])
            metadata_df.append(i["meta"])

        pixels_df_merged = pd.concat(pixels_df)
        metadata_df_merged = pd.concat(metadata_df)

        pixels_df_merged.to_sql(
            "epic_pixels",
            con=ServiceHub._pg_engine,
            dtype=dict(pixel_skypos=Geometry(geometry_type="POINT", srid=4326)),
            if_exists="append",
            index=False,
        )

        metadata_df_merged.to_sql(
            "epic_img_metadata",
            con=ServiceHub._pg_engine,
            if_exists="append",
            index=False,
        )

    def pg_query(self, query: str, args: tuple[Any, ...] | None = None) -> list[Any]:
        if self._pg_conn is None:
            raise (ConnectionRefusedError("Could not establish connection to the pgdb"))
        else:
            result = []
            result = self._pg_conn.execute(query, args).fetchall()
            return result

    def schedule_job_delay(
        self,
        func: Callable[..., Any],
        args: tuple[Any, ...] | None = None,
        seconds: int | float = 604800,
    ) -> None:
        self._scheduler.add_job(
            func,
            trigger="date",
            args=args,
            run_date=datetime.now() + timedelta(seconds),
        )

    def schedule_job_date(
        self,
        func: Callable[..., Any],
        args: tuple[Any, ...] | None = None,
        timestamp: datetime = datetime.now(),
    ) -> None:
        self._scheduler.add_job(func, trigger="date", args=args, run_date=timestamp)
