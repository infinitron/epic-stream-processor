from numbers import Number
from typing import Iterator, Callable
import psycopg
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta
from pytz import utc
from streamz import Stream
from sqlalchemy import create_engine
import pandas as pd
from geoalchemy2 import Geometry


class ServiceHub(object):
    _pg_conn = None
    _scheduler = BackgroundScheduler(timezone=utc)
    _pipeline = Stream()
    _pg_engine = create_engine(
        'postgresql:///postgres?host=/var/run/postgresql')

    def __init__(self) -> None:
        """connect to ectd and and also have a scheduler from apscheduler"""
        self._scheduler.start()
        self._connect_pgdb()

        # define the stream processor timed_window(5)
        self._pipeline.map(ServiceHub.detect_transient)\
            .timed_window(1).sink(ServiceHub.insert_pgdb)

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(ServiceHub, cls).__new__(cls)
        return cls.instance

    def _connect_pgdb(self) -> None:
        # establish postgres connection
        # TODO: fetch the connection details from sys config service
        pg_conn_params = dict(
            dbname="postgres", user="batman")
        try:
            self._pg_conn = psycopg.connect(**pg_conn_params)
            self._pg_conn.autocommit = True
            print('Postgres connection established')
        except Exception as e:
            print(e)
            print('Retrying connection in 60 seconds')
            self.schedule_job_delay(self._connect_pgdb, None, 60)

    def insert_into_db(self, pixel_df: pd.DataFrame,
                       pixel_meta_df: pd.DataFrame) -> None:
        self._pipeline.emit(dict(pixels=pixel_df, meta=pixel_meta_df))

    @staticmethod
    def detect_transient(upstream: object) -> object:
        return upstream

    @staticmethod
    def insert_pgdb(upstream: list) -> None:
        pixels_df = []
        metadata_df = []

        if len(upstream) < 1:
            return

        for i in upstream:
            pixels_df.append(i['pixels'])
            metadata_df.append(i['meta'])

        pixels_df = pd.concat(pixels_df)
        metadata_df = pd.concat(metadata_df)

        pixels_df.to_sql('epic_pixels', con=ServiceHub._pg_engine,
                         dtype=dict(
                             pixel_skypos=Geometry(
                                 geometry_type='POINT', srid=4326)
                         ), if_exists='append', index=False)

        metadata_df.to_sql('epic_img_metadata',
                           con=ServiceHub._pg_engine,
                           if_exists='append', index=False)

    def pg_query(self, query: str, args: tuple = None) -> Iterator:
        if self._pg_conn is None:
            raise (ConnectionRefusedError(
                "Could not establish connection to the pgdb"))
        else:
            result = self._pg_conn.execute(query, args).fetchall()
            # self._pg_conn.commit()
            return result

    def schedule_job_delay(self, func: Callable, args: tuple = None,
                           seconds: Number = 604800) -> None:
        self._scheduler.add_job(
            func, trigger='date', args=args,
            run_date=datetime.now()+timedelta(seconds))

    def schedule_job_date(self, func: Callable, args: tuple = None,
                          timestamp: datetime = datetime.now()) -> None:
        self._scheduler.add_job(
            func, trigger='date', args=args,
            run_date=timestamp)
