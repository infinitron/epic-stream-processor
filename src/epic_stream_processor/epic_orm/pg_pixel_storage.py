from typing import Optional

import pandas as pd
#from geoalchemy2 import Geometry
from sqlalchemy import ARRAY
from sqlalchemy import TIMESTAMP
from sqlalchemy import Column
from sqlalchemy import Float
from sqlalchemy import Integer
from sqlalchemy import Text
from sqlalchemy import create_engine, text
from sqlalchemy import func
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import UUID, ARRAY as pg_ARRAY
from sqlalchemy.engine import Engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session

from .pg_types import PgPointType
from .pg_types import XMLType


_default_pg_conn_str = "postgresql://"
Base = declarative_base()


class EpicPixelsTable(Base):
    __tablename__ = "epic_pixels"

    counter = Column(Integer, primary_key=True)
    id = Column(UUID, nullable=False)
    pixel_values = Column(ARRAY(Float), nullable=False)
    pixel_coord = Column(PgPointType, nullable=False)
    pixel_lm = Column(PgPointType, nullable=False)
    source_names = Column(Text, nullable=False)
    #pixel_skypos = Column(Geometry(geometry_type="POINT", srid=4326))
    pix_ofst_x = Column(Integer, nullable=False)
    pix_ofst_y = Column(Integer, nullable=False)


class EpicImgMetadataTable(Base):
    __tablename__ = "epic_img_metadata"

    id = Column(UUID, primary_key=True, nullable=False, index=True)
    img_time = Column(TIMESTAMP, nullable=False, index=True)
    n_chan = Column(Integer, nullable=False)
    n_pol = Column(Integer, nullable=False)
    chan0 = Column(Float, nullable=False)
    chan_bw = Column(Float, nullable=False)
    epic_version = Column(Text, nullable=False)
    img_size = Column(PgPointType, nullable=False)
    int_time = Column(Float, nullable=False)
    # filename = Column(Text, nullable=False)
    source_names = Column(pg_ARRAY(Text), nullable=False)


class EpicWatchdogTable(Base):
    __tablename__ = "epic_watchdog"

    id = Column(Integer(), primary_key=True)  # serial type
    source = Column(Text())
    # event_skypos = Column(Geometry("POINT", srid=4326))
    ra_deg = Column(Float())
    dec_deg = Column(Float())
    event_time = Column(TIMESTAMP())
    event_type = Column(Text())
    t_start = Column(TIMESTAMP(), server_default=func.now())
    t_end = Column(TIMESTAMP())
    watch_mode = Column(Text())
    patch_type = Column(Integer())
    reason = Column(Text())
    author = Column(Text())
    watch_status = Column(Text(), server_default="watching")
    voevent = Column(XMLType())


class Database:
    def __init__(
        self, engine: Optional[Engine] = None, create_all_tables: bool = False
    ) -> None:
        self._engine = engine or create_engine(_default_pg_conn_str, pool_pre_ping=True)

        self._connection = self._engine.connect()
        self._session = Session(self._connection, autoflush=True)

        if create_all_tables:
            self.create_all_tables()
    
    def reconnect(self) -> None:
        self._connection.close()
        self._connection = self._engine.connect()

    def create_all_tables(self) -> None:
        Base.metadata.create_all(bind=self._connection, checkfirst=True)

    def set_src_watched(self, id: int) -> None:
        self._session.query(EpicWatchdogTable).filter(
            EpicWatchdogTable.id == id
        ).update({EpicWatchdogTable.watch_status: "watched"})

    def list_watch_sources(
        self,
    ) -> pd.DataFrame:
        watch_d = EpicWatchdogTable
        stmnt = select(
            # [
            watch_d.id,
            watch_d.source.label("source_name"),
            # func.ST_X(watch_d.event_skypos).label("ra"),
            # func.ST_Y(watch_d.event_skypos).label("dec"),
            watch_d.ra_deg.label("ra"),
            watch_d.dec_deg.label("dec"),
            watch_d.t_start,
            watch_d.t_end,
            watch_d.watch_mode,
            watch_d.patch_type,
            # ]
        ).where(watch_d.watch_status == "watching")
        # stmnt = select(EpicWatchdogTable).where(EpicWatchdogTable.watch_status == "watching")
        # print(stmnt)
        with self._engine.connect() as conn:
            watch_list = conn.execute(stmnt).all()
            conn.commit()

        return pd.DataFrame(watch_list)
        # return pd.read_sql(
        #     stmnt,
        #     self._engine,
        # )
