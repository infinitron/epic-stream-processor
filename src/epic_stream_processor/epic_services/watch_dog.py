from datetime import datetime
from datetime import timedelta
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
from uuid import uuid4

import numpy as np
import pandas as pd
from astropy.io.fits import Header
from sqlalchemy import insert, update, select
from sqlalchemy.sql.expression import bindparam

from .._utils import DynSources
from .._utils import PatchMan
from .._utils import get_lmn_grid
from ..epic_types import NDArrayBool_t
from ..epic_types import NDArrayNum_t
from ..epic_types import Patch_t
from ..epic_types import WatchMode_t
from .service_hub import ServiceHub
from ..epic_orm.pg_pixel_storage import EpicWatchdogTable


class WatchDog:
    """
    Monitors the locations of specified sources on EPIC images.
    """

    def __init__(self, serviceHub: Optional[ServiceHub] = None):
        self._service_Hub: ServiceHub
        if serviceHub is not None:
            self._service_Hub = serviceHub

        self._load_sources()
        self._watch_df = pd.DataFrame(
            columns=["id", "source_name", "ra", "dec", "patch_type"]
        )
        self._update_watch_df()
        self._service_Hub._scheduler.add_job(
            self._update_watch_df, "interval", minutes=5
        )
        # self._staging_watch_df = pd.DataFrame(
        #     columns=["id", "source_name", "ra", "dec", "patch_type", "t_start", "t_end"]
        # )
        # self._watch_df = pd.concat(
        #     [
        #         self._watch_df,
        #         pd.DataFrame(
        #             dict(
        #                 id=[1, 2, 3, 4, 5],
        #                 source_name=["jupiter", "c2", "sun", "Cyg A", "Cyg X-1"],
        #                 ra=[130.551, 234.1, 0.0, 299.86815191, 299.59031556],
        #                 dec=[34.348, 34.348, 0.0, 40.73391574, 35.20160681],
        #                 patch_type=["3x3", "3x3", "5x5", "3x3", "3x3"],
        #             )
        #         ),
        #     ]
        # )
        print("Watch list")
        print(self._watch_df)

    def watch_source(
        self,
        id: int,
        name: str,
        ra: float,
        dec: float,
        t_start: datetime,
        t_end: datetime = None,
        watch_mode: WatchMode_t = "continuous",
        patch_type: Patch_t = "3x3",
    ) -> None:

        if watch_mode == "continuous":
            t_end = datetime.utcnow() + timedelta(days=99 * 365.25)

        self._staging_watch_df = self._staging_watch_df.append(
            dict(
                id=id,
                source_name=name,
                ra=ra,
                dec=dec,
                patch_type=patch_type,
                t_start=t_start,
                t_end=t_end,
                watch_mode=watch_mode,
            ),
            ignore_index=True,
        )

        self._update_watch_df()
        # if watch_until is not None and watch_mode != "continuous":
        #     self._add_watch_timer(f"{id}", watch_until)

    def _update_watch_df(self):
        stag = self._staging_watch_df
        now = datetime.utcnow()

        watched_ids = stag[stag["t_end"] <= now]["id"].to_numpy()
        if watched_ids.shape[0] == 0:
            if self._watch_df.shape[0] == 0 and self._staging_watch_df.shape[0] > 0:
                self._watch_df = self._staging_watch_df[
                    ["id", "source_name", "ra", "dec", "patch_type"]
                ]
            return

        self._staging_watch_df = stag[(stag["t_end"] > now)]
        self._watch_df = self._staging_watch_df[
            self._staging_watch_df["t_start"] < now
        ][["id", "source_name", "ra", "dec", "patch_type"]]

        upd_dicts = [dict(_id=int(i), watch_status="watched") for i in watched_ids]
        stmt = (
            update(EpicWatchdogTable)
            .where(EpicWatchdogTable.id == bindparam("_id"))
            .values({"watch_status": bindparam("watch_status")})
        )
        self._service_Hub._pgdb._connection.execute(stmt, upd_dicts)

    def add_voevent_and_watch(self, voevent: str) -> None:
        raise NotImplementedError("External VOEvent handler not implemented yet")

    def add_source_and_watch(
        self,
        source_name: str,
        ra: float,
        dec: float,
        event_time: Optional[str],
        t_start: Optional[str],
        t_end: Optional[str],
        author: str = "batman",
        reason: str = "Detection of FRBs",
        watch_mode: WatchMode_t = "continuous",
        patch_type: Patch_t = "3x3",
        event_type: str = "Manual trigger",
        voevent: str = "<?xml version='1.0'?><Empty></Empty>",
        **kwargs,
    ) -> None:
        # check if the name already exists
        stmt = select(EpicWatchdogTable).filter_by(source=source_name)
        result = self._service_Hub._pgdb._connection.execute(stmt).all()
        if len(result) > 0:
            raise Exception(f"{source_name} already exists in the watch list.")

        t_start = datetime.fromisoformat(t_start) or datetime.utcnow()
        t_end = datetime.fromisoformat(t_end) or datetime.utcnow() + timedelta(days=7)
        stmt = (
            insert(EpicWatchdogTable)
            .values(
                source=source_name,
                event_skypos=f"SRID=4326; POINT({ra} {dec})",
                event_time=datetime.fromisoformat(event_time) or datetime.utcnow(),
                event_type=event_type,
                t_start=t_start,
                t_end=t_end,
                watch_mode=watch_mode,
                patch_type=patch_type,
                reason=reason,
                author=author,
                watch_status="watching",
                voevent=voevent,
            )
            .returning(EpicWatchdogTable.id)
        )

        result = self._service_Hub._pgdb._connection.execute(stmt).all()
        id = result[0][0]

        self.watch_source(
            id, source_name, ra, dec, t_start, t_end, watch_mode, patch_type
        )

        # id = Column(Integer, primary_key=True)  # serial type
        # source = Column(Text, nullable=False)
        # event_skypos = Column(Geometry("POINT", srid=4326), nullable=False)
        # event_time = Column(TIMESTAMP, nullable=False)
        # event_type = Column(Text, nullable=False)
        # t_start = Column(TIMESTAMP, server_default=func.now())
        # t_end = Column(TIMESTAMP, nullable=False)
        # watch_mode = Column(Text, nullable=False)
        # patch_type = Column(Text, nullable=False)
        # reason = Column(Text, nullable=False)
        # author = Column(Text, nullable=False)
        # watch_status = Column(Text, nullable=False, server_default="watching")
        # voevent = Column(XMLType, nullable=False)
        # '<?xml version="1.0"?><book><title>Manual</title><chapter>...</chapter></book>'
        # = datetime.now() + timedelta(604800)

        # raise NotImplementedError("Manual source watching is not Implemented yet")

    # def get_list(self) -> pd.DataFrame:
    #     return self._watch_df[["source_name", "skycoord", "patch_type"]]

    # def _flatten_series(self, series: pd.Series) -> List[T]:
    #     return list(chain.from_iterable(series))

    # def _remove_outside_sky_sources(
    #     self, df: pd.DataFrame, pos_column: str
    # ) -> pd.DataFrame:
    #     return df[~(df[pos_column].apply(lambda x: np.isnan(x).any()))]

    # def _remove_outside_sky_patches(
    #     self,
    #     df: pd.DataFrame,
    #     src_column: str,
    #     pos_column: str,
    # ) -> pd.DataFrame:
    #     outside_sources = df[(df[pos_column].apply(lambda x: np.isnan(x).any()))][
    #         src_column
    #     ].unique()

    #     return df[~(df[src_column].isin(outside_sources))]

    # def _get_dyn_skypos(
    #     self, src_name: str, t_obs_str: str, ra: float, dec: float
    # ) -> List[float]:
    #     dyn_fun = f"get_skypos_{src_name}"
    #     # if hasattr(DynSources, dyn_fun):
    #     #     ra_dec: List[float] = getattr(DynSources, dyn_fun).__call__(t_obs_str)
    #     #     return ra_dec
    #     # else:
    #     #     return [ra, dec]
    #     try:
    #         ra_dec: List[float] = getattr(DynSources, dyn_fun).__call__(t_obs_str)
    #         return ra_dec
    #     except Exception:
    #         return [ra, dec]

    # def _update_src_skypos(
    #     self,
    #     source_list: pd.DataFrame,
    #     t_obs_str: str,
    #     ra_col: str = "ra",
    #     dec_col: str = "dec",
    # ) -> pd.DataFrame:
    #     # return source_list

    #     # print("IN UPD", source_list)

    #     source_list[[ra_col, dec_col]] = source_list.apply(
    #         lambda row: pd.Series(
    #             self._get_dyn_skypos(
    #                 row["source_name"], t_obs_str, row["ra"], row["dec"]
    #             ),
    #             index=[ra_col, dec_col],
    #         ),
    #         axis=1,
    #     )

    # return source_list

    # def get_watch_indices(
    #     self,
    #     header_str: str,
    #     img_axes: List[int] = [1, 2],
    # ) -> pd.DataFrame:
    #     header = Header.fromstring(header_str)
    #     wcs = WCS(header, naxis=img_axes)
    #     sources = self._watch_df["source_name"]
    #     patch_types = self._watch_df["patch_type"]

    #     t_obs = header["DATETIME"]

    #     # for souces with changing ra and dec like the Sun
    #     start = timer()
    #     src_pos = self._update_src_skypos(self._watch_df, t_obs)
    #     print(f"Elapsed UPD: {timer()-start} s")
    #     # print("UPD", src_pos)

    #     # drop any sources outside the sky
    #     pixels = wcs.all_world2pix(src_pos[["ra", "dec"]].to_numpy(), 1)

    #     source_pixel_df = pd.DataFrame.from_dict(
    #         dict(pixel=pixels.tolist(), source=sources, patch_type=patch_types)
    #     )

    #     source_pixel_df = self._remove_outside_sky_sources(source_pixel_df, "pixel")

    #     # generate pixel patches for each source
    #     # each patch cell in the df contains a list of patch pixels (x or y)
    #     source_pixel_df[["xpatch", "ypatch"]] = source_pixel_df.apply(
    #         lambda x: pd.Series(
    #             [
    #                 PatchMan.get_patch_pixels(x["pixel"], patch_type=x["patch_type"])[
    #                     0
    #                 ].tolist(),
    #                 PatchMan.get_patch_pixels(x["pixel"], patch_type=x["patch_type"])[
    #                     1
    #                 ].tolist(),
    #             ],
    #             index=["xpatch", "ypatch"],
    #         ),
    #         axis=1,
    #     )

    #     source_pixel_df["patch_name"] = source_pixel_df.apply(
    #         lambda x: [x["source"]] * len(x["xpatch"]), axis=1
    #     )

    #     # expand pixel patches into individual rows
    #     source_patch_df = pd.DataFrame(
    #         dict(
    #             xpatch=self._flatten_series(source_pixel_df["xpatch"]),
    #             ypatch=self._flatten_series(source_pixel_df["ypatch"]),
    #             patch_name=self._flatten_series(source_pixel_df["patch_name"]),
    #         )
    #     )

    #     # filter out sources whose patches fall outside the sky
    #     # at least partially
    #     source_patch_df["patch_pixels"] = source_patch_df.apply(
    #         lambda x: tuple([round(x["xpatch"]), round(x["ypatch"])]), axis=1
    #     )

    #     source_patch_df["patch_skypos"] = wcs.all_pix2world(
    #         np.stack(source_patch_df["patch_pixels"].tolist(), axis=0), 1
    #     ).tolist()

    #     source_patch_df = self._remove_outside_sky_patches(
    #         source_patch_df,
    #         "patch_name",
    #         "patch_skypos",
    #     )

    #     # aggregate common pixels
    #     source_patch_df = source_patch_df.groupby(["patch_pixels"]).agg(
    #         patch_name=("patch_name", list), patch_skypos=("patch_skypos", "first")
    #     )

    #     source_patch_df.reset_index()
    #     source_patch_df["patch_pixels"] = source_patch_df.index.get_level_values(0)

    #     # columns patch_name <list of sources>, patch_pixel, patch_skypos
    #     return source_patch_df

    # @staticmethod
    # def header_to_metadict(image_hdr: str, epic_version: str) -> Dict[str, Any]:
    #     ihdr = Header.fromstring(image_hdr)
    #     return dict(
    #         id=[str(uuid4())],
    #         img_time=[datetime.strptime(ihdr["DATETIME"], "%Y-%m-%dT%H:%M:%S.%f")],
    #         n_chan=[int(ihdr["NAXIS3"])],
    #         n_pol=[int(ihdr["NAXIS4"])],
    #         chan0=[ihdr["CRVAL3"] - ihdr["CDELT3"] * ihdr["CRPIX3"]],
    #         chan_bw=[ihdr["CDELT3"]],
    #         epic_version=[epic_version],
    #         img_size=[str((ihdr["NAXIS1"], ihdr["NAXIS2"]))],
    #     )

    # @staticmethod
    # def insert_lm_coords_df(
    #     df: pd.DataFrame,
    #     xsize: int,
    #     ysize: int,
    #     pixel_idx_col: str,
    #     lm_coord_col: str,
    # ) -> pd.DataFrame:
    #     lmn_grid = get_lmn_grid(xsize, ysize)
    #     df[lm_coord_col] = df[pixel_idx_col].apply(
    #         lambda x: str(
    #             (lmn_grid[0, x[0] - 1, x[1] - 1], lmn_grid[1, x[0] - 1, x[1] - 1])
    #         )
    #     )
    #     return df

    # @staticmethod
    # def insert_pixels_df(
    #     df: pd.DataFrame,
    #     pixels: npt.NDArray[np.float64],
    #     pixel_idx_col: str = "patch_pixels",
    #     val_col: str = "pixel_values",
    # ) -> pd.DataFrame:
    #     df[val_col] = df[pixel_idx_col].apply(
    #         lambda x: pixels[:, :, :, x[1] - 1, x[0] - 1].ravel().tolist()
    #     )
    #     return df

    # @staticmethod
    # def format_skypos_pg(
    #     df: pd.DataFrame,
    #     skypos_col: str = "patch_skypos",
    #     skypos_fmt_col: str = "pixel_skypos",
    # ) -> pd.DataFrame:
    #     df[skypos_fmt_col] = df[skypos_col].apply(
    #         lambda x: f"SRID=4326;POINT({x[0]} {x[1]})"
    #     )

    #     return df

    # def filter_and_store_imgdata(
    #     self,
    #     header: str,
    #     img_array: npt.NDArray[np.float64],
    #     epic_version: str = "0.0.2",
    # ) -> None:  # Tuple[pd.DataFrame, pd.DataFrame]:

    #     start = timer()
    #     pixel_idx_df = self.get_watch_indices(header)
    #     print(f"Elapsed2a: {timer()-start}")
    #     start = timer()
    #     pixel_meta_df = pd.DataFrame.from_dict(
    #         self.header_to_metadict(header, epic_version=epic_version)
    #     )
    #     print(f"Elapsed2b: {timer()-start}")
    #     pixel_idx_df["id"] = pixel_meta_df.iloc[0]["id"]

    #     pixel_idx_df = self.format_skypos_pg(
    #         pixel_idx_df, "patch_skypos", "pixel_skypos"
    #     )

    #     xsize, ysize = img_array.shape[4], img_array.shape[3]

    #     pixel_idx_df = self.insert_lm_coords_df(
    #         pixel_idx_df, xsize, ysize, "patch_pixels", "pixel_lm"
    #     )

    #     pixel_idx_df["pixel_coord"] = pixel_idx_df["patch_pixels"].astype(str)
    #     pixel_idx_df["source_names"] = pixel_idx_df["patch_name"]

    #     pixel_idx_df = self.insert_pixels_df(
    #         pixel_idx_df,
    #         img_array,
    #         pixel_idx_col="patch_pixels",
    #         val_col="pixel_values",
    #     )

    #     # pixel_idx_df = pixel_idx_df[
    #     #     [
    #     #         "id",
    #     #         "pixel_values",
    #     #         "pixel_coord",
    #     #         "pixel_skypos",
    #     #         "source_names",
    #     #         "pixel_lm",
    #     #     ]
    #     # ]

    #     # print(pixel_idx_df, pixel_meta_df)

    #     self._service_Hub.insert_single_epoch_pgdb(
    #         pixel_idx_df[
    #             [
    #                 "id",
    #                 "pixel_values",
    #                 "pixel_coord",
    #                 "pixel_skypos",
    #                 "source_names",
    #                 "pixel_lm",
    #             ]
    #         ],
    #         pixel_meta_df,
    #     )

    #     # return 0,0#pixel_idx_df, pixel_meta_df

    # def _remove_source(self, id: str) -> None:
    #     self._watch_df.drop(
    #         (self._watch_df[self._watch_df["id"] == id]).index, inplace=True
    #     )
    #     self._service_Hub._pgdb.set_src_watched(self._watch_df["id"])
    #     # .pg_query(
    #     #     "UPDATE epic_watchdog SET watch_status=%s", tuple("watched")
    #     # )

    # def _add_watch_timer(self, id: str, date_watch: datetime) -> None:
    #     self._service_Hub.schedule_job_date(
    #         self._remove_source, args=tuple(id), timestamp=date_watch
    #     )

    def _load_sources(self, overwrite: bool = False) -> None:
        # if self._watch_df.shape[0] > 0 and overwrite is False:
        #     raise Exception(
        #         "Sources are already being watched and overwrite is \
        #             set to false"
        #     )

        self._staging_watch_df = self._service_Hub._pgdb.list_watch_sources()
        # else:
        #     for source in self._watch_df:
        #         source["job"].remove()

        # sources = self._service_Hub.pg_query(
        #     "SELECT id,source,ST_X(skypos),ST_Y(skypos),t_end, watch_mode,\
        #          patch_type from \
        #          epic_watchdog where watch_status=%s",
        #     tuple("watching"),
        # )

        # self._watch_df = self._watch_df.head(0)
        # for source in sources:
        #     self.watch_source(
        #         id=source[0],
        #         name=source[1],
        #         ra=source[2],
        #         dec=source[3],
        #         watch_until=source[4],
        #         watch_mode=source[5],
        #         patch_type=source[6],
        #     )

    def change_storage_servicer(self, servicer: ServiceHub) -> None:
        self._service_Hub = servicer


class EpicPixels:
    def __init__(
        self,
        img_hdr: str,
        primary_hdr: str,
        img_array: NDArrayNum_t,
        watch_df: pd.DataFrame,
        epic_ver: str = "0.0.1",
        img_axes: List[int] = [1, 2],
        elevation_limit: float = 0.0,
    ) -> None:
        self.img_array = img_array
        # self.header_str = header
        self.epic_ver = epic_ver

        self._watch_df = watch_df
        self.img_hdr = Header.fromstring(img_hdr)
        self.primary_hdr = Header.fromstring(primary_hdr)

        self.ra0 = self.img_hdr["CRVAL1"]
        self.dec0 = self.img_hdr["CRVAL2"]

        self.x0 = self.img_hdr["CRPIX1"]
        self.y0 = self.img_hdr["CRPIX2"]

        self.dx = self.img_hdr["CDELT1"]
        self.dy = self.img_hdr["CDELT2"]

        self.delta_min = elevation_limit

        self.xdim = self.primary_hdr["GRIDDIMX"]
        self.ydim = self.primary_hdr["GRIDDIMY"]

        self.dgridx = self.primary_hdr["DGRIDX"]
        self.dgrixy = self.primary_hdr["DGRIDY"]

        self.t_obs = self.img_hdr["DATETIME"]

        self.max_rad = self.xdim * self.dgridx * np.cos(np.deg2rad(elevation_limit))

    def ra2x(self, ra: Union[float, NDArrayNum_t]) -> Union[float, NDArrayNum_t]:
        """Return the X-pixel (0-based index) number given an RA"""
        pix = (ra - self.ra0) / self.dx + self.x0
        return self.nearest_pix(pix) - 1

    def nearest_pix(
        self, pix: Union[float, NDArrayNum_t]
    ) -> Union[float, NDArrayNum_t]:
        frac_dist = np.minimum(np.modf(pix)[0], 0.5)
        nearest_pix: Union[float, NDArrayNum_t] = np.floor(pix + frac_dist)
        return nearest_pix

    def dec2y(self, dec: Union[float, NDArrayNum_t]) -> Union[float, NDArrayNum_t]:
        """Return the Y-pixel (0-based index) number given a DEC"""
        pix = (dec - self.dec0) / self.dy + self.y0
        return self.nearest_pix(pix) - 1

    def is_skycoord_fov(
        self,
        ra: Union[float, NDArrayNum_t, pd.Series],
        dec: Union[float, NDArrayNum_t, pd.Series],
    ) -> NDArrayBool_t:
        """Return a bool index indicating whether the
        specified sky coordinates lie inside the fov
        """
        is_fov: NDArrayBool_t = np.less_equal(
            np.linalg.norm(
                np.vstack(
                    [self.ra2x(ra) - self.xdim / 2, self.dec2y(dec) - self.ydim / 2]
                ),
                axis=0,
            ),
            self.max_rad,
        )
        return is_fov

    def is_pix_fov(
        self,
        x: Union[float, NDArrayNum_t, pd.Series],
        y: Union[float, NDArrayNum_t, pd.Series],
    ) -> NDArrayBool_t:
        """Return a bool index indicating whether the
        specified pixel coordinates lie inside the fov
        """
        # print(np.vstack([x - self.xdim/2,y - self.ydim/2]))
        # print(np.linalg.norm(np.vstack([x - self.xdim/2,y - self.ydim/2]),axis=0))
        is_fov: NDArrayBool_t = np.less_equal(
            np.linalg.norm(np.vstack([x - self.xdim / 2, y - self.ydim / 2]), axis=0),
            self.max_rad,
        )
        return is_fov

    def header_to_metadict(self) -> Dict[str, Any]:
        ihdr = self.img_hdr
        return dict(
            id=[str(uuid4())],
            img_time=[datetime.strptime(ihdr["DATETIME"], "%Y-%m-%dT%H:%M:%S.%f")],
            n_chan=[int(ihdr["NAXIS3"])],
            n_pol=[int(ihdr["NAXIS4"])],
            chan0=[ihdr["CRVAL3"] - ihdr["CDELT3"] * ihdr["CRPIX3"]],
            chan_bw=[ihdr["CDELT3"]],
            epic_version=[self.epic_ver],
            img_size=[str((ihdr["NAXIS1"], ihdr["NAXIS2"]))],
        )

    def store_pg(self, s_hub: ServiceHub) -> None:
        if self.pixel_idx_df is None or self.pixel_meta_df is None:
            # no sources in the fov to update
            return
        s_hub.insert_single_epoch_pgdb(self.pixel_idx_df, self.pixel_meta_df)

    def gen_pixdata_dfs(
        self,
    ) -> pd.DataFrame:
        self.idx_l = self._watch_df.index.to_numpy()
        self.src_l = self._watch_df["source_name"].to_numpy().astype(str)
        self.ra_l = self._watch_df["ra"].to_numpy()
        self.dec_l = self._watch_df["dec"].to_numpy()
        self.patch_size_l = (
            self._watch_df["patch_type"].str.split("x").str[0].astype(float).to_numpy()
        )
        self.patch_npix_l = self.patch_size_l**2

        # for souces with changing ra and dec like the Sun
        # start = timer()
        self._update_src_skypos(self.t_obs)
        # print(f'Elapsed UPD: {timer()-start} s',self._watch_df.shape)

        # drop any sources outside the sky
        self.x_l = self.ra2x(self.ra_l)
        self.y_l = self.dec2y(self.dec_l)
        self.in_fov = self.is_pix_fov(self.x_l, self.y_l)

        # filter the indices or sources outside the sky
        self.watch_l = np.vstack(
            [
                self.idx_l,
                self.ra_l,
                self.dec_l,
                self.x_l,
                self.y_l,
                self.in_fov,
                self.patch_size_l,
            ]
        )

        if not self.in_fov.any():
            # no sources within the FOV
            self.pixel_idx_df = None
            self.pixel_meta_df = None
            return
        self.watch_l = self.watch_l[:, self.in_fov]

        xpatch_pix_idx, ypatch_pix_idx = np.hstack(
            list(map(PatchMan.get_patch_idx, self.watch_l[-1, :]))
        )

        self.watch_l = np.repeat(
            self.watch_l, self.watch_l[-1, :].astype(int) ** 2, axis=1
        )

        # update the pixel indices and remove sources if they cross the fov
        self.watch_l[3, :] += xpatch_pix_idx
        self.watch_l[4, :] += ypatch_pix_idx

        self.watch_l[-2, :] = self.is_pix_fov(self.watch_l[3, :], self.watch_l[4, :])

        # test fov crossing
        groups = np.split(
            self.watch_l[-2, :], np.unique(self.watch_l[0, :], return_index=True)[1][1:]
        )

        is_out_fov = [i if np.all(i) else np.logical_and(i, False) for i in groups]

        # filter patches crossing fov
        self.watch_l = self.watch_l[:, np.concatenate(is_out_fov).astype(bool).ravel()]

        # update the ra, dec
        self.watch_l[1, :] += xpatch_pix_idx * self.dx
        self.watch_l[2, :] += ypatch_pix_idx * self.dy

        # extract the pixel values for each pixel
        # img_array indices [complex, npol, nchan, y, x]
        pix_values = self.img_array[
            :, :, :, self.watch_l[4, :].astype(int), self.watch_l[3, :].astype(int)
        ]
        pix_values_l = [
            pix_values[:, :, :, i].ravel().tolist() for i in range(pix_values.shape[-1])
        ]

        skypos_pg_fmt = [
            f"SRID=4326;POINT({i} {j})"
            for i, j in zip(self.watch_l[1, :], self.watch_l[2, :])
        ]

        # grab lm coords
        lmn_grid = get_lmn_grid(self.xdim, self.ydim)
        l_vals = lmn_grid[
            0, self.watch_l[3, :].astype(int), self.watch_l[4, :].astype(int)
        ]
        m_vals = lmn_grid[
            1, self.watch_l[3, :].astype(int), self.watch_l[4, :].astype(int)
        ]

        lm_coord_fmt = [f"({i},{j})" for i, j in zip(l_vals, m_vals)]
        pix_coord_fmt = [
            f"({i},{j})"
            for i, j in zip(
                self.watch_l[3, :].astype(int), self.watch_l[4, :].astype(int)
            )
        ]
        # skycoord_fmt = [f"[{i},{j}]" for i,j in zip(self.watch_l[1,:].astype(int),self.watch_l[2,:].astype(int))]
        source_names = self.src_l[self.watch_l[0, :].astype(int)]

        self.pixel_meta_df = pd.DataFrame.from_dict(self.header_to_metadict())

        self.pixel_idx_df = pd.DataFrame.from_dict(
            dict(
                id=[self.pixel_meta_df.iloc[0]["id"] for i in range(len(l_vals))],
                pixel_values=pix_values_l,
                pixel_coord=pix_coord_fmt,
                pixel_skypos=skypos_pg_fmt,
                source_names=source_names,
                pixel_lm=lm_coord_fmt,
            )
        )

    def _update_src_skypos(
        self,
        # source_list: pd.DataFrame,
        t_obs_str: str,
    ) -> None:

        for i, src in enumerate(self.src_l):
            if src in DynSources.bodies:
                self.ra_l[i], self.dec_l[i] = DynSources.get_lwasv_skypos(
                    src, t_obs_str
                )
