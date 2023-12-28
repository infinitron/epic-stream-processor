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
from astropy.wcs import WCS
from sqlalchemy import insert
from sqlalchemy import select
from sqlalchemy import update
from sqlalchemy.sql.expression import bindparam

from .._utils import DynSources
from .._utils import PatchMan
from .._utils import get_lmn_grid
from ..epic_orm.pg_pixel_storage import EpicWatchdogTable
from ..epic_types import NDArrayBool_t
from ..epic_types import NDArrayNum_t
from ..epic_types import Patch_t
from ..epic_types import WatchMode_t
from .service_hub import ServiceHub


class WatchDog:
    """
    Monitors the locations of specified sources on EPIC images.
    """

    def __init__(self, service_hub: Optional[ServiceHub] = None):
        self._service_Hub: ServiceHub
        if service_hub is not None:
            self._service_Hub = service_hub

        self._staging_watch_df = pd.DataFrame()
        self._load_sources()
        self._watch_df = pd.DataFrame(
            columns=["id", "source_name", "ra", "dec", "patch_type"]
        )
        self._update_watch_df()
        self._service_Hub._scheduler.add_job(
            self._update_watch_df, "interval", minutes=1
        )
        print("Watch list")
        print(self._watch_df)

    def watch_source(
        self,
        id: int,
        name: str,
        ra: float,
        dec: float,
        t_start: datetime,
        t_end: Optional[datetime] = None,
        watch_mode: WatchMode_t = "continuous",
        patch_type: Patch_t = 5,
    ) -> None:
        if watch_mode == "continuous" and t_end is not None:
            t_end = datetime.utcnow() + timedelta(days=99 * 365.25)

        if watch_mode == "timed" and t_end is None:
            t_end = t_start + timedelta(days=7)

        new_row_df = pd.DataFrame([dict(
                id=id,
                source_name=name,
                ra=ra,
                dec=dec,
                patch_type=patch_type,
                t_start=t_start,
                t_end=t_end,
                watch_mode=watch_mode,
            )])
        

        self._staging_watch_df = pd.concat([self._staging_watch_df, new_row_df],
            ignore_index=True,
        )

        print(self._staging_watch_df)

        self._update_watch_df()
        # if watch_until is not None and watch_mode != "continuous":
        #     self._add_watch_timer(f"{id}", watch_until)

    def _update_watch_df(self) -> None:
        print("Updating watchdf")
        stag = self._staging_watch_df
        if len(stag.index)==0:
            return
        now = datetime.utcnow()

        watched_ids = stag[stag["t_end"] <= now]["id"].to_numpy()
        if watched_ids.shape[0] == 0:
            if (
                self._watch_df.shape[0] == 0
                and self._staging_watch_df.shape[0] > 0
            ):
                self._watch_df = self._staging_watch_df[
                    ["id", "source_name", "ra", "dec", "patch_type"]
                ]
            # return

        self._staging_watch_df = stag[(stag["t_end"] > now)]
        self._watch_df = self._staging_watch_df[
            self._staging_watch_df["t_start"] < now
        ][["id", "source_name", "ra", "dec", "patch_type"]]

        print(self._watch_df)

        upd_dicts = [
            dict(_id=int(i), watch_status="watched") for i in watched_ids
        ]
        if len(upd_dicts) == 0:
            return
        stmt = (
            update(EpicWatchdogTable)  # type: ignore[arg-type]
            .where(EpicWatchdogTable.id == bindparam("_id"))
            .values({"watch_status": bindparam("watch_status")})
        )
        self._service_Hub._pgdb._connection.execute(
            stmt, upd_dicts
        )  # type: ignore[no-untyped-call]
        self._service_Hub._pgdb._connection.commit()

    def add_voevent_and_watch(self, voevent: str) -> None:
        raise NotImplementedError(
            "External VOEvent handler not implemented yet"
        )

    def add_source_and_watch(
        self,
        source_name: str,
        ra: float,
        dec: float,
        event_time: str,
        t_start: str,
        t_end: str,
        author: str = "batman",
        reason: str = "Detection of FRBs",
        watch_mode: WatchMode_t = "continuous",
        patch_type: Patch_t = "3x3",
        event_type: str = "Manual trigger",
        voevent: str = "<?xml version='1.0'?><Empty></Empty>",
        **kwargs: str,
    ) -> None:
        # check if the name already exists
        stmt = select(EpicWatchdogTable).filter_by(  # type: ignore[arg-type, attr-defined]
            source=source_name
        )
        result = self._service_Hub._pgdb._connection.execute(
            stmt
        ).all()  # type: ignore[no-untyped-call]
        if len(result) > 0:
            raise Exception(f"{source_name} already exists in the watch list.")

        t_start_dt = datetime.fromisoformat(t_start) or datetime.utcnow()
        t_end_dt = datetime.fromisoformat(
            t_end
        ) or datetime.utcnow() + timedelta(days=7)
        stmt = (
            insert(EpicWatchdogTable)  # type: ignore[arg-type]
            .values(
                source=source_name,
                ra_deg=ra,
                dec_deg=dec,
                event_time=datetime.fromisoformat(event_time)
                or datetime.utcnow(),
                event_type=event_type,
                t_start=t_start_dt,
                t_end=t_end_dt,
                watch_mode=watch_mode,
                patch_type=patch_type,
                reason=reason,
                author=author,
                watch_status="watching",
                voevent=voevent,
            )
            .returning(EpicWatchdogTable.id)
        )
        print(stmt)
        result = self._service_Hub._pgdb._connection.execute(stmt).all()  # type: ignore[no-untyped-call]
        id = result[0][0]
        print(
            self._service_Hub._pgdb._connection.execute(
                select(EpicWatchdogTable)
            ).all()
        )
        print("ID:", id)
        self._service_Hub._pgdb._connection.commit()
        # self._service_Hub._pgdb._session.commit()
        # self._service_Hub._pgdb._session.flush()

        self.watch_source(
            id,
            source_name,
            ra,
            dec,
            t_start_dt,
            t_end_dt,
            watch_mode,
            patch_type,
        )

    def _load_sources(self, overwrite: bool = False) -> None:
        self._staging_watch_df = self._service_Hub._pgdb.list_watch_sources()

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

        self.inttime = self.primary_hdr["INTTIM"]

        self.t_obs = self.img_hdr["DATETIME"]

        self.wcs = WCS(self.img_hdr, naxis=2)

        self.max_rad = self.xdim * 0.5 * np.cos(np.deg2rad(elevation_limit))

        self.filename = self.img_hdr["FILENAME"]

    def ra2x(
        self, ra: Union[float, NDArrayNum_t]
    ) -> Union[float, NDArrayNum_t]:
        """Return the X-pixel (0-based index) number given an RA"""
        pix = (ra - self.ra0) / self.dx + self.x0
        return self.nearest_pix(pix) - 1

    def nearest_pix(
        self, pix: Union[float, NDArrayNum_t]
    ) -> Union[float, NDArrayNum_t]:
        frac_dist = np.minimum(np.modf(pix)[0], 0.5)
        nearest_pix: Union[float, NDArrayNum_t] = np.floor(pix + frac_dist)
        return nearest_pix

    def dec2y(
        self, dec: Union[float, NDArrayNum_t]
    ) -> Union[float, NDArrayNum_t]:
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
                    [
                        self.ra2x(ra) - self.xdim / 2,
                        self.dec2y(dec) - self.ydim / 2,
                    ]
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
            np.linalg.norm(
                np.vstack([x - self.xdim / 2, y - self.ydim / 2]), axis=0
            ),
            self.max_rad,
        )
        return is_fov

    def header_to_metadict(self, source_names: List[str]) -> Dict[str, Any]:
        ihdr = self.img_hdr
        return dict(
            id=[str(uuid4())],
            img_time=[
                datetime.strptime(ihdr["DATETIME"], "%Y-%m-%dT%H:%M:%S.%f")
            ],
            n_chan=[int(ihdr["NAXIS3"])],
            n_pol=[int(ihdr["NAXIS4"])],
            chan0=[ihdr["CRVAL3"] - ihdr["CDELT3"] * ihdr["CRPIX3"]],
            chan_bw=[ihdr["CDELT3"]],
            epic_version=[self.epic_ver],
            img_size=[str((ihdr["NAXIS1"], ihdr["NAXIS2"]))],
            int_time=self.inttime,
            filename=self.filename,
            source_names=[source_names.tolist()],
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
            self._watch_df["patch_type"]
            .str.split("x")
            .str[0]
            .astype(float)
            .to_numpy()
        )
        self.patch_npix_l = self.patch_size_l**2

        self._update_src_skypos(self.t_obs)

        self.x_l, self.y_l = self.wcs.all_world2pix(
            self.ra_l, self.dec_l, 1
        )  # 1-based index
        self.x_l, self.y_l = self.nearest_pix(self.x_l), self.nearest_pix(
            self.y_l
        )
        self.in_fov = np.logical_and((self.x_l >= 0), (self.y_l >= 0))

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

        # update the pixel indices
        self.watch_l[3, :] += xpatch_pix_idx
        self.watch_l[4, :] += ypatch_pix_idx

        # remove sources if they cross the fov
        self.watch_l[1, :], self.watch_l[2, :] = self.wcs.all_pix2world(
            self.watch_l[3, :], self.watch_l[4, :], 1
        )

        self.watch_l[-2, :] = np.logical_not(
            np.isnan(self.watch_l[1, :]) | np.isnan(self.watch_l[2, :])
        ) & self.is_pix_fov(self.watch_l[3, :], self.watch_l[4, :])

        # test fov crossing
        groups = np.split(
            self.watch_l[-2, :],
            np.unique(self.watch_l[0, :], return_index=True)[1][1:],
        )

        is_out_fov = [
            i if np.all(i) else np.logical_and(i, False) for i in groups
        ]

        # filter patches crossing fov
        is_out_fov = np.concatenate(is_out_fov).astype(bool).ravel()
        self.watch_l = self.watch_l[:, is_out_fov]
        xpatch_pix_idx = xpatch_pix_idx[is_out_fov]
        ypatch_pix_idx = ypatch_pix_idx[is_out_fov]

        # extract the pixel values for each pixel
        # img_array indices [complex, npol, nchan, y, x]
        pix_values = self.img_array[
            :,
            :,
            :,
            self.watch_l[4, :].astype(int)
            - 1,  # convert 1-based to 0-based for np indexing
            self.watch_l[3, :].astype(int) - 1,
        ]
        pix_values_l = [
            pix_values[:, :, :, i].ravel().tolist()
            for i in range(pix_values.shape[-1])
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
        source_names = self.src_l[self.watch_l[0, :].astype(int)]

        self.pixel_meta_df = pd.DataFrame.from_dict(
            self.header_to_metadict(source_names=np.unique(source_names))
        )

        self.pixel_idx_df = pd.DataFrame.from_dict(
            dict(
                id=[
                    self.pixel_meta_df.iloc[0]["id"]
                    for i in range(len(l_vals))
                ],
                pixel_coord=pix_coord_fmt,
                pixel_values=pix_values_l,
                pixel_skypos=skypos_pg_fmt,
                source_names=source_names,
                pixel_lm=lm_coord_fmt,
                pix_ofst_x=xpatch_pix_idx,
                pix_ofst_y=ypatch_pix_idx,
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
