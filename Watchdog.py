from datetime import datetime, timedelta
import pandas as pd
from astropy.wcs import WCS
from astropy.io.fits import Header
from itertools import chain
from Utils import PatchMan
import numpy as np


class WatchDog(object):
    """
    Monitors the locations of specified sources on EPIC images.
    """

    def __init__(self, serviceHub):
        self._service_Hub = serviceHub
        self._watch_df = pd.DataFrame(
            columns=['id', 'source_name', 'ra', 'dec', 'patch_type'])
        self._watch_df = pd.concat(
            [self._watch_df,
             pd.DataFrame(dict(id=[1, 2], source_name=['center', 'c2'], ra=[
                 130.551, 234.1], dec=[34.348, 34.348],
                 patch_type=['3x3', '3x3']))])
        print('DF', self._watch_df)

    def watch_source(self, id, name, ra, dec, watch_until=None,
                     watch_mode='continuous', patch_type='3x3'):
        self._watch_df.append(dict(
            source_name=name, ra=ra, dec=dec,
            patch_type=patch_type), ignore_index=True)
        if watch_mode != 'continuous':
            self._add_timer(f'{id}', watch_until)

    def add_voevent_and_watch(self, voevent):
        raise NotImplementedError(
            'External VOEvent handler not implemented yet')

    def add_source_and_watch(self, source_name, ra, dec,
                             watch_mode='continuous',
                             patch_type='3x3', reason='Detection of FRBs',
                             author='batman', event_type='FRB followup',
                             event_time=datetime.now(), t_start=datetime.now(),
                             t_end=datetime.now()+timedelta(604800)):
        raise NotImplementedError(
            'Manual source watching is not Implemented yet')

    def get_list(self):
        return \
            self._watch_df[['source_name',
                            'skycoord', 'patch_type']]

    def _flatten_series(self, series):
        return list(chain.from_iterable(series))

    def _remove_outside_sky_sources(self, df: pd.DataFrame, pos_column: str):
        return df[~(df[pos_column].apply(lambda x: np.isnan(x).any()))]

    def _remove_outside_sky_patches(self, df: pd.DataFrame,
                                    src_column: str, pos_column: str):
        outside_sources = df[(
            df[pos_column].apply(
                lambda x: np.isnan(x).any()))][src_column].unique()

        return df[~(df[src_column].isin(outside_sources))]

    def get_watch_indices(self, header_str: str, img_axes=[1, 2]):
        header = Header.fromstring(header_str)
        wcs = WCS(header, naxis=img_axes)
        sources = self._watch_df['source_name']
        patch_types = self._watch_df['patch_type']

        # drop any sources outside the sky
        pixels = wcs.all_world2pix(self._watch_df[['ra', 'dec']].to_numpy(), 1)

        source_pixel_df = pd.DataFrame.from_dict(
            dict(pixel=pixels.tolist(),
                 source=sources,
                 patch_type=patch_types))

        source_pixel_df = self._remove_outside_sky_sources(
            source_pixel_df, 'pixel')

        # generate pixel patches for each source
        # each patch cell contains a list of patch pixels (x or y)
        source_pixel_df[['xpatch', 'ypatch']] = source_pixel_df.apply(
            lambda x: pd.Series([
                PatchMan.get_patch_pixels(x['pixel'], x['patch_type'])[
                    0].tolist(),
                PatchMan.get_patch_pixels(
                    x['pixel'], x['patch_type'])[1].tolist()
            ], index=['xpatch', 'ypatch']),
            axis=1)

        source_pixel_df['patch_name'] = source_pixel_df.apply(
            lambda x: [x['source']] * len(x['xpatch']), axis=1)

        # expand pixel patches into individual rows
        source_patch_df = pd.DataFrame(
            dict(
                xpatch=self._flatten_series(source_pixel_df['xpatch']),
                ypatch=self._flatten_series(source_pixel_df['ypatch']),
                patch_name=self._flatten_series(source_pixel_df['patch_name'])
            )
        )

        # filter out sources whose patches fall outside the sky
        # at least partially
        source_patch_df['patch_pixels'] = source_patch_df.apply(
            lambda x: tuple([round(x['xpatch']), round(x['ypatch'])]), axis=1
        )

        source_patch_df['patch_skypos'] = \
            wcs.all_pix2world(
                np.stack(source_patch_df['patch_pixels'].tolist(),
                         axis=0), 1).tolist()

        source_patch_df = self._remove_outside_sky_patches(
            source_patch_df, 'patch_name', 'patch_skypos',)

        # aggregate common pixels
        source_patch_df = source_patch_df.groupby(
            ['patch_pixels']
        ).agg(patch_name=('patch_name', list),
              patch_skypos=('patch_skypos', 'first'))

        source_patch_df.reset_index()
        source_patch_df['patch_pixels'] = \
            source_patch_df.index.get_level_values(0)

        # columns patch_name <list of sources>, patch_pixel, patch_skypos
        return source_patch_df

    def _remove_source(self, id: str):
        self._watch_df.drop(
            (self._watch_df[self._watch_df['id'] == id]).index, inplace=True)
        self._service_Hub.query(
            'UPDATE epic_watchdog SET watch_status=%s', 'watched')

    def _add_watch_timer(self, id: str, date: datetime) -> None:
        self._service_Hub.schedule_job_date(
            self._remove_source, args=(id), timestamp=date)

    def _load_sources(self, overwrite=False):
        if self._watch_df.shape[0] > 0 and overwrite is False:
            raise Exception(
                'Sources are already being watched and overwrite is \
                    set to false')
        else:
            for source in self._watch_df:
                source['job'].remove()

        sources = self._service_Hub.query(
            "SELECT id,source,ST_X(skypos),ST_Y(skypos),t_end, watch_mode,\
                 patch_type from \
                 epic_watchdog where watch_status=%s", 'watching')

        self._watch_df = self._watch_df.head(0)
        for source in sources:
            self.watch_source(id=source[0], name=source[1],
                              ra=source[2], dec=source[3],
                              watch_until=source[4], watch_mode=source[5],
                              patch_type=source[6])
