from datetime import datetime, timedelta
import pandas as pd
from astropy.wcs import WCS
from astropy.io.fits import Header
from itertools import chain
from Utils import PatchMan


class WatchDog(object):
    """
    Monitors the locations of specified sources on EPIC images.
    """
    _watch_df = pd.DataFrame(
        columns=['id', 'source_name', 'ra', 'dec', 'patch_type'])

    def __init__(self, serviceHub):
        self._service_Hub = serviceHub

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

    def _flatten_series(series):
        return pd.Series(list(chain(series)))

    def get_watch_indices(self, header_str: str, img_axes=[0, 1]):
        header = Header.fromstring(header_str)
        wcs = WCS(header, naxis=img_axes)
        sources = self._watch_df['source_name']
        patch_types = self._watch_df['patch_type']

        # filter any sources outside the sky
        pixels = wcs.all_world2pix(self._watch_df[['ra', 'dec']].to_numpy(), 0)

        source_pixel_df = pd.DataFrame.from_dict(
            dict(pixel=pixels,
                 source=sources,
                 patch_type=patch_types))
        source_pixel_df.dropna(inplace=True)

        # store pixel patches for each source
        source_pixel_df[['xpatch', 'ypatch']] = source_pixel_df.apply(
            lambda x: pd.Series(
                PatchMan.get_patch_pixels(x['pixels'], x['patch_type'])[0],
                PatchMan.get_patch_pixels(x['pixels'], x['patch_type'])[1]
            ),
            axis=1)

        source_pixel_df['patch_name'] = source_pixel_df.apply(
            lambda x: [x['source']] * x['xpatch'].size)

        source_patch_df = pd.DataFrame(
            dict(
                xpatch=self._flatten_series(source_pixel_df['xpatch']),
                ypatch=self._flatten_series(source_pixel_df['ypatch']),
                source_name=self._flatten_series(source_pixel_df['ypatch'])
            )
        )

        # filter out sources whose patches fall outside the sky
        # at least partially
        source_patch_df['patch_pixels'] = source_patch_df.apply(
            lambda x: [x['xpatch'], x['ypatch']], axis=1
        )

        source_patch_df['patch_skypos'] = \
            wcs.all_pix2world(*source_patch_df['patch_pixels'], 1)

        invalid_sources = source_patch_df[
            pd.isna(source_patch_df['patch_skypos'])]['patch_name'].unique()

        source_patch_df = source_patch_df[
            ~(source_patch_df['patch_name'].isin(invalid_sources))]

        # aggregate common pixels and their sources
        source_patch_df = source_patch_df.groupby(
            [['patch_pixels', 'patch_skypos']]
        )['patch_name'].agg(list).to_numpy()

        return source_patch_df['patch_name'], source_patch_df.index.values

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
