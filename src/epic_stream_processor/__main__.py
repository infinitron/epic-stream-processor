"""Command-line interface."""
from typing import Optional
from typing import Union

import click
from click import Context


@click.group()
@click.version_option()
# @click.option("--start", "-S", is_flag=True, help="Start the stream processor")
@click.pass_context
def main(ctx: Context) -> None:
    """Epic Stream Processor."""
    pass


# @click.command()
# @click.version_option()
# def main():
#     """Epic Stream Processor."""


@main.command()
@click.option("-d", "--db-name", default="epic_db", help="DB name")
@click.option("-p", "--port", default="5432", help="DB port", type=int)
@click.option("-U", "--db-user", default="epic", help="DB user")
@click.option(
    "-DB",
    "--db-type",
    default="postgresql",
    help="DB Type",
    type=click.Choice(["postgresql"]),
)
@click.option(
    "-h",
    "--db-host",
    default="/var/run/postgresql",
    help="DB host (use sockets for local connections)",
)
@click.option(
    "--password",
    prompt=False,
    hide_input=True,
    default=None,
    help="DB password. Defaults to peer authentication if not set",
    required=False,
)
def start(
    db_name: str, port: int, db_user: str, db_type: str, db_host: str, password: str
) -> None:
    """Start the epic stream processing server"""
    from sqlalchemy import create_engine

    from epic_stream_processor._utils.Utils import get_thread_UDS_addr
    from epic_stream_processor.epic_services.uds_server import ThreadedServer

    # import os

    conn_str = f"{db_type}:///?"
    pars = [db_name, port, db_user, db_host]
    par_names = ["database", "port", "user", "host"]
    conn_pars = []

    for name, par in zip(par_names, pars):
        conn_pars.append(f"{name}={par}")

    if password is not None:
        conn_pars.append(f"password={password}")

    conn_str = conn_str + "&".join(conn_pars)

    click.echo("Starting...")
    addr = get_thread_UDS_addr()
    engine = create_engine(conn_str)
    print(f"Listening on UDS: {addr}")
    # os.nice(10)  # unsure if this will help remove the startup lag
    ThreadedServer(addr, engine=engine).listen()


@main.command()
@click.option("-n", "--source-name", help="Name of the source", required=True)
@click.option(
    "-ra", "--right-ascension", help="Right ascension (J2000, ICRS)", required=True
)
@click.option("-dec", "--declination", help="Declination (J2000, ICRS)", required=True)
@click.option(
    "-fmt",
    "--skypos-format",
    help="Format of the specified sky coordinates. Defaults to decimal degrees",
    type=click.Choice(["deg", "hms"]),
    default="deg",
)
@click.option(
    "-wm",
    "--watch-mode",
    help="Watching mode for the source. Defaults to timed (1 week)",
    type=click.Choice(["continuous", "timed"]),
    required=False,
    default="timed",
)
@click.option(
    "-pt",
    "--patch-type",
    help="Size of the aggregation window (nxn). Defaults to 3x3",
    default="3x3",
)
@click.option(
    "-R",
    "--reason",
    help="Short description for the reason to monitor this source (e.g., Detect FRBs)",
    required=True,
)
@click.option("-A", "--author", help="Author's name", required=True)
@click.option(
    "-wb",
    "--watch-begin",
    help="UTC time to begin watching the source. Defaults to now.\
         Use the format YYYY-MM-DDThh:mm:ss.ssss.",
    default=None,
)
# @click.option(
#     "-we",
#     "--watch-end",
#     help="UTC time to stop watching the source. Defaults to one week from now. Use the format YYYY-MM-DDThh:mm:ss.ss",
#     default=None
# )
@click.option(
    "-wd",
    "--watch-duration",
    default=None,
    help="Duration of the watch. Defaults to one week. Provide duration in human readable form. For example,  as '6d23h59m59s9ms1us' or '7d'. This value is not used when watch_mode is set to continuous."
    #  But why microseconds? Because Batman wants to.
)
def watch(
    source_name: str,
    right_ascension: Union[float, str],
    declination: Union[float, str],
    skypos_format: str,
    watch_mode: str,
    patch_type: str,
    reason: str,
    author: str,
    watch_begin: Optional[str],
    watch_duration: Optional[str],
) -> None:
    """Watch for the specified source in the incoming data.

    Any pixel data falling within the image FOV will be stored
    in a pgDB. For solar system sources specify the ra and dec as 0, 0.
    """
    from datetime import timedelta

    from epic_stream_processor.epic_services.uds_client import send_man_watch_req

    # format the sky coordinates
    if skypos_format == "hms":
        from astropy.coordinates import SkyCoord

        skypos = SkyCoord(
            right_ascension, declination, frame="icrs", unit=("hourangle", "deg")
        )
        right_ascension = skypos.ra.value
        declination = skypos.dec.value

    if watch_begin is None:
        from datetime import datetime

        t_start = datetime.utcnow()

    else:
        from datetime import datetime

        t_start = datetime.fromisoformat(watch_begin)

    if watch_duration is None:
        t_end = t_start + timedelta(days=7)
    else:
        import humanreadable as hr

        t_end = t_start + timedelta(
            days=hr.Time(watch_duration).days,
            hours=hr.Time(watch_duration).hours,
            minutes=hr.Time(watch_duration).minutes,
            seconds=hr.Time(watch_duration).seconds,
            milliseconds=hr.Time(watch_duration).milliseconds,
            microseconds=hr.Time(watch_duration).microseconds,
        )

    if watch_mode == "continuous":
        t_end = t_start + timedelta(days=99 * 365.25)

    resp = send_man_watch_req(
        source_name=source_name,
        ra=float(right_ascension),
        dec=float(declination),
        author=author,
        t_start=t_start,
        t_end=t_end,
        reason=reason,
        watch_mode=watch_mode,
        patch_type=patch_type,
    )

    if resp == "added":
        print(f"{source_name} is added to the watchlist.")
    else:
        print(resp)


@main.command()
@click.option(
    "-addr",
    "--addr",
    default="239.168.40.14",
    help="Address to fetch F-engines packets from",
)
@click.option("-p", "--port", default="4016", help="F-engine packet port")
@click.option(
    "-outd",
    "--out_dir",
    default=None,
    help="Output directory to store the images",
    required=True,
)
@click.option(
    "-c",
    "--cores",
    default="11,12,13,14,15,16",
    help="list of CPU cores to bind epic to",
    required=True,
)
@click.option(
    "-g", "--gpu", default="1", help="ID of GPU to bind epic to", required=True
)
@click.option(
    "-chns",
    "--channels",
    default=22,
    help="Number of channels to process. If the number of channels divides 132,\
         the channels cover the entire bandwidth.",
)
@click.option("-imsize", "--imagesize", default=90, help="Size of the output image")
@click.option("-imres", "--imageres", default=1.444, help="Image resolution in degrees")
@click.option("-accum", "--accumulate", default=96, help="Image accumulation time")
@click.option("-nts", "--nts", default=800, help="Number of images in a gulp")
@click.option(
    "-singlepol", "--singlepol", default=False, help="Process only X-pol data"
)
@click.option("-dur", "--duration", default=10, help="Duration of the run")
def run_epic(
    addr,
    port,
    out_dir,
    cores,
    gpu,
    channels,
    imagesize,
    imageres,
    accumulate,
    nts,
    singlepol,
    duration,
):
    """
    Start an instance of EPIC imager.

    """
    args = locals()
    from epic_stream_processor.epic_services.uds_client import req_epic_run

    resp = req_epic_run(**args)
    print(resp)


@main.command()
def get_epic_instances():
    from epic_stream_processor.epic_services.uds_client import req_epic_instances
    import json

    instances = req_epic_instances()
    instances = json.loads(instances)
    if len(instances) == 0:
        print("No instances running")
    else:
        print(f"Running {len(instances)} EPIC instances.")
    for i, inst in enumerate(instances):
        print(f"Instance {i+1}")
        print(f"{inst['options']}")
        print(f"PID: {inst['pid']}")


# if __name__ == "__main__":
#    main(prog_name="epic-stream-processor", obj={})  # pragma: no cover
