"""Command-line interface."""
import click


@click.group()
@click.version_option()
# @click.option("--start", "-S", is_flag=True, help="Start the stream processor")
@click.pass_context
def main(ctx) -> None:
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
def start(db_name, port, db_user, db_type, db_host, password):
    """Start the epic stream processing server"""
    from epic_stream_processor.epic_services.uds_server import ThreadedServer
    from epic_stream_processor._utils.Utils import get_thread_UDS_addr
    from sqlalchemy import create_engine
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
    source_name,
    right_ascension,
    declination,
    skypos_format,
    watch_mode,
    patch_type,
    reason,
    author,
    watch_begin,
    watch_duration,
):
    """Watch for the specified source in the incoming data.

    Any pixel data falling within the image FOV will be stored
    in a pgDB. For solar system sources specify the ra and dec as 0, 0.
    """
    from epic_stream_processor.epic_services.uds_client import send_man_watch_req
    from datetime import timedelta

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

        watch_begin = datetime.utcnow()

    if watch_duration is None:
        watch_end = watch_begin + timedelta(days=7)
    else:
        import humanreadable as hr

        watch_end = watch_begin + timedelta(
            days=hr.Time(watch_duration).days,
            hours=hr.Time(watch_duration).hours,
            minutes=hr.Time(watch_duration).minutes,
            seconds=hr.Time(watch_duration).seconds,
            milliseconds=hr.Time(watch_duration).milliseconds,
            microseconds=hr.Time(watch_duration).microseconds,
        )

    if watch_mode == "continuous":
        watch_end = watch_begin + timedelta(days=99 * 365.25)

    resp = send_man_watch_req(
        source_name=source_name,
        ra=right_ascension,
        dec=declination,
        author=author,
        t_start=watch_begin,
        t_end=watch_end,
        reason=reason,
        watch_mode=watch_mode,
        patch_type=patch_type,
    )

    if resp == "added":
        print(f"{source_name} is added to the watchlist.")
    else:
        print(resp)


# if __name__ == "__main__":
#    main(prog_name="epic-stream-processor", obj={})  # pragma: no cover
