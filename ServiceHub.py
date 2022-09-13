import psycopg
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timezone, timedelta


class ServiceHub(object):
    _pg_conn = None
    _scheduler = BackgroundScheduler(timezone=timezone.utc)

    def __init__(self) -> None:
        """connect to ectd and and also have a scheduler from apscheduler"""
        self._scheduler.start()
        self._connect_pgdb()

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(ServiceHub, cls).__new__(cls)
        return cls.instance

    def _connect_pgdb(self) -> None:
        # establish postgres connection
        # TODO: fetch the connection details from sys config service
        pg_conn_params = dict(
            host="localhost", database="postgres", user="postgres")
        try:
            self._pg_conn = psycopg.connect(**pg_conn_params)
            # self._pg_cursor = self._pg_conn.cursor()
        except Exception as e:
            print(e)
            print('Retrying connection in 60 seconds')
            self.schedule_job_delay(self._connect_pgdb, (self), 60)

    def pg_query(self, query, args):
        if self._pg_conn is None:
            raise (ConnectionRefusedError(
                "Could not establish connection to the pgdb"))
        else:
            result = self._pg_conn.execute(query, args).fetchall()
            self._pg_conn.commit()
            return result

    def schedule_job_delay(self, func, args=None, seconds=604800):
        self._scheduler.add_job(
            func, trigger='date', args=args,
            run_date=datetime.now()+timedelta(seconds))

    def schedule_job_date(self, func, args=None, timestamp=datetime.now()):
        self._scheduler.add_job(
            func, trigger='date', args=args,
            run_date=timestamp)
