from pytest_pgsql import PostgreSQLTestDB

from epic_stream_processor.epic_orm.pg_pixel_storage import Base
from epic_stream_processor.epic_orm.pg_pixel_storage import Database


def test_get_watch_list(postgresql_db: PostgreSQLTestDB) -> None:
    # pass
    # if postgresql_db.is_extension_available('postgis') == True:
    #     print('installing postgis')
    #     postgresql_db.install_extension('postgis')
    db = Database(postgresql_db.engine)
    db.create_all_tables()
    df = db.list_watch_sources()
    print(df, Base.metadata.tables)
