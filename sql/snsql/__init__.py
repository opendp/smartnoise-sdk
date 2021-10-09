from .sql.private_reader import PrivateReader
from .sql.privacy import Privacy

def from_df(conn, *ignore, privacy, metadata, engine='pandas', **kwargs):
    if engine != 'pandas':
        raise ValueError("from_df requires engine='pandas'")
    return PrivateReader.from_connection(conn, privacy=privacy, metadata=metadata, engine=engine)

def from_connection(conn, *ignore, privacy, metadata, engine=None, **kwargs):
    return PrivateReader.from_connection(conn, privacy=privacy, metadata=metadata, engine=engine)

__all__ = ['Privacy', 'from_df', 'from_connection']