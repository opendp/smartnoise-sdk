import numpy as np
import datetime
from snsynth.transform.definitions import ColumnType
from .base import ColumnTransformer

class DateTimeTransformer(ColumnTransformer):
    """Converts a datetime column into an float column with number of days since the epoch.

    :param epoch: The epoch to use.  If None, the standard epoch of Jan 1, 1970 will be used.
    """
    def __init__(self, epoch=None):
        super().__init__()
        if epoch is None:
            self.epoch = datetime.datetime(1970, 1, 1)
        elif isinstance(epoch, str):
            self.epoch = None
            epoch = self._parse_date(epoch)
            if epoch is None:
                raise ValueError(f"Invalid epoch: {epoch}")
            self.epoch = epoch
        elif isinstance(epoch, datetime.date):
            self.epoch = datetime.datetime(epoch.year, epoch.month, epoch.day)
        elif isinstance(epoch, datetime.datetime):
            self.epoch = epoch
        else:
            raise ValueError(f"Invalid epoch: {epoch}.")
        self.format = "datetime" # can also be date or time, will be inferred from data
        self.string = False
    @property
    def output_type(self):
        return ColumnType.CONTINUOUS
    @property
    def cardinality(self):
        return [None]
    def _fit(self, val, idx=None):
        pass
    def _clear_fit(self):
        # this transform doesn't need fit
        self._fit_complete = True
        self.output_width = 1
    def _transform(self, val):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return np.nan
        else:
            parsed = self._parse_date(val)
            if parsed is None:
                raise ValueError(f"Invalid date: {val}")
            distance = parsed - self.epoch
            return float(distance.total_seconds() / (60 * 60 * 24))
    def _inverse_transform(self, val):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        else:
            new_date = self.epoch + datetime.timedelta(days=val)
            if self.string:
                if self.format == "datetime":
                    return new_date.isoformat()
                elif self.format == "date":
                    return new_date.date().isoformat()
                elif self.format == "time":
                    return new_date.time().isoformat()
            else:
                if self.format == "datetime":
                    return new_date
                elif self.format == "date":
                    return new_date.date()
                elif self.format == "time":
                    return new_date.time()
                else:
                    raise ValueError(f"Invalid format: {self.format}")
    def _parse_date(self, val):
        if val is None:
            return None
        if isinstance(val, datetime.datetime):
            self.format = "datetime"
            self.string = False
            return val
        elif isinstance(val, datetime.date):
            self.format = "date"
            self.string = False
            return datetime.datetime(val.year, val.month, val.day)
        elif isinstance(val, datetime.time):
            self.format = "time"
            self.string = False
            return datetime.datetime.combine(self.epoch, val)
        elif isinstance(val, str):
            self.string = True
            try:
                parsed = datetime.datetime.strptime(val, "%Y-%m-%d")
                self.format = "date"
                return parsed
            except:
                pass
            try:
                parsed = datetime.datetime.strptime(val, "%H:%M:%S").time()
                self.format = "time"
                if self.epoch is None:
                    self.epoch = datetime.datetime(1970, 1, 1)
                return datetime.datetime.combine(self.epoch, parsed)
            except:
                pass
            try:
                parsed = datetime.datetime.strptime(val, "%H:%M:%S.%f").time()
                self.format = "time"
                if self.epoch is None:
                    self.epoch = datetime.datetime(1970, 1, 1)
                return datetime.datetime.combine(self.epoch, parsed)
            except:
                pass
            try:
                parsed = datetime.datetime.strptime(val, "%I:%M:%S %p").time()
                self.format = "time"
                if self.epoch is None:
                    self.epoch = datetime.datetime(1970, 1, 1)
                return datetime.datetime.combine(self.epoch, parsed)
            except:
                pass
            try:
                parsed = datetime.datetime.strptime(val, "%I:%M:%S.%f %p").time()
                self.format = "time"
                if self.epoch is None:
                    self.epoch = datetime.datetime(1970, 1, 1)
                return datetime.datetime.combine(self.epoch, parsed)
            except:
                pass
            try:
                parsed = datetime.datetime.fromisoformat(val)
                self.format = "datetime"
                return parsed
            except:
                pass
            try:
                parsed = datetime.datetime.strptime(val, "%Y-%m-%d %H:%M:%S")
                self.format = "datetime"
                return parsed
            except:
                pass
            try:
                parsed = datetime.datetime.strptime(val, "%Y-%m-%d %H:%M:%S.%f")
                self.format = "datetime"
                return parsed
            except:
                pass
            try:
                parsed = datetime.datetime.strptime(val, "%Y-%m-%d %I:%M:%S %p")
                self.format = "datetime"
                return parsed
            except:
                pass
            try:
                parsed = datetime.datetime.strptime(val, "%Y-%m-%d %I:%M:%S.%f %p")
                self.format = "datetime"
                return parsed
            except:
                pass
            try:
                parsed = datetime.datetime.strptime(val, "%Y-%m-%dT%H:%M:%S")
                self.format = "datetime"
                return parsed
            except:
                pass
            try:
                parsed = datetime.datetime.strptime(val, "%Y-%m-%dT%H:%M:%S.%f")
                self.format = "datetime"
                return parsed
            except:
                pass
            try:
                parsed = datetime.datetime.strptime(val, "%Y-%m-%dT%H:%M:%S%z")
                self.format = "datetime"
                return parsed
            except:
                pass
            try:
                parsed = datetime.datetime.strptime(val, "%Y-%m-%dT%H:%M:%S.%f%z")
                self.format = "datetime"
                return parsed
            except:
                pass
            return None
        else:
            return None
