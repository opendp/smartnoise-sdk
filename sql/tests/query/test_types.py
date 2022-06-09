from os.path import dirname, join

from snsql.metadata import Metadata
from snsql.sql.parse import QueryParser

dir_name = dirname(__file__)

metadata = Metadata.from_file(join(dir_name, "Devices.yaml"))

def qp(query_string):
    return QueryParser().query(query_string)

#
#   Unit tests
#
class TestTypes:

    def test_s12(self):
            q = qp("SELECT Refurbished FROM Telemetry.Crashes;")
            q.load_symbols(metadata)
            assert q["Refurbished"].expression.type() == "boolean"
            assert q["Refurbished"].expression.sensitivity() == 1

    def test_s13(self):
            q = qp("SELECT * FROM Telemetry.Crashes;")
            q.load_symbols(metadata)
            assert q["Refurbished"].expression.type() == "boolean"
            assert q["Refurbished"].expression.sensitivity() == 1
            assert q["Temperature"].expression.sensitivity() == 65.0

    def test_s20(self):
            q = qp("SELECT 7 * Temperature, Crashes * 101.11 FROM Telemetry.Crashes;")
            q.load_symbols(metadata)
            assert len(list(q._named_symbols.keys()) ) == 0
            assert q._select_symbols[0].expression.type() == "float"
            assert q._select_symbols[1].expression.type() == "float"
            assert q._select_symbols[1].expression.sensitivity() > 1010
            assert q._select_symbols[0].expression.sensitivity() == 455

    def test_s23(self):
        q = qp("SELECT SUM(C.Crashes) AS Crashes FROM (Telemetry.Crashes) AS C;")
        q.load_symbols(metadata)
        assert q["Crashes"].expression.type() == "int"
        assert q["Crashes"].expression.sensitivity() == 10

    def test_s34(self):
        q = qp("SELECT OEM, Crashes FROM (Telemetry.Rollouts INNER JOIN Telemetry.Census INNER JOIN Telemetry.Crashes USING(DeviceID));")
        q.load_symbols(metadata)
        assert q["Crashes"].expression.type() == "int"
        assert q["OEM"].expression.type() == "string"

    def test_s45(self):
        q = qp("SELECT Temperature FROM (SELECT * FROM (SELECT Temperature, Crashes FROM (Telemetry.Census JOIN Telemetry.Crashes USING(DeviceID))));")
        q.load_symbols(metadata)
        assert q["Temperature"].expression.type() == "float"

    def test_s46(self):
        q = qp("SELECT Temperature FROM (SELECT * FROM (SELECT * FROM (SELECT Temperature, Crashes FROM Telemetry.Census JOIN Telemetry.Crashes USING(DeviceID))));")
        q.load_symbols(metadata)
        assert q["Temperature"].expression.type() == "float"
        assert q["Temperature"].expression.sensitivity() == 65

    def test_s47(self):
        q = qp("SELECT (1 + Temperature) AS Temperature FROM (SELECT * FROM (SELECT * FROM (SELECT Temperature, Crashes FROM Telemetry.Census JOIN Telemetry.Crashes USING(DeviceID))));")
        q.load_symbols(metadata)
        assert q["Temperature"].expression.type() == "float"
        assert q["Temperature"].expression.sensitivity() == 65

    def test_s48(self):
        q = qp("SELECT (3 * COUNT(Temperature)) AS Temperature FROM (SELECT * FROM (SELECT * FROM (SELECT Temperature, Crashes FROM Telemetry.Census JOIN Telemetry.Crashes USING(DeviceID))));")
        q.load_symbols(metadata)
        assert q["Temperature"].expression.type() == "int"
        assert q["Temperature"].expression.sensitivity() == 3

    def test_s49(self):
        q = qp("SELECT COUNT(Temperature) / 7 AS Temperature FROM (SELECT * FROM (SELECT * FROM (SELECT Temperature, Crashes FROM Telemetry.Census JOIN Telemetry.Crashes USING(DeviceID))));")
        q.load_symbols(metadata)
        assert q["Temperature"].expression.type() == "float"
        assert q["Temperature"].expression.sensitivity() == 1.0 / 7.0
