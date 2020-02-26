from os.path import dirname, join

from burdock.query.sql import CollectionMetadata, QueryParser

dir_name = dirname(__file__)

metadata = CollectionMetadata.from_file(join(dir_name, "Devices.yaml"))

def qp(query_string):
    return QueryParser().query(query_string)

#
#   Unit tests
#
class TestTypes:

    def test_s12(self):
            q = qp("SELECT Refurbished FROM Telemetry.Crashes;")
            q.load_symbols(metadata)
            print(str(q["Refurbished"]))
            assert q["Refurbished"].type() == "boolean"
            assert q["Refurbished"].sensitivity() == 1

    def test_s13(self):
            q = qp("SELECT * FROM Telemetry.Crashes;")
            q.load_symbols(metadata)
            assert q["Refurbished"].type() == "boolean"
            assert q["Refurbished"].sensitivity() == 1
            assert q["Temperature"].sensitivity() == 65.0

    def test_s20(self):
            q = qp("SELECT 7 * Temperature, Crashes * 101.11 FROM Telemetry.Crashes;")
            q.load_symbols(metadata)
            assert len(list(q.m_sym_dict.keys()) ) == 0
            assert q.m_symbols[0][1].type() == "float"
            assert q.m_symbols[1][1].type() == "float"
            assert q.m_symbols[1][1].sensitivity() > 1010
            assert q.m_symbols[0][1].sensitivity() == 455

    def test_s23(self):
        q = qp("SELECT SUM(C.Crashes) AS Crashes FROM (Telemetry.Crashes) AS C;")
        q.load_symbols(metadata)
        assert q["Crashes"].type() == "int"
        assert q["Crashes"].sensitivity() == 10

    def test_s34(self):
        q = qp("SELECT OEM, Crashes FROM (Telemetry.Rollouts INNER JOIN Telemetry.Census INNER JOIN Telemetry.Crashes USING(DeviceID));")
        q.load_symbols(metadata)
        assert q["Crashes"].type() == "int"
        assert q["OEM"].type() == "string"

    def test_s45(self):
        q = qp("SELECT Temperature FROM (SELECT * FROM (SELECT Temperature, Crashes FROM (Telemetry.Census JOIN Telemetry.Crashes USING(DeviceID))));")
        q.load_symbols(metadata)
        assert q["Temperature"].type() == "float"

    def test_s46(self):
        q = qp("SELECT Temperature FROM (SELECT * FROM (SELECT * FROM (SELECT Temperature, Crashes FROM Telemetry.Census JOIN Telemetry.Crashes USING(DeviceID))));")
        q.load_symbols(metadata)
        assert q["Temperature"].type() == "float"
        assert q["Temperature"].sensitivity() == 65

    def test_s47(self):
        q = qp("SELECT (1 + Temperature) AS Temperature FROM (SELECT * FROM (SELECT * FROM (SELECT Temperature, Crashes FROM Telemetry.Census JOIN Telemetry.Crashes USING(DeviceID))));")
        q.load_symbols(metadata)
        assert q["Temperature"].type() == "float"
        assert q["Temperature"].sensitivity() == 65

    def test_s48(self):
        q = qp("SELECT (3 * COUNT(Temperature)) AS Temperature FROM (SELECT * FROM (SELECT * FROM (SELECT Temperature, Crashes FROM Telemetry.Census JOIN Telemetry.Crashes USING(DeviceID))));")
        q.load_symbols(metadata)
        assert q["Temperature"].type() == "int"
        assert q["Temperature"].sensitivity() == 3

    def test_s49(self):
        q = qp("SELECT COUNT(Temperature) / 7 AS Temperature FROM (SELECT * FROM (SELECT * FROM (SELECT Temperature, Crashes FROM Telemetry.Census JOIN Telemetry.Crashes USING(DeviceID))));")
        q.load_symbols(metadata)
        assert q["Temperature"].type() == "float"
        assert q["Temperature"].sensitivity() == 1.0 / 7.0
        