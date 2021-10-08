SELECT OEM, TrialGroup, COUNT(DeviceID) AS N, SUM(Crashes) FROM (
       SELECT Crashes, T.* FROM Telemetry.Crashes JOIN ( 
        SELECT DeviceID, OEM, TrialGroup FROM 
         (Telemetry.Census JOIN Telemetry.Rollouts USING(DeviceID)) AS Q
             ) AS T USING(DeviceID) 
      ) AS Z GROUP BY OEM, TrialGroup ORDER BY OEM, TrialGroup;
SELECT  TrialGroup, COUNT(DeviceID) AS N, AVG(Temperature) FROM (
       SELECT TC.*, T.TrialGroup, T.OEM FROM Telemetry.Crashes AS TC JOIN ( 
        SELECT DeviceID, OEM, TrialGroup FROM 
         (Telemetry.Census JOIN Telemetry.Rollouts USING(DeviceID)) AS Q
             ) AS T USING(DeviceID) 
      ) AS Z GROUP BY  TrialGroup ORDER BY  TrialGroup;
SELECT OEM, AVG(Temperature) FROM Telemetry.Census JOIN Telemetry.Crashes USING(DeviceID) GROUP BY OEM ORDER BY OEM;
SELECT OEM, AVG(Temperature) FROM Telemetry.Census JOIN Telemetry.Crashes USING(DeviceID) GROUP BY OEM ORDER BY OEM;
--SELECT SUM(Crashes), TrialGroup FROM Telemetry.Census JOIN Telemetry.Crashes USING(DeviceID) JOIN Telemetry.Rollouts USING(DeviceID) GROUP BY TrialGroup ORDER BY TrialGroup;
