-- Синтетический пример SCHEDULE-файла для тестов eclipse_schedule.
-- Структура (порядок и набор ключевых слов) повторяет реальный экспорт
-- Petrel, но скважины/даты/дебиты выдуманы и малы по объёму.

RPTSCHED                               -- Generated : Petrel
  FIP WELLS /

RPTRST                                 -- Generated : Petrel
  BASIC=3 FLOWS FREQ /

SKIP                                   -- Generated : Petrel
--Default history strategy generated.

ENDSKIP                                -- Generated : Petrel

WELSPECS                               -- Generated : Petrel
--'A1' is the simulation well name used to describe flow from 'A1'
--'A2' is the simulation well name used to describe flow from 'A2'
--
  'A1' 'GROUP 1' 10 20 1* OIL /
  'A2' 'GROUP 1' 12 22 1* OIL /
  /

WLIST                                  -- Generated : Petrel
  '*WELLS' NEW
  'A1' 'A2' /
  /

COMPDAT                                -- Generated : Petrel
  'A1' 10 20 100 101 OPEN 1* 5.0 0.168 400.0 0 1* Z 5.0 /
  'A2' 12 22 105 106 OPEN 1* 6.0 0.168 450.0 0 1* Z 5.1 /
  /

WCONHIST                               -- Generated : Petrel
  'A1' OPEN LRAT 50.0 0 0 3* 0 /
  'A2' OPEN LRAT 0.114943 0 0 3* 0 /
  /

TUNING                                 -- Generated : Petrel
2* 0.000001 /
/
/

DATES                                  -- Generated : Petrel
  1 FEB 1980 /
  /

WELSPECS                               -- Generated : Petrel
--'A3' is the simulation well name used to describe flow from 'A3'
--
  'A3' 'GROUP 1' 15 25 1* OIL /
  /

WLIST                                  -- Generated : Petrel
  '*WELLS' ADD
  'A3' /
  /

COMPDAT                                -- Generated : Petrel
  'A3' 15 25 110 111 OPEN 1* 4.5 0.168 380.0 0 1* Z 5.0 /
  /

WCONHIST                               -- Generated : Petrel
  'A1' OPEN LRAT 55.0 1.0 0 3* 0 /
  'A2' OPEN LRAT 0.114943 0 0 3* 0 /
  'A3' OPEN LRAT 0.114943 0 0 3* 0 /
  /

GRUPTREE                               -- Generated : Petrel
  'GROUP 1' FIELD /
  /

FBHPDEF                                -- Generated : Petrel
5 /

DATES                                  -- Generated : Petrel
  1 MAR 1980 /
  /

WCONHIST                               -- Generated : Petrel
  'A1' OPEN LRAT 60.0 3.0 0 3* 0 /
  'A2' OPEN LRAT 20.0 0.5 0 3* 0 /
  'A3' OPEN LRAT 18.0 0.2 0 3* 0 /
  /

WCONINJH                               -- Generated : Petrel
  'A2' WATER OPEN 25.0 0 6* RATE /
  /

DATES                                  -- Generated : Petrel
  1 APR 1980 /
  /

WCONHIST                               -- Generated : Petrel
  'A1' OPEN LRAT 58.0 4.0 0 3* 0 /
  'A2' OPEN LRAT 22.0 0.8 0 3* 0 /
  'A3' OPEN LRAT 19.0 0.3 0 3* 0 /
  /

WCONINJH                               -- Generated : Petrel
  'A2' WATER OPEN 30.0 0 6* RATE /
  /
