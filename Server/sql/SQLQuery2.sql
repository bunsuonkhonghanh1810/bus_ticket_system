USE BUS_TICKETS_MANAGEMENT

SELECT * FROM PASSENGERS

SELECT * FROM TICKETS

SELECT * FROM TICKETCLASS

SELECT * FROM BUSENTRY

SELECT * FROM TICKETCLASS WHERE TicketType = 'Single'

DELETE FROM PASSENGERS WHERE PassengerId = '93F80A9C-7DDC-4CE7-9474-F651FC9D163B'

DELETE FROM TICKETS WHERE TicketId = '202455B1-752C-4B1E-859B-D4853C363D38'

SELECT * FROM TICKETS JOIN TICKETCLASS ON TICKETS.TicketClassId = TICKETCLASS.TicketClassId WHERE PassengerId = '47235974-60AD-426F-857A-165FD41FCD3A' AND TicketType = 'Single' AND TicketState = 'Unused' ORDER BY PurchaseTime ASC