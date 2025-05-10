import pyodbc

def connect_db():
    server = '10.90.103.36'  # IP của máy chủ
    database = 'BUS_TICKETS_MANAGEMENT'
    username = 'bus_management'
    password = '123456'

    conn_str = (
        f'DRIVER={{SQL Server}};'
        f'SERVER={server};'
        f'DATABASE={database};'
        f'UID={username};'
        f'PWD={password};'
        f'Connect Timeout=30'
    )

    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        return conn, cursor
    except Exception as e:
        print("Lỗi khi kết nối SQL Server:", str(e))
        return None, None

def close_db(conn, cursor):
    if cursor:
        cursor.close()
    if conn:
        conn.close()