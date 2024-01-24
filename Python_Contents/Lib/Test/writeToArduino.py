#importing package from pySerial library
import serial.tools.list_ports

serialInst = serial.Serial()    #creating an empty instance of the Serial
serialInst.baudrate = 9600      #defining the baud rate
serialInst.port = "COM3"        #connecting to the port COM3
serialInst.open()               #opening the connection

while True:                     #creating an infinite loop
    command = input("Arduino Command (ON/OFF):")
    serialInst.write(command.encode('utf-8'))

    if command == 'exit':
        exit()
