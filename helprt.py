import motors
import sqlite3
motors = motors.Motors(400)

motors.update(150, 250)
for a in range(200):
    motors.main()

for a in range(200):
    motors.main()
