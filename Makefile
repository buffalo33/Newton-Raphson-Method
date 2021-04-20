SRC=src

all: bairstow electrostatic forces jacob newton_raphson

bairstow:
	python3 src/Bairstow.py

electrostatic:
	python3 src/electrostatic.py

forces:
	python3 src/forces.py

jacob:
	python3 src/jacob.py

newton_raphson:
	python3 src/Newton_Raphson.py