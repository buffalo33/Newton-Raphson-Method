Newton-Raphson Method
=====================

Abstract
--------
This project is based on the development of an algorithm that can search the roots of a non-linear equation system. The method used is the Newton-Raphson algorithm and two examples were done to test this algorithm, the computation of the Lagrangian points and the search of electrostatics equilibrium. Another algorithm was implemented to factorize polynomials with the Bairstow method. This project was developed using Python. The authors are Hugo PIERRE, Ibrahim LMOURID, Yassine MOUGOU, Saad ABIDI and Clément PREAUT.

Theoretical explanations
----------------------
A detailed report available in `report/report.pdf` explains the theory used in programs, what cases were tested, and interpretations concerning the results.

Instructions
------------
* `make jacob` : Computes the Jacobian Matrix on a non-linear function and displays results on a graph showing the relative gap between theoretical and computed Jacobian matrix.
* `make newton_raphson` : Computes three graphs using different parameters to test the Newton-Raphson algorithm with and without backtracking.
* `make forces` : uses Newton_Raphson algorithm to obtain the 5 Lagrangian points of an object moving in the plane and subject to gravitational forces. It displays the results on a graph.
* `make electrostatic` : Computes the electrostatic equilibrium of an electric charge system and computes the Energy of one charge according to its position. It displays the results on a graph.
* `make bairstow` : Uses Bairstrow method to compute roots of polynomials that does not have only real roots. It displays a graph showing the result differences between our algorithm and the Python standard one, according to the polynomial's degree.
* `make all` : Executes all the previous commands.